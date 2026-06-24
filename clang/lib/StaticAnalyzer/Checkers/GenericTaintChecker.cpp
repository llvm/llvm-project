//== GenericTaintChecker.cpp ----------------------------------- -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker defines the attack surface for generic taint propagation.
//
// The taint information produced by it might be useful to other checkers. For
// example, checkers should report errors which involve tainted data more
// aggressively, even if the involved symbols are under constrained.
//
//===----------------------------------------------------------------------===//

#include "Yaml.h"
#include "clang/AST/Attr.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Checkers/Taint.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#define DEBUG_TYPE "taint-checker"

using namespace clang;
using namespace ento;
using namespace taint;

using llvm::ImmutableSet;

namespace {

class GenericTaintChecker;

/// Check for CWE-134: Uncontrolled Format String.
constexpr llvm::StringLiteral MsgUncontrolledFormatString =
    "Untrusted data is used as a format string "
    "(CWE-134: Uncontrolled Format String)";

/// Check for:
/// CERT/STR02-C. "Sanitize data passed to complex subsystems"
/// CWE-78, "Failure to Sanitize Data into an OS Command"
constexpr llvm::StringLiteral MsgSanitizeSystemArgs =
    "Untrusted data is passed to a system call "
    "(CERT/STR02-C. Sanitize data passed to complex subsystems)";

/// Check if tainted data is used as a custom sink's parameter.
constexpr llvm::StringLiteral MsgCustomSink =
    "Untrusted data is passed to a user-defined sink";

using ArgIdxTy = int;
using ArgVecTy = llvm::SmallVector<ArgIdxTy, 2>;

/// Denotes the return value.
constexpr ArgIdxTy ReturnValueIndex{-1};

static ArgIdxTy fromArgumentCount(unsigned Count) {
  assert(Count <=
             static_cast<std::size_t>(std::numeric_limits<ArgIdxTy>::max()) &&
         "ArgIdxTy is not large enough to represent the number of arguments.");
  return Count;
}

/// Check if the region the expression evaluates to is the standard input,
/// and thus, is tainted.
/// FIXME: Move this to Taint.cpp.
bool isStdin(SVal Val, const ASTContext &ACtx) {
  // FIXME: What if Val is NonParamVarRegion?

  // The region should be symbolic, we do not know it's value.
  const auto *SymReg = dyn_cast_or_null<SymbolicRegion>(Val.getAsRegion());
  if (!SymReg)
    return false;

  // Get it's symbol and find the declaration region it's pointing to.
  const auto *DeclReg =
      dyn_cast_or_null<DeclRegion>(SymReg->getSymbol()->getOriginRegion());
  if (!DeclReg)
    return false;

  // This region corresponds to a declaration, find out if it's a global/extern
  // variable named stdin with the proper type.
  if (const auto *D = dyn_cast_or_null<VarDecl>(DeclReg->getDecl())) {
    D = D->getCanonicalDecl();
    if (D->getName() == "stdin" && D->hasExternalStorage() && D->isExternC()) {
      const QualType FILETy = ACtx.getFILEType().getCanonicalType();
      const QualType Ty = D->getType().getCanonicalType();

      if (Ty->isPointerType())
        return Ty->getPointeeType() == FILETy;
    }
  }
  return false;
}

SVal getPointeeOf(ProgramStateRef State, Loc LValue) {
  const QualType ArgTy = LValue.getType(State->getStateManager().getContext());
  if (!ArgTy->isPointerType() || !ArgTy->getPointeeType()->isVoidType())
    return State->getSVal(LValue);

  // Do not dereference void pointers. Treat them as byte pointers instead.
  // FIXME: we might want to consider more than just the first byte.
  return State->getSVal(LValue, State->getStateManager().getContext().CharTy);
}

/// Given a pointer/reference argument, return the value it refers to.
std::optional<SVal> getPointeeOf(ProgramStateRef State, SVal Arg) {
  if (auto LValue = Arg.getAs<Loc>())
    return getPointeeOf(State, *LValue);
  return std::nullopt;
}

/// Given a pointer, return the SVal of its pointee or if it is tainted,
/// otherwise return the pointer's SVal if tainted.
/// Also considers stdin as a taint source.
std::optional<SVal> getTaintedPointeeOrPointer(ProgramStateRef State,
                                               SVal Arg) {
  if (auto Pointee = getPointeeOf(State, Arg))
    if (isTainted(State, *Pointee)) // FIXME: isTainted(...) ? Pointee : None;
      return Pointee;

  if (isTainted(State, Arg))
    return Arg;
  return std::nullopt;
}

bool isTaintedOrPointsToTainted(ProgramStateRef State, SVal ExprSVal) {
  return getTaintedPointeeOrPointer(State, ExprSVal).has_value();
}

/// Helps in printing taint diagnostics.
/// Marks the incoming parameters of a function interesting (to be printed)
/// when the return value, or the outgoing parameters are tainted.
const NoteTag *taintOriginTrackerTag(CheckerContext &C,
                                     std::vector<SymbolRef> TaintedSymbols,
                                     std::vector<ArgIdxTy> TaintedArgs,
                                     const StackFrame *CallSF) {
  return C.getNoteTag([TaintedSymbols = std::move(TaintedSymbols),
                       TaintedArgs = std::move(TaintedArgs),
                       CallSF](PathSensitiveBugReport &BR) -> std::string {
    // We give diagnostics only for taint related reports
    if (!BR.isInteresting(CallSF) ||
        BR.getBugType().getCategory() != categories::TaintedData) {
      return "";
    }
    if (TaintedSymbols.empty())
      return "Taint originated here";

    for (auto Sym : TaintedSymbols) {
      BR.markInteresting(Sym);
    }
    LLVM_DEBUG(for (auto Arg
                    : TaintedArgs) {
      llvm::dbgs() << "Taint Propagated from argument " << Arg + 1 << "\n";
    });
    return "";
  });
}

/// Helps in printing taint diagnostics.
/// Marks the function interesting (to be printed)
/// when the return value, or the outgoing parameters are tainted.
const NoteTag *taintPropagationExplainerTag(
    CheckerContext &C, std::vector<SymbolRef> TaintedSymbols,
    std::vector<ArgIdxTy> TaintedArgs, const StackFrame *CallSF) {
  assert(TaintedSymbols.size() == TaintedArgs.size());
  return C.getNoteTag([TaintedSymbols = std::move(TaintedSymbols),
                       TaintedArgs = std::move(TaintedArgs),
                       CallSF](PathSensitiveBugReport &BR) -> std::string {
    SmallString<256> Msg;
    llvm::raw_svector_ostream Out(Msg);
    // We give diagnostics only for taint related reports
    if (TaintedSymbols.empty() ||
        BR.getBugType().getCategory() != categories::TaintedData) {
      return "";
    }
    int nofTaintedArgs = 0;
    for (auto [Idx, Sym] : llvm::enumerate(TaintedSymbols)) {
      if (BR.isInteresting(Sym)) {
        BR.markInteresting(CallSF);
        if (TaintedArgs[Idx] != ReturnValueIndex) {
          LLVM_DEBUG(llvm::dbgs() << "Taint Propagated to argument "
                                  << TaintedArgs[Idx] + 1 << "\n");
          if (nofTaintedArgs == 0)
            Out << "Taint propagated to the ";
          else
            Out << ", ";
          Out << TaintedArgs[Idx] + 1
              << llvm::getOrdinalSuffix(TaintedArgs[Idx] + 1) << " argument";
          nofTaintedArgs++;
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Taint Propagated to return value.\n");
          Out << "Taint propagated to the return value";
        }
      }
    }
    return std::string(Out.str());
  });
}

/// ArgSet is used to describe arguments relevant for taint detection or
/// taint application. A discrete set of argument indexes and a variadic
/// argument list signified by a starting index are supported.
class ArgSet {
public:
  ArgSet() = default;
  ArgSet(ArgVecTy &&DiscreteArgs,
         std::optional<ArgIdxTy> VariadicIndex = std::nullopt)
      : DiscreteArgs(std::move(DiscreteArgs)),
        VariadicIndex(std::move(VariadicIndex)) {}

  bool contains(ArgIdxTy ArgIdx) const {
    if (llvm::is_contained(DiscreteArgs, ArgIdx))
      return true;

    return VariadicIndex && ArgIdx >= *VariadicIndex;
  }

  bool isEmpty() const { return DiscreteArgs.empty() && !VariadicIndex; }

private:
  ArgVecTy DiscreteArgs;
  std::optional<ArgIdxTy> VariadicIndex;
};

/// A struct used to specify taint propagation rules for a function.
///
/// If any of the possible taint source arguments is tainted, all of the
/// destination arguments should also be tainted. If ReturnValueIndex is added
/// to the dst list, the return value will be tainted.
class GenericTaintRule {
  /// Arguments which are taints sinks and should be checked, and a report
  /// should be emitted if taint reaches these.
  ArgSet SinkArgs;
  /// Arguments which should be sanitized on function return.
  ArgSet FilterArgs;
  /// Arguments which can participate in taint propagation. If any of the
  /// arguments in PropSrcArgs is tainted, all arguments in  PropDstArgs should
  /// be tainted.
  ArgSet PropSrcArgs;
  ArgSet PropDstArgs;

  /// A message that explains why the call is sensitive to taint.
  std::optional<StringRef> SinkMsg;

  GenericTaintRule() = default;

  GenericTaintRule(ArgSet &&Sink, ArgSet &&Filter, ArgSet &&Src, ArgSet &&Dst,
                   std::optional<StringRef> SinkMsg = std::nullopt)
      : SinkArgs(std::move(Sink)), FilterArgs(std::move(Filter)),
        PropSrcArgs(std::move(Src)), PropDstArgs(std::move(Dst)),
        SinkMsg(SinkMsg) {}

public:
  /// Make a rule that reports a warning if taint reaches any of \p FilterArgs
  /// arguments.
  static GenericTaintRule Sink(ArgSet &&SinkArgs,
                               std::optional<StringRef> Msg = std::nullopt) {
    return {std::move(SinkArgs), {}, {}, {}, Msg};
  }

  /// Make a rule that sanitizes all FilterArgs arguments.
  static GenericTaintRule Filter(ArgSet &&FilterArgs) {
    return {{}, std::move(FilterArgs), {}, {}};
  }

  /// Make a rule that unconditionally taints all Args.
  /// If Func is provided, it must also return true for taint to propagate.
  static GenericTaintRule Source(ArgSet &&SourceArgs) {
    return {{}, {}, {}, std::move(SourceArgs)};
  }

  /// Make a rule that taints all PropDstArgs if any of PropSrcArgs is tainted.
  static GenericTaintRule Prop(ArgSet &&SrcArgs, ArgSet &&DstArgs) {
    return {{}, {}, std::move(SrcArgs), std::move(DstArgs)};
  }

  /// Process a function which could either be a taint source, a taint sink, a
  /// taint filter or a taint propagator.
  void process(const GenericTaintChecker &Checker, const CallEvent &Call,
               CheckerContext &C) const;

  /// Handles the resolution of indexes of type ArgIdxTy to Expr*-s.
  static const Expr *GetArgExpr(ArgIdxTy ArgIdx, const CallEvent &Call) {
    return ArgIdx == ReturnValueIndex ? Call.getOriginExpr()
                                      : Call.getArgExpr(ArgIdx);
  };

  /// Functions for custom taintedness propagation.
  static bool UntrustedEnv(CheckerContext &C);
};

using RuleLookupTy = CallDescriptionMap<GenericTaintRule>;

/// Used to parse the configuration file.
struct TaintConfiguration {
  using NameScopeArgs = std::tuple<std::string, std::string, ArgVecTy>;
  enum class VariadicType { None, Src, Dst };

  struct Common {
    std::string Name;
    std::string Scope;
  };

  struct Sink : Common {
    ArgVecTy SinkArgs;
  };

  struct Filter : Common {
    ArgVecTy FilterArgs;
  };

  struct Propagation : Common {
    ArgVecTy SrcArgs;
    ArgVecTy DstArgs;
    VariadicType VarType;
    ArgIdxTy VarIndex;
  };

  std::vector<Propagation> Propagations;
  std::vector<Filter> Filters;
  std::vector<Sink> Sinks;

  TaintConfiguration() = default;
  TaintConfiguration(const TaintConfiguration &) = default;
  TaintConfiguration(TaintConfiguration &&) = default;
  TaintConfiguration &operator=(const TaintConfiguration &) = default;
  TaintConfiguration &operator=(TaintConfiguration &&) = default;
};

struct GenericTaintRuleParser {
  GenericTaintRuleParser(CheckerManager &Mgr) : Mgr(Mgr) {}
  /// Container type used to gather call identification objects grouped into
  /// pairs with their corresponding taint rules. It is temporary as it is used
  /// to finally initialize RuleLookupTy, which is considered to be immutable.
  using RulesContTy = std::vector<std::pair<CallDescription, GenericTaintRule>>;
  RulesContTy parseConfiguration(const std::string &Option,
                                 TaintConfiguration &&Config) const;

private:
  using NamePartsTy = llvm::SmallVector<StringRef, 2>;

  /// Validate part of the configuration, which contains a list of argument
  /// indexes.
  void validateArgVector(const std::string &Option, const ArgVecTy &Args) const;

  template <typename Config> static NamePartsTy parseNameParts(const Config &C);

  // Takes the config and creates a CallDescription for it and associates a Rule
  // with that.
  template <typename Config>
  static void consumeRulesFromConfig(const Config &C, GenericTaintRule &&Rule,
                                     RulesContTy &Rules);

  void parseConfig(const std::string &Option, TaintConfiguration::Sink &&P,
                   RulesContTy &Rules) const;
  void parseConfig(const std::string &Option, TaintConfiguration::Filter &&P,
                   RulesContTy &Rules) const;
  void parseConfig(const std::string &Option,
                   TaintConfiguration::Propagation &&P,
                   RulesContTy &Rules) const;

  CheckerManager &Mgr;
};

class GenericTaintChecker
    : public Checker<check::PreCall, check::PostCall, check::BeginFunction> {
public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkBeginFunction(CheckerContext &C) const;

  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;

  /// Generate a report if the expression is tainted or points to tainted data.
  bool generateReportIfTainted(const Expr *E, StringRef Msg,
                               CheckerContext &C) const;

  bool isTaintReporterCheckerEnabled = false;
  std::optional<BugType> BT;

private:
  bool checkUncontrolledFormatString(const CallEvent &Call,
                                     CheckerContext &C) const;

  void taintUnsafeSocketProtocol(const CallEvent &Call,
                                 CheckerContext &C) const;

  /// The taint rules are initalized with the help of a CheckerContext to
  /// access user-provided configuration.
  LLVM_ATTRIBUTE_MINSIZE void initTaintRules(CheckerContext &C) const;

  // TODO: The two separate `CallDescriptionMap`s were introduced when
  // `CallDescription` was unable to restrict matches to the global namespace
  // only. This limitation no longer exists, so the following two maps should
  // be unified.
  mutable std::optional<RuleLookupTy> StaticTaintRules;
  mutable std::optional<RuleLookupTy> DynamicTaintRules;
};
} // end of anonymous namespace

/// YAML serialization mapping.
LLVM_YAML_IS_SEQUENCE_VECTOR(TaintConfiguration::Sink)
LLVM_YAML_IS_SEQUENCE_VECTOR(TaintConfiguration::Filter)
LLVM_YAML_IS_SEQUENCE_VECTOR(TaintConfiguration::Propagation)

namespace llvm {
namespace yaml {
template <> struct MappingTraits<TaintConfiguration> {
  static void mapping(IO &IO, TaintConfiguration &Config) {
    IO.mapOptional("Propagations", Config.Propagations);
    IO.mapOptional("Filters", Config.Filters);
    IO.mapOptional("Sinks", Config.Sinks);
  }
};

template <> struct MappingTraits<TaintConfiguration::Sink> {
  static void mapping(IO &IO, TaintConfiguration::Sink &Sink) {
    IO.mapRequired("Name", Sink.Name);
    IO.mapOptional("Scope", Sink.Scope);
    IO.mapRequired("Args", Sink.SinkArgs);
  }
};

template <> struct MappingTraits<TaintConfiguration::Filter> {
  static void mapping(IO &IO, TaintConfiguration::Filter &Filter) {
    IO.mapRequired("Name", Filter.Name);
    IO.mapOptional("Scope", Filter.Scope);
    IO.mapRequired("Args", Filter.FilterArgs);
  }
};

template <> struct MappingTraits<TaintConfiguration::Propagation> {
  static void mapping(IO &IO, TaintConfiguration::Propagation &Propagation) {
    IO.mapRequired("Name", Propagation.Name);
    IO.mapOptional("Scope", Propagation.Scope);
    IO.mapOptional("SrcArgs", Propagation.SrcArgs);
    IO.mapOptional("DstArgs", Propagation.DstArgs);
    IO.mapOptional("VariadicType", Propagation.VarType);
    IO.mapOptional("VariadicIndex", Propagation.VarIndex);
  }
};

template <> struct ScalarEnumerationTraits<TaintConfiguration::VariadicType> {
  static void enumeration(IO &IO, TaintConfiguration::VariadicType &Value) {
    IO.enumCase(Value, "None", TaintConfiguration::VariadicType::None);
    IO.enumCase(Value, "Src", TaintConfiguration::VariadicType::Src);
    IO.enumCase(Value, "Dst", TaintConfiguration::VariadicType::Dst);
  }
};
} // namespace yaml
} // namespace llvm

/// A set which is used to pass information from call pre-visit instruction
/// to the call post-visit. The values are signed integers, which are either
/// ReturnValueIndex, or indexes of the pointer/reference argument, which
/// points to data, which should be tainted on return.
REGISTER_MAP_WITH_PROGRAMSTATE(TaintArgsOnPostVisit, const StackFrame *,
                               ImmutableSet<ArgIdxTy>)
REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(ArgIdxFactory, ArgIdxTy)

void GenericTaintRuleParser::validateArgVector(const std::string &Option,
                                               const ArgVecTy &Args) const {
  for (ArgIdxTy Arg : Args) {
    if (Arg < ReturnValueIndex) {
      Mgr.reportInvalidCheckerOptionValue(
          Mgr.getChecker<GenericTaintChecker>(), Option,
          "an argument number for propagation rules greater or equal to -1");
    }
  }
}

template <typename Config>
GenericTaintRuleParser::NamePartsTy
GenericTaintRuleParser::parseNameParts(const Config &C) {
  NamePartsTy NameParts;
  if (!C.Scope.empty()) {
    // If the Scope argument contains multiple "::" parts, those are considered
    // namespace identifiers.
    StringRef{C.Scope}.split(NameParts, "::", /*MaxSplit*/ -1,
                             /*KeepEmpty*/ false);
  }
  NameParts.emplace_back(C.Name);
  return NameParts;
}

template <typename Config>
void GenericTaintRuleParser::consumeRulesFromConfig(const Config &C,
                                                    GenericTaintRule &&Rule,
                                                    RulesContTy &Rules) {
  NamePartsTy NameParts = parseNameParts(C);
  Rules.emplace_back(CallDescription(CDM::Unspecified, NameParts),
                     std::move(Rule));
}

void GenericTaintRuleParser::parseConfig(const std::string &Option,
                                         TaintConfiguration::Sink &&S,
                                         RulesContTy &Rules) const {
  validateArgVector(Option, S.SinkArgs);
  consumeRulesFromConfig(S, GenericTaintRule::Sink(std::move(S.SinkArgs)),
                         Rules);
}

void GenericTaintRuleParser::parseConfig(const std::string &Option,
                                         TaintConfiguration::Filter &&S,
                                         RulesContTy &Rules) const {
  validateArgVector(Option, S.FilterArgs);
  consumeRulesFromConfig(S, GenericTaintRule::Filter(std::move(S.FilterArgs)),
                         Rules);
}

void GenericTaintRuleParser::parseConfig(const std::string &Option,
                                         TaintConfiguration::Propagation &&P,
                                         RulesContTy &Rules) const {
  validateArgVector(Option, P.SrcArgs);
  validateArgVector(Option, P.DstArgs);
  bool IsSrcVariadic = P.VarType == TaintConfiguration::VariadicType::Src;
  bool IsDstVariadic = P.VarType == TaintConfiguration::VariadicType::Dst;
  std::optional<ArgIdxTy> JustVarIndex = P.VarIndex;

  ArgSet SrcDesc(std::move(P.SrcArgs),
                 IsSrcVariadic ? JustVarIndex : std::nullopt);
  ArgSet DstDesc(std::move(P.DstArgs),
                 IsDstVariadic ? JustVarIndex : std::nullopt);

  consumeRulesFromConfig(
      P, GenericTaintRule::Prop(std::move(SrcDesc), std::move(DstDesc)), Rules);
}

GenericTaintRuleParser::RulesContTy
GenericTaintRuleParser::parseConfiguration(const std::string &Option,
                                           TaintConfiguration &&Config) const {

  RulesContTy Rules;

  for (auto &F : Config.Filters)
    parseConfig(Option, std::move(F), Rules);

  for (auto &S : Config.Sinks)
    parseConfig(Option, std::move(S), Rules);

  for (auto &P : Config.Propagations)
    parseConfig(Option, std::move(P), Rules);

  return Rules;
}

enum class TaintRuleKind : uint8_t { Source, Prop, Sink };
enum class TaintRuleMessage : uint8_t {
  None,
  SanitizeSystemArgs,
  UncontrolledFormatString,
};

struct ArgSetDescriptor {
  int8_t Args[4];
  uint8_t Count;
  int8_t VariadicIndex;
};

struct TaintRuleDescriptor {
  uint16_t NameOffset;
  uint8_t NameLength;
  uint8_t Mode;
  TaintRuleKind Kind;
  TaintRuleMessage Message;
  ArgSetDescriptor First;
  ArgSetDescriptor Second;
};

static_assert(sizeof(ArgSetDescriptor) == 6);
static_assert(sizeof(TaintRuleDescriptor) == 18);

// Keep each rule in a compact static descriptor. Construct the dynamic
// CallDescription and GenericTaintRule objects once when the checker is
// first used.
// clang-format off
#define TAINT_ARGS(A0, A1, A2, A3, Count, Variadic) \
  {{A0, A1, A2, A3}, Count, Variadic}
#define TAINT_CLIB static_cast<uint8_t>(CDM::CLibrary)
#define TAINT_CLIB_HARDENED \
  static_cast<uint8_t>(CDM::CLibraryMaybeHardened)
#define TAINT_SOURCE TaintRuleKind::Source
#define TAINT_PROP TaintRuleKind::Prop
#define TAINT_SINK TaintRuleKind::Sink
#define TAINT_NO_MESSAGE TaintRuleMessage::None
#define TAINT_SANITIZE_SYSTEM_ARGS TaintRuleMessage::SanitizeSystemArgs
#define TAINT_UNCONTROLLED_FORMAT_STRING \
  TaintRuleMessage::UncontrolledFormatString
#define TAINT_RULES(M) \
  /* Sources. */ \
  M(fdopen, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(fopen, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(freopen, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getch, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getchar, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getchar_unlocked, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(gets, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(0, -1, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(gets_s, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(0, -1, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(scanf, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 0, 1), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(scanf_s, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 0, 1), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(wgetch, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  /* _IO_getc could be a propagator, but that would require modeling all */ \
  /* possible sources of the _IO_FILE * argument. */ \
  M(_IO_getc, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getcwd, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(0, -1, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getwd, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(0, -1, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(readlink, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(1, -1, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(readlinkat, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(2, -1, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(get_current_dir_name, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(gethostname, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getnameinfo, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(2, 4, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getseuserbyname, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(1, 2, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getgroups, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(1, -1, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getlogin, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(-1, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(getlogin_r, TAINT_CLIB, TAINT_SOURCE, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  /* Propagators. */ \
  M(accept, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(atoi, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(atol, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(atoll, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(fgetc, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(fgetln, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(fgets, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(2, 0, 0, 0, 1, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(fgetws, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(2, 0, 0, 0, 1, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(fscanf, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, 2)) \
  M(fscanf_s, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, 2)) \
  M(sscanf, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, 2)) \
  M(sscanf_s, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, 2)) \
  M(getc, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(getc_unlocked, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(getdelim, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(3, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 1, -2)) \
  /* TODO: This also matches std::getline(); rule it out explicitly. */ \
  M(getline, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(2, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 1, -2)) \
  M(getw, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(pread, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 2, 3, 4, -2), TAINT_ARGS(1, -1, 0, 0, 2, -2)) \
  M(read, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 2, 0, 0, 2, -2), TAINT_ARGS(1, -1, 0, 0, 2, -2)) \
  M(fread, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(3, 0, 0, 0, 1, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(recv, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(1, -1, 0, 0, 2, -2)) \
  M(recvfrom, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(1, -1, 0, 0, 2, -2)) \
  M(ttyname, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(ttyname_r, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(1, -1, 0, 0, 2, -2)) \
  M(basename, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(dirname, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(fnmatch, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(mbtowc, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 0, 0, 0, 1, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(wctomb, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 0, 0, 0, 1, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(wcwidth, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(memcmp, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 2, 0, 3, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(memcpy, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 2, 0, 0, 2, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(memmove, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 2, 0, 0, 2, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(bcopy, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 2, 0, 0, 2, -2), TAINT_ARGS(1, 0, 0, 0, 1, -2)) \
  /* These search functions only propagate taint from the haystack. */ \
  M(memmem, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strstr, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strcasestr, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(memchr, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(memrchr, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(rawmemchr, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strchr, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strrchr, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strchrnul, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(index, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(rindex, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  /* FIXME: For arrays, only the first array element gets tainted. */ \
  M(qsort, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 1, -2)) \
  M(qsort_r, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 1, -2)) \
  M(strcmp, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strcasecmp, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strncmp, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 2, 0, 3, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strncasecmp, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 2, 0, 3, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strspn, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strcspn, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strpbrk, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strndup, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strndupa, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strdup, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strdupa, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(wcsdup, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  /* strlen, wcslen, strnlen, and similar functions intentionally do not */ \
  /* propagate taint. See https://github.com/llvm/llvm-project/pull/66086. */ \
  M(strtol, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(1, -1, 0, 0, 2, -2)) \
  M(strtoll, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(1, -1, 0, 0, 2, -2)) \
  M(strtoul, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(1, -1, 0, 0, 2, -2)) \
  M(strtoull, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(1, -1, 0, 0, 2, -2)) \
  M(tolower, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(toupper, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isalnum, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isalpha, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isascii, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isblank, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(iscntrl, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isdigit, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isgraph, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(islower, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isprint, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(ispunct, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isspace, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isupper, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(isxdigit, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(-1, 0, 0, 0, 1, -2)) \
  M(strcpy, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 0, 0, 0, 1, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(stpcpy, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 0, 0, 0, 1, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(strcat, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(wcsncat, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(strncpy, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 2, 0, 0, 2, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(strncat, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 2, 0, 3, -2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(strlcpy, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 2, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 1, -2)) \
  M(strlcat, TAINT_CLIB_HARDENED, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(0, 1, 2, 0, 3, -2), TAINT_ARGS(0, 0, 0, 0, 1, -2)) \
  /* The hardened sprintf variants insert parameters in the middle, so */ \
  /* CLibraryMaybeHardened cannot model them together with the base calls. */ \
  M(snprintf, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 2, 0, 0, 2, 3), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(sprintf, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 0, 0, 0, 1, 2), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(__snprintf_chk, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(1, 4, 0, 0, 2, 5), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  M(__sprintf_chk, TAINT_CLIB, TAINT_PROP, TAINT_NO_MESSAGE, TAINT_ARGS(3, 0, 0, 0, 1, 4), TAINT_ARGS(0, -1, 0, 0, 2, -2)) \
  /* Sinks. */ \
  M(system, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(popen, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(execl, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 0, 0, 0, 0, 0), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(execle, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 0, 0, 0, 0, 0), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(execlp, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 0, 0, 0, 0, 0), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(execv, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(execve, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 1, 2, 0, 3, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(fexecve, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 1, 2, 0, 3, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(execvp, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 1, 0, 0, 2, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(execvpe, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 1, 2, 0, 3, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(dlopen, TAINT_CLIB, TAINT_SINK, TAINT_SANITIZE_SYSTEM_ARGS, TAINT_ARGS(0, 0, 0, 0, 1, -2), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  /* Allocation functions are intentionally not unconditional sinks because */ \
  /* that produces false positives; specialized checkers should model them. */ \
  M(setproctitle, TAINT_CLIB, TAINT_SINK, TAINT_UNCONTROLLED_FORMAT_STRING, TAINT_ARGS(0, 0, 0, 0, 1, 1), TAINT_ARGS(0, 0, 0, 0, 0, -2)) \
  M(setproctitle_fast, TAINT_CLIB, TAINT_SINK, TAINT_UNCONTROLLED_FORMAT_STRING, TAINT_ARGS(0, 0, 0, 0, 1, 1), TAINT_ARGS(0, 0, 0, 0, 0, -2))
// clang-format on

struct TaintRuleNameTable {
#define TAINT_RULE_NAME(Name, ...) char N_##Name[sizeof(#Name)];
  TAINT_RULES(TAINT_RULE_NAME)
#undef TAINT_RULE_NAME
};

constexpr TaintRuleNameTable TaintRuleNames = {
#define TAINT_RULE_NAME(Name, ...) #Name,
    TAINT_RULES(TAINT_RULE_NAME)
#undef TAINT_RULE_NAME
};

static_assert(sizeof(TaintRuleNameTable) <=
              std::numeric_limits<uint16_t>::max());

#define TAINT_RULE_DESCRIPTOR(Name, ...)                                       \
  {static_cast<uint16_t>(offsetof(TaintRuleNameTable, N_##Name)),              \
   static_cast<uint8_t>(sizeof(TaintRuleNames.N_##Name) - 1), __VA_ARGS__},
constexpr TaintRuleDescriptor TaintRuleDescriptors[] = {
    TAINT_RULES(TAINT_RULE_DESCRIPTOR)};
#undef TAINT_RULE_DESCRIPTOR
#undef TAINT_RULES
#undef TAINT_UNCONTROLLED_FORMAT_STRING
#undef TAINT_SANITIZE_SYSTEM_ARGS
#undef TAINT_NO_MESSAGE
#undef TAINT_SINK
#undef TAINT_PROP
#undef TAINT_SOURCE
#undef TAINT_CLIB_HARDENED
#undef TAINT_CLIB
#undef TAINT_ARGS

void GenericTaintChecker::initTaintRules(CheckerContext &C) const {
  // Check for exact name match for functions without builtin substitutes.
  // Use qualified name, because these are C functions without namespace.

  if (StaticTaintRules || DynamicTaintRules)
    return;

  using RulesConstructionTy =
      std::vector<std::pair<CallDescription, GenericTaintRule>>;
  using TR = GenericTaintRule;

  RulesConstructionTy GlobalCRules;
  GlobalCRules.reserve(std::size(TaintRuleDescriptors) + 2);

  auto MakeArgSet = [](const ArgSetDescriptor &Desc) {
    ArgVecTy Args;
    Args.append(Desc.Args, Desc.Args + Desc.Count);
    std::optional<ArgIdxTy> VariadicIndex;
    if (Desc.VariadicIndex != -2)
      VariadicIndex = Desc.VariadicIndex;
    return ArgSet(std::move(Args), VariadicIndex);
  };

  for (const TaintRuleDescriptor &Desc : TaintRuleDescriptors) {
    StringRef Name(reinterpret_cast<const char *>(&TaintRuleNames) +
                       Desc.NameOffset,
                   Desc.NameLength);
    CallDescription Call(static_cast<CDM>(Desc.Mode), {Name});
    ArgSet First = MakeArgSet(Desc.First);
    ArgSet Second = MakeArgSet(Desc.Second);
    GenericTaintRule Rule = [&]() {
      switch (Desc.Kind) {
      case TaintRuleKind::Source:
        return TR::Source(std::move(First));
      case TaintRuleKind::Prop:
        return TR::Prop(std::move(First), std::move(Second));
      case TaintRuleKind::Sink: {
        std::optional<StringRef> Message;
        switch (Desc.Message) {
        case TaintRuleMessage::None:
          break;
        case TaintRuleMessage::SanitizeSystemArgs:
          Message = MsgSanitizeSystemArgs;
          break;
        case TaintRuleMessage::UncontrolledFormatString:
          Message = MsgUncontrolledFormatString;
          break;
        }
        return TR::Sink(std::move(First), Message);
      }
      }
      llvm_unreachable("unknown taint rule kind");
    }();
    GlobalCRules.emplace_back(std::move(Call), std::move(Rule));
  }

  if (TR::UntrustedEnv(C)) {
    // void setproctitle_init(int argc, char *argv[], char *envp[])
    // TODO: replace `MsgCustomSink` with a message that fits this situation.
    GlobalCRules.push_back({{CDM::CLibrary, {"setproctitle_init"}},
                            TR::Sink({{1, 2}}, MsgCustomSink)});

    // `getenv` returns taint only in untrusted environments.
    GlobalCRules.push_back(
        {{CDM::CLibrary, {"getenv"}}, TR::Source({{ReturnValueIndex}})});
  }
  CheckerManager *Mgr = C.getAnalysisManager().getCheckerManager();

  StaticTaintRules = RuleLookupTy{};
  if (Mgr->getAnalyzerOptions().getCheckerBooleanOption(this,
                                                        "EnableDefaultConfig"))
    StaticTaintRules.emplace(std::make_move_iterator(GlobalCRules.begin()),
                             std::make_move_iterator(GlobalCRules.end()));

  // User-provided taint configuration.
  const GenericTaintRuleParser ConfigParser{*Mgr};
  std::string Option{"Config"};
  StringRef ConfigFile =
      Mgr->getAnalyzerOptions().getCheckerStringOption(this, Option);
  std::optional<TaintConfiguration> Config =
      getConfiguration<TaintConfiguration>(*Mgr, this, Option, ConfigFile);
  if (!Config) {
    // We don't have external taint config, no parsing required.
    DynamicTaintRules = RuleLookupTy{};
    return;
  }

  GenericTaintRuleParser::RulesContTy Rules{
      ConfigParser.parseConfiguration(Option, std::move(*Config))};

  DynamicTaintRules.emplace(std::make_move_iterator(Rules.begin()),
                            std::make_move_iterator(Rules.end()));
}

bool isPointerToCharArray(const QualType &QT) {
  if (!QT->isPointerType())
    return false;
  QualType PointeeType = QT->getPointeeType();
  return PointeeType->isPointerType() &&
         PointeeType->getPointeeType()->isCharType();
}

// The incoming parameters of the main function get tainted
// if the program called in an untrusted environment.
void GenericTaintChecker::checkBeginFunction(CheckerContext &C) const {
  if (!C.inTopFrame() || C.getAnalysisManager()
                             .getAnalyzerOptions()
                             .ShouldAssumeControlledEnvironment)
    return;

  const auto *FD = dyn_cast<FunctionDecl>(C.getStackFrame()->getDecl());
  if (!FD || !FD->isMain() || FD->param_size() < 2)
    return;

  if (!FD->parameters()[0]->getType()->isIntegerType())
    return;

  if (!isPointerToCharArray(FD->parameters()[1]->getType()))
    return;
  ProgramStateRef State = C.getState();

  const MemRegion *ArgcReg =
      State->getRegion(FD->parameters()[0], C.getStackFrame());
  SVal ArgcSVal = State->getSVal(ArgcReg);
  State = addTaint(State, ArgcSVal);
  StringRef ArgcName = FD->parameters()[0]->getName();
  if (auto N = ArgcSVal.getAs<NonLoc>()) {
    ConstraintManager &CM = C.getConstraintManager();
    // The upper bound is the ARG_MAX on an arbitrary Linux
    // to model that is is typically smaller than INT_MAX.
    State = CM.assumeInclusiveRange(State, *N, llvm::APSInt::getUnsigned(1),
                                    llvm::APSInt::getUnsigned(2097152), true);
  }

  const MemRegion *ArgvReg =
      State->getRegion(FD->parameters()[1], C.getStackFrame());
  SVal ArgvSVal = State->getSVal(ArgvReg);
  State = addTaint(State, ArgvSVal);
  StringRef ArgvName = FD->parameters()[1]->getName();

  bool HaveEnvp = FD->param_size() > 2;
  SVal EnvpSVal;
  StringRef EnvpName;
  if (HaveEnvp && !isPointerToCharArray(FD->parameters()[2]->getType()))
    return;
  if (HaveEnvp) {
    const MemRegion *EnvPReg =
        State->getRegion(FD->parameters()[2], C.getStackFrame());
    EnvpSVal = State->getSVal(EnvPReg);
    EnvpName = FD->parameters()[2]->getName();
    State = addTaint(State, EnvpSVal);
  }

  const NoteTag *OriginatingTag =
      C.getNoteTag([ArgvSVal, ArgcSVal, ArgcName, ArgvName, EnvpSVal,
                    EnvpName](PathSensitiveBugReport &BR) -> std::string {
        if ((!BR.isInteresting(ArgcSVal) && !BR.isInteresting(ArgvSVal) &&
             !BR.isInteresting(EnvpSVal)))
          return "";
        if (BR.getBugType().getCategory() != categories::TaintedData)
          return "";
        std::string Message = "";
        if (BR.isInteresting(ArgvSVal))
          Message += "'" + ArgvName.str() + "'";
        if (BR.isInteresting(ArgcSVal)) {
          if (Message.size() > 0)
            Message += ", ";
          Message += "'" + ArgcName.str() + "'";
        }
        if (BR.isInteresting(EnvpSVal)) {
          if (Message.size() > 0)
            Message += ", ";
          Message += "'" + EnvpName.str() + "'";
        }
        return "Taint originated in " + Message;
      });
  C.addTransition(State, OriginatingTag);
}

void GenericTaintChecker::checkPreCall(const CallEvent &Call,
                                       CheckerContext &C) const {

  initTaintRules(C);

  // FIXME: this should be much simpler.
  if (const auto *Rule =
          Call.isGlobalCFunction() ? StaticTaintRules->lookup(Call) : nullptr)
    Rule->process(*this, Call, C);
  else if (const auto *Rule = DynamicTaintRules->lookup(Call))
    Rule->process(*this, Call, C);

  // FIXME: These edge cases are to be eliminated from here eventually.
  //
  // Additional check that is not supported by CallDescription.
  // TODO: Make CallDescription be able to match attributes such as printf-like
  // arguments.
  checkUncontrolledFormatString(Call, C);

  // TODO: Modeling sockets should be done in a specific checker.
  // Socket is a source, which taints the return value.
  taintUnsafeSocketProtocol(Call, C);
}

void GenericTaintChecker::checkPostCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  // Set the marked values as tainted. The return value only accessible from
  // checkPostStmt.
  ProgramStateRef State = C.getState();
  const StackFrame *CurrentFrame = C.getStackFrame();

  // Depending on what was tainted at pre-visit, we determined a set of
  // arguments which should be tainted after the function returns. These are
  // stored in the state as TaintArgsOnPostVisit set.
  TaintArgsOnPostVisitTy TaintArgsMap = State->get<TaintArgsOnPostVisit>();

  const ImmutableSet<ArgIdxTy> *TaintArgs = TaintArgsMap.lookup(CurrentFrame);
  if (!TaintArgs)
    return;
  assert(!TaintArgs->isEmpty());

  LLVM_DEBUG(for (ArgIdxTy I
                  : *TaintArgs) {
    llvm::dbgs() << "PostCall<";
    Call.dump(llvm::dbgs());
    llvm::dbgs() << "> actually wants to taint arg index: " << I << '\n';
  });

  const NoteTag *InjectionTag = nullptr;
  std::vector<SymbolRef> TaintedSymbols;
  std::vector<ArgIdxTy> TaintedIndexes;
  for (ArgIdxTy ArgNum : *TaintArgs) {
    // Special handling for the tainted return value.
    if (ArgNum == ReturnValueIndex) {
      State = addTaint(State, Call.getReturnValue());
      std::vector<SymbolRef> TaintedSyms =
          getTaintedSymbols(State, Call.getReturnValue());
      if (!TaintedSyms.empty()) {
        TaintedSymbols.push_back(TaintedSyms[0]);
        TaintedIndexes.push_back(ArgNum);
      }
      continue;
    }
    // The arguments are pointer arguments. The data they are pointing at is
    // tainted after the call.
    if (auto V = getPointeeOf(State, Call.getArgSVal(ArgNum))) {
      State = addTaint(State, *V);
      std::vector<SymbolRef> TaintedSyms = getTaintedSymbols(State, *V);
      if (!TaintedSyms.empty()) {
        TaintedSymbols.push_back(TaintedSyms[0]);
        TaintedIndexes.push_back(ArgNum);
      }
    }
  }
  // Create a NoteTag callback, which prints to the user where the taintedness
  // was propagated to.
  InjectionTag = taintPropagationExplainerTag(C, TaintedSymbols, TaintedIndexes,
                                              Call.getCalleeStackFrame(0));
  // Clear up the taint info from the state.
  State = State->remove<TaintArgsOnPostVisit>(CurrentFrame);
  C.addTransition(State, InjectionTag);
}

void GenericTaintChecker::printState(raw_ostream &Out, ProgramStateRef State,
                                     const char *NL, const char *Sep) const {
  printTaint(State, Out, NL, Sep);
}

void GenericTaintRule::process(const GenericTaintChecker &Checker,
                               const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const ArgIdxTy CallNumArgs = fromArgumentCount(Call.getNumArgs());

  /// Iterate every call argument, and get their corresponding Expr and SVal.
  const auto ForEachCallArg = [&C, &Call, CallNumArgs](auto &&Fun) {
    for (ArgIdxTy I = ReturnValueIndex; I < CallNumArgs; ++I) {
      const Expr *E = GetArgExpr(I, Call);
      Fun(I, E, C.getSVal(E));
    }
  };

  /// Check for taint sinks.
  ForEachCallArg([this, &Checker, &C, &State](ArgIdxTy I, const Expr *E, SVal) {
    // Add taintedness to stdin parameters
    if (isStdin(C.getSVal(E), C.getASTContext())) {
      State = addTaint(State, C.getSVal(E));
    }
    if (SinkArgs.contains(I) && isTaintedOrPointsToTainted(State, C.getSVal(E)))
      Checker.generateReportIfTainted(E, SinkMsg.value_or(MsgCustomSink), C);
  });

  /// Check for taint filters.
  ForEachCallArg([this, &State](ArgIdxTy I, const Expr *E, SVal S) {
    if (FilterArgs.contains(I)) {
      State = removeTaint(State, S);
      if (auto P = getPointeeOf(State, S))
        State = removeTaint(State, *P);
    }
  });

  /// Check for taint propagation sources.
  /// A rule will make the destination variables tainted if PropSrcArgs
  /// is empty (taints the destination
  /// arguments unconditionally), or if any of its signified
  /// args are tainted in context of the current CallEvent.
  bool IsMatching = PropSrcArgs.isEmpty();
  std::vector<SymbolRef> TaintedSymbols;
  std::vector<ArgIdxTy> TaintedIndexes;
  ForEachCallArg([this, &C, &IsMatching, &State, &TaintedSymbols,
                  &TaintedIndexes](ArgIdxTy I, const Expr *E, SVal) {
    std::optional<SVal> TaintedSVal =
        getTaintedPointeeOrPointer(State, C.getSVal(E));
    IsMatching =
        IsMatching || (PropSrcArgs.contains(I) && TaintedSVal.has_value());

    // We track back tainted arguments except for stdin
    if (TaintedSVal && !isStdin(*TaintedSVal, C.getASTContext())) {
      std::vector<SymbolRef> TaintedArgSyms =
          getTaintedSymbols(State, *TaintedSVal);
      if (!TaintedArgSyms.empty()) {
        llvm::append_range(TaintedSymbols, TaintedArgSyms);
        TaintedIndexes.push_back(I);
      }
    }
  });

  // Early return for propagation rules which dont match.
  // Matching propagations, Sinks and Filters will pass this point.
  if (!IsMatching)
    return;

  const auto WouldEscape = [](SVal V, QualType Ty) -> bool {
    if (!isa<Loc>(V))
      return false;

    const bool IsNonConstRef = Ty->isReferenceType() && !Ty.isConstQualified();
    const bool IsNonConstPtr =
        Ty->isPointerType() && !Ty->getPointeeType().isConstQualified();

    return IsNonConstRef || IsNonConstPtr;
  };

  /// Propagate taint where it is necessary.
  auto &F = State->getStateManager().get_context<ArgIdxFactory>();
  ImmutableSet<ArgIdxTy> Result = F.getEmptySet();
  ForEachCallArg(
      [&](ArgIdxTy I, const Expr *E, SVal V) {
        if (PropDstArgs.contains(I)) {
          LLVM_DEBUG(llvm::dbgs() << "PreCall<"; Call.dump(llvm::dbgs());
                     llvm::dbgs()
                     << "> prepares tainting arg index: " << I << '\n';);
          Result = F.add(Result, I);
        }

        // Taint property gets lost if the variable is passed as a
        // non-const pointer or reference to a function which is
        // not inlined. For matching rules we want to preserve the taintedness.
        // TODO: We should traverse all reachable memory regions via the
        // escaping parameter. Instead of doing that we simply mark only the
        // referred memory region as tainted.
        if (WouldEscape(V, E->getType()) && getTaintedPointeeOrPointer(State, V)) {
          LLVM_DEBUG(if (!Result.contains(I)) {
            llvm::dbgs() << "PreCall<";
            Call.dump(llvm::dbgs());
            llvm::dbgs() << "> prepares tainting arg index: " << I << '\n';
          });
          Result = F.add(Result, I);
        }
      });

  if (!Result.isEmpty())
    State = State->set<TaintArgsOnPostVisit>(C.getStackFrame(), Result);
  const NoteTag *InjectionTag = taintOriginTrackerTag(
      C, std::move(TaintedSymbols), std::move(TaintedIndexes),
      Call.getCalleeStackFrame(0));
  C.addTransition(State, InjectionTag);
}

bool GenericTaintRule::UntrustedEnv(CheckerContext &C) {
  return !C.getAnalysisManager()
              .getAnalyzerOptions()
              .ShouldAssumeControlledEnvironment;
}

bool GenericTaintChecker::generateReportIfTainted(const Expr *E, StringRef Msg,
                                                  CheckerContext &C) const {
  assert(E);
  if (!isTaintReporterCheckerEnabled)
    return false;
  std::optional<SVal> TaintedSVal =
      getTaintedPointeeOrPointer(C.getState(), C.getSVal(E));

  if (!TaintedSVal)
    return false;

  // Generate diagnostic.
  assert(BT);
  if (ExplodedNode *N = C.generateNonFatalErrorNode(C.getState())) {
    auto report = std::make_unique<PathSensitiveBugReport>(*BT, Msg, N);
    report->addRange(E->getSourceRange());
    for (auto TaintedSym : getTaintedSymbols(C.getState(), *TaintedSVal)) {
      report->markInteresting(TaintedSym);
    }
    C.emitReport(std::move(report));
    return true;
  }
  return false;
}

/// TODO: remove checking for printf format attributes and socket whitelisting
/// from GenericTaintChecker, and that means the following functions:
/// getPrintfFormatArgumentNum,
/// GenericTaintChecker::checkUncontrolledFormatString,
/// GenericTaintChecker::taintUnsafeSocketProtocol

static bool getPrintfFormatArgumentNum(const CallEvent &Call,
                                       const CheckerContext &C,
                                       ArgIdxTy &ArgNum) {
  // Find if the function contains a format string argument.
  // Handles: fprintf, printf, sprintf, snprintf, vfprintf, vprintf, vsprintf,
  // vsnprintf, syslog, custom annotated functions.
  const Decl *CallDecl = Call.getDecl();
  if (!CallDecl)
    return false;
  const FunctionDecl *FDecl = CallDecl->getAsFunction();
  if (!FDecl)
    return false;

  const ArgIdxTy CallNumArgs = fromArgumentCount(Call.getNumArgs());

  for (const auto *Format : FDecl->specific_attrs<FormatAttr>()) {
    // The format attribute uses 1-based parameter indexing, for example
    // plain `printf(const char *fmt, ...)` would be annotated with
    // `__format__(__printf__, 1, 2)`, so we need to subtract 1 to get a
    // 0-based index. (This checker uses 0-based parameter indices.)
    ArgNum = Format->getFormatIdx() - 1;
    // The format attribute also counts the implicit `this` parameter of
    // methods, so e.g. in `SomeClass::method(const char *fmt, ...)` could be
    // annotated with `__format__(__printf__, 2, 3)`. This checker doesn't
    // count the implicit `this` parameter, so in this case we need to subtract
    // one again.
    // FIXME: Apparently the implementation of the format attribute doesn't
    // support methods with an explicit object parameter, so we cannot
    // implement proper support for that rare case either.
    const CXXMethodDecl *MDecl = dyn_cast<CXXMethodDecl>(FDecl);
    if (MDecl && !MDecl->isStatic())
      ArgNum--;

    if ((Format->getType()->getName() == "printf") && CallNumArgs > ArgNum)
      return true;
  }

  return false;
}

bool GenericTaintChecker::checkUncontrolledFormatString(
    const CallEvent &Call, CheckerContext &C) const {
  // Check if the function contains a format string argument.
  ArgIdxTy ArgNum = 0;
  if (!getPrintfFormatArgumentNum(Call, C, ArgNum))
    return false;

  // If either the format string content or the pointer itself are tainted,
  // warn.
  return generateReportIfTainted(Call.getArgExpr(ArgNum),
                                 MsgUncontrolledFormatString, C);
}

void GenericTaintChecker::taintUnsafeSocketProtocol(const CallEvent &Call,
                                                    CheckerContext &C) const {
  if (Call.getNumArgs() < 1)
    return;
  const IdentifierInfo *ID = Call.getCalleeIdentifier();
  if (!ID)
    return;
  if (ID->getName() != "socket")
    return;

  SourceLocation DomLoc = Call.getArgExpr(0)->getExprLoc();
  std::string DomName = C.getMacroNameOrSpelling(DomLoc);
  // Allow internal communication protocols.
  bool SafeProtocol = DomName == "AF_SYSTEM" || DomName == "AF_LOCAL" ||
                      DomName == "AF_UNIX" || DomName == "AF_RESERVED_36";
  if (SafeProtocol)
    return;

  ProgramStateRef State = C.getState();
  auto &F = State->getStateManager().get_context<ArgIdxFactory>();
  ImmutableSet<ArgIdxTy> Result = F.add(F.getEmptySet(), ReturnValueIndex);
  State = State->set<TaintArgsOnPostVisit>(C.getStackFrame(), Result);
  C.addTransition(State);
}

/// Checker registration
void ento::registerTaintPropagationChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<GenericTaintChecker>();
}

bool ento::shouldRegisterTaintPropagationChecker(const CheckerManager &mgr) {
  return true;
}

void ento::registerGenericTaintChecker(CheckerManager &Mgr) {
  GenericTaintChecker *checker = Mgr.getChecker<GenericTaintChecker>();
  checker->isTaintReporterCheckerEnabled = true;
  checker->BT.emplace(Mgr.getCurrentCheckerName(), "Use of Untrusted Data",
                      categories::TaintedData);
}

bool ento::shouldRegisterGenericTaintChecker(const CheckerManager &mgr) {
  return true;
}
