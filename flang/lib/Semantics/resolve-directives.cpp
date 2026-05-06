//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "resolve-directives.h"

#include "check-acc-structure.h"
#include "check-omp-structure.h"
#include "resolve-names-utils.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/openmp-dsa.h"
#include "flang/Semantics/openmp-modifiers.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Support/Flags.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h.inc"
#include "llvm/Support/Debug.h"
#include <list>
#include <map>

namespace Fortran::semantics {

template <typename T>
static Scope *GetScope(SemanticsContext &context, const T &x) {
  if (auto source{GetLastSource(x)}) {
    return &context.FindScope(*source);
  } else {
    return nullptr;
  }
}

template <typename T> class DirectiveAttributeVisitor {
public:
  explicit DirectiveAttributeVisitor(SemanticsContext &context)
      : context_{context} {}

  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}

protected:
  struct DirContext {
    DirContext(const parser::CharBlock &source, T d, Scope &s)
        : directiveSource{source}, directive{d}, scope{s} {}
    parser::CharBlock directiveSource;
    T directive;
    Scope &scope;
    Symbol::Flag defaultDSA{Symbol::Flag::AccShared}; // TODOACC
    std::map<const Symbol *, Symbol::Flag> objectWithDSA;
    std::map<parser::OmpVariableCategory::Value,
        parser::OmpDefaultmapClause::ImplicitBehavior>
        defaultMap;

    std::optional<Symbol::Flag> FindSymbolWithDSA(const Symbol &symbol) {
      if (auto it{objectWithDSA.find(&symbol)}; it != objectWithDSA.end()) {
        return it->second;
      }
      return std::nullopt;
    }

    bool withinConstruct{false};
    std::int64_t associatedLoopLevel{0};
  };

  DirContext &GetContext() {
    CHECK(!dirContext_.empty());
    return dirContext_.back();
  }
  void PushContext(const parser::CharBlock &source, T dir, Scope &scope) {
    if constexpr (std::is_same_v<T, llvm::acc::Directive>) {
      dirContext_.emplace_back(source, dir, scope);
      if (std::size_t size{dirContext_.size()}; size > 1) {
        std::size_t lastIndex{size - 1};
        dirContext_[lastIndex].defaultDSA =
            dirContext_[lastIndex - 1].defaultDSA;
      }
    } else {
      dirContext_.emplace_back(source, dir, scope);
    }
  }
  void PushContext(const parser::CharBlock &source, T dir) {
    PushContext(source, dir, context_.FindScope(source));
  }
  void PopContext() { dirContext_.pop_back(); }
  void SetContextDirectiveSource(parser::CharBlock &dir) {
    GetContext().directiveSource = dir;
  }
  Scope &currScope() { return GetContext().scope; }
  void AddContextDefaultmapBehaviour(parser::OmpVariableCategory::Value VarCat,
      parser::OmpDefaultmapClause::ImplicitBehavior ImpBehav) {
    GetContext().defaultMap[VarCat] = ImpBehav;
  }
  void SetContextDefaultDSA(Symbol::Flag flag) {
    GetContext().defaultDSA = flag;
  }
  void AddToContextObjectWithDSA(
      const Symbol &symbol, Symbol::Flag flag, DirContext &context) {
    context.objectWithDSA.emplace(&symbol, flag);
  }
  void AddToContextObjectWithDSA(const Symbol &symbol, Symbol::Flag flag) {
    AddToContextObjectWithDSA(symbol, flag, GetContext());
  }
  bool IsObjectWithDSA(const Symbol &symbol) {
    return GetContext().FindSymbolWithDSA(symbol).has_value();
  }
  bool IsObjectWithVisibleDSA(const Symbol &symbol) {
    for (std::size_t i{dirContext_.size()}; i != 0; i--) {
      if (dirContext_[i - 1].FindSymbolWithDSA(symbol).has_value()) {
        return true;
      }
    }
    return false;
  }

  bool WithinConstruct() {
    return !dirContext_.empty() && GetContext().withinConstruct;
  }

  void SetContextAssociatedLoopLevel(std::int64_t level) {
    GetContext().associatedLoopLevel = level;
  }
  Symbol &MakeAssocSymbol(
      const SourceName &name, const Symbol &prev, Scope &scope) {
    const auto pair{scope.try_emplace(name, Attrs{}, HostAssocDetails{prev})};
    return *pair.first->second;
  }
  Symbol &MakeAssocSymbol(const SourceName &name, const Symbol &prev) {
    return MakeAssocSymbol(name, prev, currScope());
  }
  void AddDataSharingAttributeObject(SymbolRef object) {
    dataSharingAttributeObjects_.insert(object);
  }
  void ClearDataSharingAttributeObjects() {
    dataSharingAttributeObjects_.clear();
  }
  bool HasDataSharingAttributeObject(const Symbol &);

  /// Extract the iv and bounds of a DO loop:
  /// 1. The loop index/induction variable
  /// 2. The lower bound
  /// 3. The upper bound
  /// 4. The step/increment (or nullptr if not present)
  ///
  /// Each returned tuple value can be nullptr if not present. Diagnoses an
  /// error if the the DO loop is a DO WHILE or DO CONCURRENT loop.
  std::tuple<const parser::Name *, const parser::ScalarExpr *,
      const parser::ScalarExpr *, const parser::ScalarExpr *>
  GetLoopBounds(const parser::DoConstruct &);

  /// Extract the loop index/induction variable from a DO loop. Diagnoses an
  /// error if the the DO loop is a DO WHILE or DO CONCURRENT loop and returns
  /// nullptr.
  const parser::Name *GetLoopIndex(const parser::DoConstruct &);

  const parser::DoConstruct *GetDoConstructIf(
      const parser::ExecutionPartConstruct &);
  Symbol *DeclareNewAccessEntity(const Symbol &, Symbol::Flag, Scope &);
  Symbol *DeclareAccessEntity(const parser::Name &, Symbol::Flag, Scope &);
  Symbol *DeclareAccessEntity(Symbol &, Symbol::Flag, Scope &);
  Symbol *DeclareOrMarkOtherAccessEntity(const parser::Name &, Symbol::Flag);

  UnorderedSymbolSet dataSharingAttributeObjects_; // on one directive
  SemanticsContext &context_;
  std::vector<DirContext> dirContext_; // used as a stack
};

class AccAttributeVisitor : DirectiveAttributeVisitor<llvm::acc::Directive> {
public:
  explicit AccAttributeVisitor(SemanticsContext &context, Scope *topScope)
      : DirectiveAttributeVisitor(context), topScope_(topScope) {}

  template <typename A> void Walk(const A &x) { parser::Walk(x, *this); }
  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}

  bool Pre(const parser::OpenACCBlockConstruct &);
  void Post(const parser::OpenACCBlockConstruct &) { PopContext(); }
  bool Pre(const parser::OpenACCCombinedConstruct &);
  void Post(const parser::OpenACCCombinedConstruct &) { PopContext(); }
  void Post(const parser::AccBeginCombinedDirective &) {
    GetContext().withinConstruct = true;
  }

  bool Pre(const parser::OpenACCDeclarativeConstruct &);
  void Post(const parser::OpenACCDeclarativeConstruct &) { PopContext(); }

  void Post(const parser::AccDeclarativeDirective &) {
    GetContext().withinConstruct = true;
  }

  bool Pre(const parser::OpenACCRoutineConstruct &);
  bool Pre(const parser::AccBindClause &);
  void Post(const parser::OpenACCStandaloneDeclarativeConstruct &);

  void Post(const parser::AccBeginBlockDirective &) {
    GetContext().withinConstruct = true;
  }

  bool Pre(const parser::OpenACCLoopConstruct &);
  void Post(const parser::OpenACCLoopConstruct &) { PopContext(); }
  void Post(const parser::AccLoopDirective &) {
    GetContext().withinConstruct = true;
  }

  // TODO: We should probably also privatize ConcurrentBounds.
  template <typename A>
  bool Pre(const parser::LoopBounds<parser::ScalarName, A> &x) {
    if (!dirContext_.empty() && GetContext().withinConstruct) {
      if (auto *symbol{ResolveAcc(
              x.Name().thing, Symbol::Flag::AccPrivate, currScope())}) {
        AddToContextObjectWithDSA(*symbol, Symbol::Flag::AccPrivate);
      }
    }
    return true;
  }

  bool Pre(const parser::OpenACCStandaloneConstruct &);
  void Post(const parser::OpenACCStandaloneConstruct &) { PopContext(); }
  void Post(const parser::AccStandaloneDirective &) {
    GetContext().withinConstruct = true;
  }

  bool Pre(const parser::OpenACCWaitConstruct &);
  void Post(const parser::OpenACCWaitConstruct &) { PopContext(); }
  bool Pre(const parser::OpenACCAtomicConstruct &);
  void Post(const parser::OpenACCAtomicConstruct &) { PopContext(); }

  bool Pre(const parser::OpenACCCacheConstruct &);
  void Post(const parser::OpenACCCacheConstruct &) { PopContext(); }

  void Post(const parser::AccDefaultClause &);

  bool Pre(const parser::AccClause::Attach &);
  bool Pre(const parser::AccClause::Detach &);

  bool Pre(const parser::AccClause::Copy &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccCopy);
    return false;
  }

  bool Pre(const parser::AccClause::Create &x) {
    const auto &objectList{std::get<parser::AccObjectList>(x.v.t)};
    ResolveAccObjectList(objectList, Symbol::Flag::AccCreate);
    return false;
  }

  bool Pre(const parser::AccClause::Copyin &x) {
    const auto &objectList{std::get<parser::AccObjectList>(x.v.t)};
    const auto &modifier{
        std::get<std::optional<parser::AccDataModifier>>(x.v.t)};
    if (modifier &&
        (*modifier).v == parser::AccDataModifier::Modifier::ReadOnly) {
      ResolveAccObjectList(objectList, Symbol::Flag::AccCopyInReadOnly);
    } else {
      ResolveAccObjectList(objectList, Symbol::Flag::AccCopyIn);
    }
    return false;
  }

  bool Pre(const parser::AccClause::Copyout &x) {
    const auto &objectList{std::get<parser::AccObjectList>(x.v.t)};
    ResolveAccObjectList(objectList, Symbol::Flag::AccCopyOut);
    return false;
  }

  bool Pre(const parser::AccClause::Present &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccPresent);
    return false;
  }
  bool Pre(const parser::AccClause::Private &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccPrivate);
    return false;
  }
  bool Pre(const parser::AccClause::Firstprivate &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccFirstPrivate);
    return false;
  }

  bool Pre(const parser::AccClause::Device &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccDevice);
    return false;
  }

  bool Pre(const parser::AccClause::DeviceResident &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccDeviceResident);
    return false;
  }

  bool Pre(const parser::AccClause::Deviceptr &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccDevicePtr);
    return false;
  }

  bool Pre(const parser::AccClause::Link &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccLink);
    return false;
  }

  bool Pre(const parser::AccClause::Host &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccHost);
    return false;
  }

  bool Pre(const parser::AccClause::Self &x) {
    const std::optional<parser::AccSelfClause> &accSelfClause = x.v;
    if (accSelfClause &&
        std::holds_alternative<parser::AccObjectList>((*accSelfClause).u)) {
      const auto &accObjectList =
          std::get<parser::AccObjectList>((*accSelfClause).u);
      ResolveAccObjectList(accObjectList, Symbol::Flag::AccSelf);
    }
    return false;
  }

  bool Pre(const parser::AccClause::Reduction &x) {
    const auto &objectList{std::get<parser::AccObjectList>(x.v.t)};
    ResolveAccObjectList(objectList, Symbol::Flag::AccReduction);
    return false;
  }

  bool Pre(const parser::AccClause::UseDevice &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccUseDevice);
    // use_device is only valid on host_data directive
    assert(GetContext().directive == llvm::acc::Directive::ACCD_host_data &&
        "use_device clause is only valid on host_data directive");
    // Check for duplicate use_device variables
    for (const auto &accObject : x.v.v) {
      if (const auto *designator{
              std::get_if<parser::Designator>(&accObject.u)}) {
        if (const auto *name{parser::GetDesignatorNameIfDataRef(*designator)}) {
          if (name->symbol) {
            AddUseDeviceObject(*name->symbol, *name);
          }
        }
      }
    }
    return false;
  }

  void Post(const parser::Name &);

private:
  std::int64_t GetAssociatedLoopLevelFromClauses(const parser::AccClauseList &);
  bool HasForceCollapseModifier(const parser::AccClauseList &);

  Symbol::Flags dataSharingAttributeFlags{Symbol::Flag::AccShared,
      Symbol::Flag::AccPrivate, Symbol::Flag::AccFirstPrivate,
      Symbol::Flag::AccReduction};

  Symbol::Flags dataMappingAttributeFlags{Symbol::Flag::AccCreate,
      Symbol::Flag::AccCopyIn, Symbol::Flag::AccCopyOut,
      Symbol::Flag::AccDelete, Symbol::Flag::AccPresent};

  Symbol::Flags accDataMvtFlags{
      Symbol::Flag::AccDevice, Symbol::Flag::AccHost, Symbol::Flag::AccSelf};

  Symbol::Flags accFlagsRequireMark{Symbol::Flag::AccCreate,
      Symbol::Flag::AccCopyIn, Symbol::Flag::AccCopyInReadOnly,
      Symbol::Flag::AccCopy, Symbol::Flag::AccCopyOut,
      Symbol::Flag::AccDevicePtr, Symbol::Flag::AccDeviceResident,
      Symbol::Flag::AccLink, Symbol::Flag::AccPresent};

  void CheckAssociatedLoop(const parser::DoConstruct &, bool forceCollapsed);
  void ResolveAccObjectList(const parser::AccObjectList &, Symbol::Flag);
  void ResolveAccObject(const parser::AccObject &, Symbol::Flag);
  Symbol *ResolveAcc(const parser::Name &, Symbol::Flag, Scope &);
  Symbol *ResolveAcc(Symbol &, Symbol::Flag, Scope &);
  Symbol *ResolveName(const parser::Name &);
  Symbol *ResolveFctName(const parser::Name &);
  Symbol *ResolveAccCommonBlockName(const parser::Name *);
  Symbol *DeclareOrMarkOtherAccessEntity(const parser::Name &, Symbol::Flag);
  Symbol *DeclareOrMarkOtherAccessEntity(Symbol &, Symbol::Flag);
  void CheckMultipleAppearances(
      const parser::Name &, const Symbol &, Symbol::Flag);
  void AllowOnlyArrayAndSubArray(const parser::AccObjectList &objectList);
  void DoNotAllowAssumedSizedArray(const parser::AccObjectList &objectList);
  void AllowOnlyVariable(const parser::AccObject &object);
  void EnsureAllocatableOrPointer(
      const llvm::acc::Clause clause, const parser::AccObjectList &objectList);
  void AddRoutineInfoToSymbol(
      Symbol &, const parser::OpenACCRoutineConstruct &);

  // Track use_device variables and check for duplicates.
  // Emits an error if the object was already added.
  void AddUseDeviceObject(const Symbol &, const parser::Name &);
  void ClearUseDeviceObjects() { useDeviceObjects_.clear(); }
  UnorderedSymbolSet useDeviceObjects_;

  Scope *topScope_;
};

// Data-sharing and Data-mapping attributes for data-refs in OpenMP construct
class OmpAttributeVisitor : DirectiveAttributeVisitor<llvm::omp::Directive> {
public:
  explicit OmpAttributeVisitor(SemanticsContext &context)
      : DirectiveAttributeVisitor(context) {}

  static bool HasStaticStorageDuration(const Symbol &symbol) {
    auto &ultSym = symbol.GetUltimate();
    // Module-scope variable
    return ultSym.owner().kind() == Scope::Kind::Module ||
        // Data statement variable
        ultSym.flags().test(Symbol::Flag::InDataStmt) ||
        // Save attribute variable
        ultSym.attrs().test(Attr::SAVE) ||
        // Referenced in a common block
        ultSym.flags().test(Symbol::Flag::InCommonBlock);
  }

  static const Symbol &GetStorageOwner(const Symbol &symbol) {
    static auto getParent = [](const Symbol *s) -> const Symbol * {
      if (auto *details{s->detailsIf<UseDetails>()}) {
        return &details->symbol();
      } else if (auto *details{s->detailsIf<HostAssocDetails>()}) {
        return &details->symbol();
      } else {
        return nullptr;
      }
    };
    static auto isPrivate = [](const Symbol &symbol) {
      static const Symbol::Flags privatizing{Symbol::Flag::OmpPrivate,
          Symbol::Flag::OmpFirstPrivate, Symbol::Flag::OmpLastPrivate,
          Symbol::Flag::OmpLinear};
      return (symbol.flags() & privatizing).any();
    };

    const Symbol *sym = &symbol;
    while (true) {
      if (isPrivate(*sym)) {
        return *sym;
      }
      if (const Symbol *parent{getParent(sym)}) {
        sym = parent;
      } else {
        return *sym;
      }
    }
    llvm_unreachable("Error while looking for storage owning symbol");
  }

  // Recognize symbols that are not created as a part of the OpenMP data-
  // sharing processing, and that are declared inside of the construct.
  // These symbols are predetermined private, but they shouldn't be marked
  // in any special way, because there is nothing to be done for them.
  // They are not symbols for which private copies need to be created,
  // they are already themselves private.
  static bool IsLocalInsideScope(const Symbol &symbol, const Scope &scope) {
    // A symbol that is marked with a DSA will be cloned in the construct
    // scope and marked as host-associated. This applies to privatized symbols
    // as well even though they will have their own storage. They should be
    // considered local regardless of the status of the original symbol.
    const Symbol &actual{GetStorageOwner(symbol)};
    return actual.owner() != scope && scope.Contains(actual.owner()) &&
        !HasStaticStorageDuration(actual);
  }

  template <typename A> void Walk(const A &x) { parser::Walk(x, *this); }
  // Normally the catch-all Pre/Post functions are templates taking
  // "const T &". For a class D derived from B, and an explicit overload
  // of Pre(const B &), a call to Pre(D) will select the template instead
  // of the base clase overload.
  // Force user-defined conversion from any const-reference, to make sure
  // that the Pre(AbsorbAnyReference) and Post(AbsorbAnyReference) overloads
  // will be worse than derived-to-base conversions. This will, for example,
  // invoke Pre(const OmpBlockConstruct &) for directives derived from it.
  struct AbsorbAnyReference {
    template <typename T> AbsorbAnyReference(const T &) {}
  };
  bool Pre(AbsorbAnyReference) { return true; }
  void Post(AbsorbAnyReference) {}

  bool Pre(const parser::SpecificationPart &) {
    partStack_.push_back(PartKind::SpecificationPart);
    return true;
  }
  void Post(const parser::SpecificationPart &) { partStack_.pop_back(); }

  bool Pre(const parser::ExecutionPart &) {
    partStack_.push_back(PartKind::ExecutionPart);
    return true;
  }
  void Post(const parser::ExecutionPart &) { partStack_.pop_back(); }

  bool Pre(const parser::StmtFunctionStmt &x) {
    const auto &parsedExpr{std::get<parser::Scalar<parser::Expr>>(x.t)};
    if (const auto *expr{GetExpr(context_, parsedExpr)}) {
      for (const Symbol &symbol : evaluate::CollectSymbols(*expr)) {
        if (!IsStmtFunctionDummy(symbol)) {
          stmtFunctionExprSymbols_.insert(symbol.GetUltimate());
        }
      }
    }
    return true;
  }

  bool Pre(const parser::UseStmt &x) {
    if (x.moduleName.symbol) {
      Scope &thisScope{context_.FindScope(x.moduleName.source)};
      common::visit(
          [&](auto &&details) {
            if constexpr (std::is_convertible_v<decltype(details),
                              const WithOmpDeclarative &>) {
              AddOmpRequiresToScope(thisScope, details.ompRequires(),
                  details.ompAtomicDefaultMemOrder());
            }
          },
          x.moduleName.symbol->details());
    }
    return true;
  }

  bool Pre(const parser::OmpStylizedDeclaration &x) {
    static llvm::StringMap<Symbol::Flag> map{
        {"omp_in", Symbol::Flag::OmpInVar},
        {"omp_orig", Symbol::Flag::OmpOrigVar},
        {"omp_out", Symbol::Flag::OmpOutVar},
        {"omp_priv", Symbol::Flag::OmpPrivVar},
    };
    if (auto &name{std::get<parser::ObjectName>(x.var.t)}; name.symbol) {
      if (auto found{map.find(name.ToString())}; found != map.end()) {
        ResolveOmp(name, found->second,
            const_cast<Scope &>(DEREF(name.symbol).owner()));
      }
    }
    return false;
  }
  bool Pre(const parser::OmpMetadirectiveDirective &x) {
    PushContext(x.v.source, llvm::omp::Directive::OMPD_metadirective);
    return true;
  }
  void Post(const parser::OmpMetadirectiveDirective &) { PopContext(); }

  bool Pre(const parser::OmpBlockConstruct &);
  void Post(const parser::OmpBlockConstruct &);

  void Post(const parser::OmpBeginDirective &x) {
    GetContext().withinConstruct = true;
  }

  bool Pre(const parser::OpenMPGroupprivate &);
  void Post(const parser::OpenMPGroupprivate &) { PopContext(); }

  bool Pre(const parser::OpenMPStandaloneConstruct &x) {
    common::visit(
        [&](auto &&s) {
          using TypeS = llvm::remove_cvref_t<decltype(s)>;
          // These two cases are handled individually.
          if constexpr ( //
              !std::is_same_v<TypeS, parser::OpenMPSimpleStandaloneConstruct> &&
              !std::is_same_v<TypeS, parser::OmpMetadirectiveDirective>) {
            PushContext(x.source, s.v.DirId());
          }
        },
        x.u);
    return true;
  }

  void Post(const parser::OpenMPStandaloneConstruct &x) {
    // These two cases are handled individually.
    if (!std::holds_alternative<parser::OpenMPSimpleStandaloneConstruct>(x.u) &&
        !std::holds_alternative<parser::OmpMetadirectiveDirective>(x.u)) {
      PopContext();
    }
  }

  bool Pre(const parser::OpenMPSimpleStandaloneConstruct &);
  void Post(const parser::OpenMPSimpleStandaloneConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPLoopConstruct &);
  void Post(const parser::OpenMPLoopConstruct &) { PopContext(); }
  bool Pre(const parser::OpenMPMisplacedEndDirective &x) { return false; }
  bool Pre(const parser::OpenMPInvalidDirective &x) { return false; }

  bool Pre(const parser::DoConstruct &);

  bool Pre(const parser::OpenMPSectionsConstruct &);
  void Post(const parser::OpenMPSectionsConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPSectionConstruct &);
  void Post(const parser::OpenMPSectionConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPCriticalConstruct &critical);
  void Post(const parser::OpenMPCriticalConstruct &) { PopContext(); }

  bool Pre(const parser::OmpDeclareSimdDirective &x) {
    PushContext(x.source, llvm::omp::Directive::OMPD_declare_simd);
    for (const parser::OmpArgument &arg : x.v.Arguments().v) {
      if (auto *object{parser::omp::GetArgumentObject(arg)}) {
        ResolveOmpObject(*object, Symbol::Flag::OmpDeclareSimd);
      }
    }
    return true;
  }
  void Post(const parser::OmpDeclareSimdDirective &) { PopContext(); }

  bool Pre(const parser::OpenMPDepobjConstruct &x) {
    PushContext(x.source, llvm::omp::Directive::OMPD_depobj);
    for (auto &arg : x.v.Arguments().v) {
      if (auto *object{parser::omp::GetArgumentObject(arg)}) {
        ResolveOmpObject(*object, Symbol::Flag::OmpDependObject);
      }
    }
    return true;
  }
  void Post(const parser::OpenMPDepobjConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPFlushConstruct &x) {
    PushContext(x.source, llvm::omp::Directive::OMPD_flush);
    for (auto &arg : x.v.Arguments().v) {
      if (auto *object{parser::omp::GetArgumentObject(arg)}) {
        if (auto *name{std::get_if<parser::Name>(&object->u)}) {
          // ResolveOmpCommonBlockName resolves the symbol as a side effect
          if (!ResolveOmpCommonBlockName(name)) {
            context_.Say(name->source, // 2.15.3
                "COMMON block must be declared in the same scoping unit "
                "in which the OpenMP directive or clause appears"_err_en_US);
          }
        }
      }
    }
    return true;
  }
  void Post(const parser::OpenMPFlushConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPRequiresConstruct &x) {
    using RequiresClauses = WithOmpDeclarative::RequiresClauses;
    PushContext(x.source, llvm::omp::Directive::OMPD_requires);

    auto getArgument{[&](auto &&maybeClause) {
      if (maybeClause) {
        // Scalar<Logical<Constant<common::Indirection<Expr>>>>
        auto &parserExpr{parser::UnwrapRef<parser::Expr>(*maybeClause)};
        evaluate::ExpressionAnalyzer ea{context_};
        if (auto &&maybeExpr{ea.Analyze(parserExpr)}) {
          if (auto v{omp::GetLogicalValue(*maybeExpr)}) {
            return *v;
          }
        }
      }
      // If the argument is missing, it is assumed to be true.
      return true;
    }};

    // Gather information from the clauses.
    RequiresClauses reqs;
    const common::OmpMemoryOrderType *memOrder{nullptr};
    for (const parser::OmpClause &clause : x.v.Clauses().v) {
      using OmpClause = parser::OmpClause;
      reqs |= common::visit(
          common::visitors{
              [&](const OmpClause::AtomicDefaultMemOrder &atomic) {
                memOrder = &atomic.v.v;
                return RequiresClauses{};
              },
              [&](auto &&s) {
                using TypeS = llvm::remove_cvref_t<decltype(s)>;
                if constexpr ( //
                    std::is_same_v<TypeS, OmpClause::DeviceSafesync> ||
                    std::is_same_v<TypeS, OmpClause::DynamicAllocators> ||
                    std::is_same_v<TypeS, OmpClause::ReverseOffload> ||
                    std::is_same_v<TypeS, OmpClause::SelfMaps> ||
                    std::is_same_v<TypeS, OmpClause::UnifiedAddress> ||
                    std::is_same_v<TypeS, OmpClause::UnifiedSharedMemory>) {
                  if (getArgument(s.v)) {
                    return RequiresClauses{clause.Id()};
                  }
                }
                return RequiresClauses{};
              },
          },
          clause.u);
    }

    // Merge clauses into parents' symbols details.
    AddOmpRequiresToScope(currScope(), &reqs, memOrder);
    return true;
  }
  void Post(const parser::OpenMPRequiresConstruct &) { PopContext(); }

  bool Pre(const parser::OmpDeclareTargetDirective &);
  void Post(const parser::OmpDeclareTargetDirective &) { PopContext(); }

  bool Pre(const parser::OmpDeclareMapperDirective &);
  void Post(const parser::OmpDeclareMapperDirective &) { PopContext(); }

  bool Pre(const parser::OmpDeclareReductionDirective &);
  void Post(const parser::OmpDeclareReductionDirective &) { PopContext(); }

  bool Pre(const parser::OpenMPThreadprivate &);
  void Post(const parser::OpenMPThreadprivate &) { PopContext(); }

  bool Pre(const parser::OmpAllocateDirective &);

  bool Pre(const parser::OpenMPAssumeConstruct &);
  void Post(const parser::OpenMPAssumeConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPAtomicConstruct &);
  void Post(const parser::OpenMPAtomicConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPDispatchConstruct &);
  void Post(const parser::OpenMPDispatchConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPAllocatorsConstruct &);
  void Post(const parser::OpenMPAllocatorsConstruct &);

  bool Pre(const parser::OpenMPUtilityConstruct &x) {
    PushContext(x.source, parser::omp::GetOmpDirectiveName(x).v);
    return true;
  }
  void Post(const parser::OpenMPUtilityConstruct &) { PopContext(); }

  bool Pre(const parser::OmpDeclareVariantDirective &x) {
    PushContext(x.source, llvm::omp::Directive::OMPD_declare_variant);
    return true;
  }
  void Post(const parser::OmpDeclareVariantDirective &) { PopContext(); };

  // 2.15.3 Data-Sharing Attribute Clauses
  bool Pre(const parser::OmpClause::Inclusive &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpInclusiveScan);
    return false;
  }
  bool Pre(const parser::OmpClause::Exclusive &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpExclusiveScan);
    return false;
  }
  void Post(const parser::OmpClause::Defaultmap &);
  void Post(const parser::OmpDefaultClause &);
  bool Pre(const parser::OmpClause::Shared &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpShared);
    return false;
  }
  bool Pre(const parser::OmpClause::Private &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpPrivate);
    return false;
  }
  bool Pre(const parser::OmpAllocateClause &x) {
    ResolveOmpObjectList(
        *parser::omp::GetOmpObjectList(x), Symbol::Flag::OmpAllocate);
    return false;
  }
  bool Pre(const parser::OmpClause::Firstprivate &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpFirstPrivate);
    return false;
  }
  bool Pre(const parser::OmpClause::Lastprivate &x) {
    ResolveOmpObjectList(
        *parser::omp::GetOmpObjectList(x), Symbol::Flag::OmpLastPrivate);
    return false;
  }
  bool Pre(const parser::OmpClause::Copyin &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpCopyIn);
    return false;
  }
  bool Pre(const parser::OmpClause::Copyprivate &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpCopyPrivate);
    return false;
  }
  bool Pre(const parser::OmpLinearClause &x) {
    ResolveOmpObjectList(
        *parser::omp::GetOmpObjectList(x), Symbol::Flag::OmpLinear);
    return false;
  }

  bool Pre(const parser::OmpClause::Uniform &x) {
    ResolveOmpNameList(x.v, Symbol::Flag::OmpUniform);
    return false;
  }

  bool Pre(const parser::OmpInReductionClause &x) {
    ResolveOmpObjectList(
        *parser::omp::GetOmpObjectList(x), Symbol::Flag::OmpInReduction);
    return false;
  }

  bool Pre(const parser::OmpClause::Reduction &x) {
    const auto &objList{*parser::omp::GetOmpObjectList(x)};
    ResolveOmpObjectList(objList, Symbol::Flag::OmpReduction);

    if (auto &modifiers{OmpGetModifiers(x.v)}) {
      auto createDummyProcSymbol = [&](const parser::Name *name) {
        // If name resolution failed, create a dummy symbol
        const auto namePair{currScope().try_emplace(
            name->source, Attrs{}, ProcEntityDetails{})};
        auto &newSymbol{*namePair.first->second};
        if (context_.intrinsics().IsIntrinsic(name->ToString())) {
          newSymbol.attrs().set(Attr::INTRINSIC);
        }
        name->symbol = &newSymbol;
      };

      for (auto &mod : *modifiers) {
        if (!std::holds_alternative<parser::OmpReductionIdentifier>(mod.u)) {
          continue;
        }
        auto &opr{std::get<parser::OmpReductionIdentifier>(mod.u)};
        if (auto *procD{parser::Unwrap<parser::ProcedureDesignator>(opr.u)}) {
          if (auto *name{parser::Unwrap<parser::Name>(procD->u)}) {
            if (!name->symbol) {
              if (!ResolveName(name)) {
                createDummyProcSymbol(name);
              }
            }
          }
          if (auto *procRef{
                  parser::Unwrap<parser::ProcComponentRef>(procD->u)}) {
            if (!procRef->v.thing.Component().symbol) {
              if (!ResolveName(&procRef->v.thing.Component())) {
                createDummyProcSymbol(&procRef->v.thing.Component());
              }
            }
          }
        }
      }
      using ReductionModifier = parser::OmpReductionModifier;
      if (auto *maybeModifier{
              OmpGetUniqueModifier<ReductionModifier>(modifiers)}) {
        if (maybeModifier->v == ReductionModifier::Value::Inscan) {
          ResolveOmpObjectList(objList, Symbol::Flag::OmpInScanReduction);
        }
      }
    }
    return false;
  }

  bool Pre(const parser::OmpAlignedClause &x) {
    ResolveOmpObjectList(
        *parser::omp::GetOmpObjectList(x), Symbol::Flag::OmpAligned);
    return false;
  }

  bool Pre(const parser::OmpClause::Nontemporal &x) {
    ResolveOmpObjectList(
        *parser::omp::GetOmpObjectList(x), Symbol::Flag::OmpNontemporal);
    return false;
  }

  void Post(const parser::OmpIteration &x) {
    if (const auto &name{std::get<parser::Name>(x.t)}; !name.symbol) {
      auto *symbol{currScope().FindSymbol(name.source)};
      if (!symbol) {
        // OmpIteration must use an existing object. If there isn't one,
        // create a fake one and flag an error later.
        symbol = &currScope().MakeSymbol(
            name.source, Attrs{}, EntityDetails(/*isDummy=*/true));
      }
      Resolve(name, symbol);
    }
  }

  bool Pre(const parser::OmpClause::UseDevicePtr &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpUseDevicePtr);
    return false;
  }

  bool Pre(const parser::OmpClause::UseDeviceAddr &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpUseDeviceAddr);
    return false;
  }

  bool Pre(const parser::OmpClause::IsDevicePtr &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpIsDevicePtr);
    return false;
  }

  bool Pre(const parser::OmpClause::HasDeviceAddr &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpHasDeviceAddr);
    return false;
  }

  void Post(const parser::Name &);

  void ResolveOmpObjectsForMapClause(
      Symbol::Flag mapFlag, const parser::OmpObjectList &objList) {
    for (const auto &ompObj : objList.v) {
      common::visit(
          common::visitors{
              [&](const parser::Designator &designator) {
                if (const auto *name{
                        parser::GetDesignatorNameIfDataRef(designator)}) {
                  if (name->symbol) {
                    name->symbol->set(mapFlag);
                  }
                }
              },
              [&](const auto &name) {},
          },
          ompObj.u);

      ResolveOmpObject(ompObj, mapFlag);
    }
  }

  void Post(const parser::OmpFromClause &x) {
    const auto &ompObjList{*parser::omp::GetOmpObjectList(x)};
    ResolveOmpObjectsForMapClause(Symbol::Flag::OmpMapFrom, ompObjList);
  }

  void Post(const parser::OmpToClause &x) {
    // This is different from the parser::OmpFromClause case, as this to applies
    // to both declare target to, and update to, so slightly different handling
    // is required in that we must exit early to avoid applying extra symbol
    // flags. This is reasonable for now, but if we wish to apply this
    // resolution to declare target to (and likely enter in another function
    // like this) we will have to extend the handling to act differently for
    // declare target rather than simply return.
    if (GetContext().directive == llvm::omp::Directive::OMPD_declare_target)
      return;

    const auto &ompObjList{*parser::omp::GetOmpObjectList(x)};
    ResolveOmpObjectsForMapClause(Symbol::Flag::OmpMapTo, ompObjList);
  }

  void Post(const parser::OmpMapClause &x) {
    unsigned version{context_.langOptions().OpenMPVersion};
    std::optional<Symbol::Flag> ompFlag;

    auto &mods{OmpGetModifiers(x)};
    if (auto *mapType{OmpGetUniqueModifier<parser::OmpMapType>(mods)}) {
      switch (mapType->v) {
      case parser::OmpMapType::Value::To:
        ompFlag = Symbol::Flag::OmpMapTo;
        break;
      case parser::OmpMapType::Value::From:
        ompFlag = Symbol::Flag::OmpMapFrom;
        break;
      case parser::OmpMapType::Value::Tofrom:
        ompFlag = Symbol::Flag::OmpMapToFrom;
        break;
      case parser::OmpMapType::Value::Alloc:
      case parser::OmpMapType::Value::Release:
      case parser::OmpMapType::Value::Storage:
        ompFlag = Symbol::Flag::OmpMapStorage;
        break;
      case parser::OmpMapType::Value::Delete:
        ompFlag = Symbol::Flag::OmpMapDelete;
        break;
      }
    }
    if (!ompFlag) {
      if (version >= 60) {
        // [6.0:275:12-15]
        // When a map-type is not specified for a clause on which it may be
        // specified, the map-type defaults to storage if the delete-modifier
        // is present on the clause or if the list item for which the map-type
        // is not specified is an assumed-size array.
        if (OmpGetUniqueModifier<parser::OmpDeleteModifier>(mods)) {
          ompFlag = Symbol::Flag::OmpMapStorage;
        }
        // Otherwise, if delete-modifier is absent, leave ompFlag unset.
      } else {
        // [5.2:151:10]
        // If a map-type is not specified, the map-type defaults to tofrom.
        ompFlag = Symbol::Flag::OmpMapToFrom;
      }
    }

    const auto &ompObjList{*parser::omp::GetOmpObjectList(x)};
    for (const auto &ompObj : ompObjList.v) {
      common::visit(
          common::visitors{
              [&](const parser::Designator &designator) {
                if (const auto *name{
                        parser::GetDesignatorNameIfDataRef(designator)}) {
                  if (name->symbol) {
                    name->symbol->set(
                        ompFlag.value_or(Symbol::Flag::OmpMapStorage));
                    AddToContextObjectWithDSA(*name->symbol,
                        ompFlag.value_or(Symbol::Flag::OmpMapStorage));
                  }
                }
              },
              [&](const auto &name) {},
          },
          ompObj.u);

      ResolveOmpObject(ompObj, ompFlag.value_or(Symbol::Flag::OmpMapStorage));
    }
  }

private:
  Symbol::Flags dataSharingAttributeFlags{Symbol::Flag::OmpShared,
      Symbol::Flag::OmpPrivate, Symbol::Flag::OmpFirstPrivate,
      Symbol::Flag::OmpLastPrivate, Symbol::Flag::OmpReduction,
      Symbol::Flag::OmpLinear};

  Symbol::Flags dataMappingAttributeFlags{Symbol::Flag::OmpMapTo,
      Symbol::Flag::OmpMapFrom, Symbol::Flag::OmpMapToFrom,
      Symbol::Flag::OmpMapStorage, Symbol::Flag::OmpMapDelete,
      Symbol::Flag::OmpIsDevicePtr, Symbol::Flag::OmpHasDeviceAddr};

  Symbol::Flags privateDataSharingAttributeFlags{Symbol::Flag::OmpPrivate,
      Symbol::Flag::OmpFirstPrivate, Symbol::Flag::OmpLastPrivate};

  Symbol::Flags ompFlagsRequireNewSymbol{Symbol::Flag::OmpPrivate,
      Symbol::Flag::OmpLinear, Symbol::Flag::OmpFirstPrivate,
      Symbol::Flag::OmpLastPrivate, Symbol::Flag::OmpShared,
      Symbol::Flag::OmpReduction, Symbol::Flag::OmpCriticalLock,
      Symbol::Flag::OmpCopyIn, Symbol::Flag::OmpUseDevicePtr,
      Symbol::Flag::OmpUseDeviceAddr, Symbol::Flag::OmpIsDevicePtr,
      Symbol::Flag::OmpHasDeviceAddr, Symbol::Flag::OmpUniform};

  Symbol::Flags ompFlagsRequireMark{Symbol::Flag::OmpThreadprivate,
      Symbol::Flag::OmpDeclareTarget, Symbol::Flag::OmpExclusiveScan,
      Symbol::Flag::OmpInclusiveScan, Symbol::Flag::OmpInScanReduction,
      Symbol::Flag::OmpGroupPrivate};

  Symbol::Flags dataCopyingAttributeFlags{
      Symbol::Flag::OmpCopyIn, Symbol::Flag::OmpCopyPrivate};

  UnorderedSymbolSet stmtFunctionExprSymbols_;

  enum class PartKind : int {
    // There are also other "parts", such as internal-subprogram-part, etc,
    // but we're keeping track of these two for now.
    SpecificationPart,
    ExecutionPart,
  };
  std::vector<PartKind> partStack_;

  // Predetermined DSA rules
  void PrivatizeAssociatedLoopIndex(const parser::OpenMPLoopConstruct &);
  void ResolveSeqLoopIndexInParallelOrTaskConstruct(const parser::Name &);

  bool IsNestedInDirective(llvm::omp::Directive directive);
  void ResolveOmpObjectList(const parser::OmpObjectList &, Symbol::Flag);
  void ResolveOmpDesignator(
      const parser::Designator &designator, Symbol::Flag ompFlag);
  void ResolveOmpCommonBlock(const parser::Name &name, Symbol::Flag ompFlag);
  void ResolveOmpObject(const parser::OmpObject &, Symbol::Flag);
  Symbol *ResolveOmp(const parser::Name &, Symbol::Flag, Scope &);
  Symbol *ResolveOmp(Symbol &, Symbol::Flag, Scope &);
  Symbol *ResolveOmpCommonBlockName(const parser::Name *);
  void ResolveOmpNameList(const std::list<parser::Name> &, Symbol::Flag);
  void ResolveOmpName(const parser::Name &, Symbol::Flag);
  void PropagateOmpFlagToEquivalenceSet(const Symbol &, Symbol::Flag);
  Symbol *ResolveName(const parser::Name *);
  Symbol *DeclareOrMarkOtherAccessEntity(const parser::Name &, Symbol::Flag);
  Symbol *DeclareOrMarkOtherAccessEntity(Symbol &, Symbol::Flag);
  void CheckMultipleAppearances(
      const parser::Name &, const Symbol &, Symbol::Flag);

  void CheckDataCopyingClause(
      const parser::Name &, const Symbol &, Symbol::Flag);
  void CheckObjectIsPrivatizable(
      const parser::Name &, const Symbol &, Symbol::Flag);

  void AddOmpRequiresToScope(Scope &,
      const WithOmpDeclarative::RequiresClauses *,
      const common::OmpMemoryOrderType *);

  void CreateImplicitSymbols(const parser::Name &, const Symbol *symbol);

  void AddToContextObjectWithExplicitDSA(Symbol &symbol, Symbol::Flag flag) {
    AddToContextObjectWithDSA(symbol, flag);
    if (dataSharingAttributeFlags.test(flag)) {
      symbol.set(Symbol::Flag::OmpExplicit);
    }
  }

  // Clear any previous data-sharing attribute flags and set the new ones.
  // Needed when setting PreDetermined DSAs, that take precedence over
  // Implicit ones.
  void SetSymbolDSA(Symbol &symbol, Symbol::Flags flags) {
    symbol.flags() &= ~(dataSharingAttributeFlags |
        Symbol::Flags{Symbol::Flag::OmpExplicit, Symbol::Flag::OmpImplicit,
            Symbol::Flag::OmpPreDetermined});
    symbol.flags() |= flags;
  }
};

void ResolveAccParts(SemanticsContext &context, const parser::ProgramUnit &node,
    Scope *topScope) {
  if (context.IsEnabled(common::LanguageFeature::OpenACC)) {
    AccAttributeVisitor{context, topScope}.Walk(node);
  }
}

void ResolveOmpParts(
    SemanticsContext &context, const parser::ProgramUnit &node) {
  if (context.IsEnabled(common::LanguageFeature::OpenMP)) {
    OmpAttributeVisitor{context}.Walk(node);
    if (!context.AnyFatalError()) {
      // The data-sharing attribute of the loop iteration variable for a
      // sequential loop (2.15.1.1) can only be determined when visiting
      // the corresponding DoConstruct, a second walk is to adjust the
      // symbols for all the data-refs of that loop iteration variable
      // prior to the DoConstruct.
      OmpAttributeVisitor{context}.Walk(node);
    }
  }
}

template <typename T>
bool DirectiveAttributeVisitor<T>::HasDataSharingAttributeObject(
    const Symbol &object) {
  auto it{dataSharingAttributeObjects_.find(object)};
  return it != dataSharingAttributeObjects_.end();
}

template <typename T>
std::tuple<const parser::Name *, const parser::ScalarExpr *,
    const parser::ScalarExpr *, const parser::ScalarExpr *>
DirectiveAttributeVisitor<T>::GetLoopBounds(const parser::DoConstruct &x) {
  using Bounds = parser::LoopControl::Bounds;
  if (x.GetLoopControl()) {
    if (const Bounds *b{std::get_if<Bounds>(&x.GetLoopControl()->u)}) {
      const auto &step = b->Step();
      return {&b->Name().thing, &b->Lower(), &b->Upper(),
          step.has_value() ? &step.value() : nullptr};
    }
  } else {
    context_
        .Say(std::get<parser::Statement<parser::NonLabelDoStmt>>(x.t).source,
            "Loop control is not present in the DO LOOP"_err_en_US)
        .Attach(GetContext().directiveSource,
            "associated with the enclosing LOOP construct"_en_US);
  }
  return {nullptr, nullptr, nullptr, nullptr};
}

template <typename T>
const parser::Name *DirectiveAttributeVisitor<T>::GetLoopIndex(
    const parser::DoConstruct &x) {
  return std::get<const parser::Name *>(GetLoopBounds(x));
}

template <typename T>
const parser::DoConstruct *DirectiveAttributeVisitor<T>::GetDoConstructIf(
    const parser::ExecutionPartConstruct &x) {
  return parser::Unwrap<parser::DoConstruct>(x);
}

template <typename T>
Symbol *DirectiveAttributeVisitor<T>::DeclareNewAccessEntity(
    const Symbol &object, Symbol::Flag flag, Scope &scope) {
  assert(object.owner() != currScope());
  auto &symbol{MakeAssocSymbol(object.name(), object, scope)};
  symbol.set(flag);
  if (flag == Symbol::Flag::OmpCopyIn) {
    // The symbol in copyin clause must be threadprivate entity.
    symbol.set(Symbol::Flag::OmpThreadprivate);
  }
  return &symbol;
}

template <typename T>
Symbol *DirectiveAttributeVisitor<T>::DeclareAccessEntity(
    const parser::Name &name, Symbol::Flag flag, Scope &scope) {
  if (!name.symbol) {
    return nullptr; // not resolved by Name Resolution step, do nothing
  }
  name.symbol = DeclareAccessEntity(*name.symbol, flag, scope);
  return name.symbol;
}

template <typename T>
Symbol *DirectiveAttributeVisitor<T>::DeclareAccessEntity(
    Symbol &object, Symbol::Flag flag, Scope &scope) {
  if (object.owner() != currScope()) {
    return DeclareNewAccessEntity(object, flag, scope);
  } else {
    object.set(flag);
    return &object;
  }
}

bool AccAttributeVisitor::Pre(const parser::OpenACCBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginBlockDirective>(x.t)};
  const auto &blockDir{std::get<parser::AccBlockDirective>(beginBlockDir.t)};
  switch (blockDir.v) {
  case llvm::acc::Directive::ACCD_data:
  case llvm::acc::Directive::ACCD_host_data:
  case llvm::acc::Directive::ACCD_kernels:
  case llvm::acc::Directive::ACCD_parallel:
  case llvm::acc::Directive::ACCD_serial:
    PushContext(blockDir.source, blockDir.v);
    break;
  default:
    break;
  }
  ClearDataSharingAttributeObjects();
  ClearUseDeviceObjects();
  return true;
}

bool AccAttributeVisitor::Pre(const parser::OpenACCDeclarativeConstruct &x) {
  if (const auto *declConstruct{
          std::get_if<parser::OpenACCStandaloneDeclarativeConstruct>(&x.u)}) {
    const auto &declDir{
        std::get<parser::AccDeclarativeDirective>(declConstruct->t)};
    PushContext(declDir.source, llvm::acc::Directive::ACCD_declare);
  }
  ClearDataSharingAttributeObjects();
  return true;
}

static const parser::AccObjectList &GetAccObjectList(
    const parser::AccClause &clause) {
  if (const auto *copyClause =
          std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
    return copyClause->v;
  } else if (const auto *createClause =
                 std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
    const Fortran::parser::AccObjectListWithModifier &listWithModifier =
        createClause->v;
    const Fortran::parser::AccObjectList &accObjectList =
        std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
    return accObjectList;
  } else if (const auto *copyinClause =
                 std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
    const Fortran::parser::AccObjectListWithModifier &listWithModifier =
        copyinClause->v;
    const Fortran::parser::AccObjectList &accObjectList =
        std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
    return accObjectList;
  } else if (const auto *copyoutClause =
                 std::get_if<Fortran::parser::AccClause::Copyout>(&clause.u)) {
    const Fortran::parser::AccObjectListWithModifier &listWithModifier =
        copyoutClause->v;
    const Fortran::parser::AccObjectList &accObjectList =
        std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
    return accObjectList;
  } else if (const auto *presentClause =
                 std::get_if<Fortran::parser::AccClause::Present>(&clause.u)) {
    return presentClause->v;
  } else if (const auto *deviceptrClause =
                 std::get_if<Fortran::parser::AccClause::Deviceptr>(
                     &clause.u)) {
    return deviceptrClause->v;
  } else if (const auto *deviceResidentClause =
                 std::get_if<Fortran::parser::AccClause::DeviceResident>(
                     &clause.u)) {
    return deviceResidentClause->v;
  } else if (const auto *linkClause =
                 std::get_if<Fortran::parser::AccClause::Link>(&clause.u)) {
    return linkClause->v;
  } else {
    llvm_unreachable("Clause without object list!");
  }
}

void AccAttributeVisitor::Post(
    const parser::OpenACCStandaloneDeclarativeConstruct &x) {
  const auto &clauseList = std::get<parser::AccClauseList>(x.t);
  for (const auto &clause : clauseList.v) {
    // Restriction - line 2414
    // We assume the restriction is present because clauses that require
    // moving data would require the size of the data to be present, but
    // the deviceptr and present clauses do not require moving data and
    // thus we permit them.
    if (!std::holds_alternative<parser::AccClause::Deviceptr>(clause.u) &&
        !std::holds_alternative<parser::AccClause::Present>(clause.u)) {
      DoNotAllowAssumedSizedArray(GetAccObjectList(clause));
    }
  }
}

bool AccAttributeVisitor::Pre(const parser::OpenACCLoopConstruct &x) {
  const auto &beginDir{std::get<parser::AccBeginLoopDirective>(x.t)};
  const auto &loopDir{std::get<parser::AccLoopDirective>(beginDir.t)};
  const auto &clauseList{std::get<parser::AccClauseList>(beginDir.t)};
  if (loopDir.v == llvm::acc::Directive::ACCD_loop) {
    PushContext(loopDir.source, loopDir.v);
  }
  ClearDataSharingAttributeObjects();
  SetContextAssociatedLoopLevel(GetAssociatedLoopLevelFromClauses(clauseList));
  const auto &outer{std::get<std::optional<parser::DoConstruct>>(x.t)};
  CheckAssociatedLoop(*outer, HasForceCollapseModifier(clauseList));
  return true;
}

bool AccAttributeVisitor::Pre(const parser::OpenACCStandaloneConstruct &x) {
  const auto &standaloneDir{std::get<parser::AccStandaloneDirective>(x.t)};
  switch (standaloneDir.v) {
  case llvm::acc::Directive::ACCD_enter_data:
  case llvm::acc::Directive::ACCD_exit_data:
  case llvm::acc::Directive::ACCD_init:
  case llvm::acc::Directive::ACCD_set:
  case llvm::acc::Directive::ACCD_shutdown:
  case llvm::acc::Directive::ACCD_update:
    PushContext(standaloneDir.source, standaloneDir.v);
    break;
  default:
    break;
  }
  ClearDataSharingAttributeObjects();
  return true;
}

Symbol *AccAttributeVisitor::ResolveName(const parser::Name &name) {
  return name.symbol;
}

Symbol *AccAttributeVisitor::ResolveFctName(const parser::Name &name) {
  Symbol *prev{currScope().FindSymbol(name.source)};
  if (prev && prev->IsFuncResult()) {
    prev = currScope().parent().FindSymbol(name.source);
  }
  if (!prev) {
    prev = &*context_.globalScope()
                 .try_emplace(name.source, ProcEntityDetails{})
                 .first->second;
  }
  CHECK(!name.symbol || name.symbol == prev);
  name.symbol = prev;
  return prev;
}

template <typename T>
common::IfNoLvalue<T, T> FoldExpr(
    evaluate::FoldingContext &foldingContext, T &&expr) {
  return evaluate::Fold(foldingContext, std::move(expr));
}

template <typename T>
MaybeExpr EvaluateExpr(
    Fortran::semantics::SemanticsContext &semanticsContext, const T &expr) {
  return FoldExpr(
      semanticsContext.foldingContext(), AnalyzeExpr(semanticsContext, expr));
}

void AccAttributeVisitor::AddRoutineInfoToSymbol(
    Symbol &symbol, const parser::OpenACCRoutineConstruct &x) {
  if (!symbol.has<SubprogramDetails>() && !symbol.has<ProcEntityDetails>())
    return;
  Fortran::semantics::OpenACCRoutineInfo info;
  std::vector<OpenACCRoutineDeviceTypeInfo *> currentDevices;
  currentDevices.push_back(&info);
  const auto &clauses{std::get<Fortran::parser::AccClauseList>(x.t)};
  for (const Fortran::parser::AccClause &clause : clauses.v) {
    if (const auto *dTypeClause{
            std::get_if<Fortran::parser::AccClause::DeviceType>(&clause.u)}) {
      currentDevices.clear();
      for (const auto &deviceTypeExpr : dTypeClause->v.v) {
        currentDevices.push_back(&info.add_deviceTypeInfo(deviceTypeExpr.v));
      }
    } else if (std::get_if<Fortran::parser::AccClause::Nohost>(&clause.u)) {
      info.set_isNohost();
    } else if (std::get_if<Fortran::parser::AccClause::Seq>(&clause.u)) {
      for (auto &device : currentDevices) {
        device->set_isSeq();
      }
    } else if (std::get_if<Fortran::parser::AccClause::Vector>(&clause.u)) {
      for (auto &device : currentDevices) {
        device->set_isVector();
      }
    } else if (std::get_if<Fortran::parser::AccClause::Worker>(&clause.u)) {
      for (auto &device : currentDevices) {
        device->set_isWorker();
      }
    } else if (const auto *gangClause{
                   std::get_if<Fortran::parser::AccClause::Gang>(&clause.u)}) {
      for (auto &device : currentDevices) {
        device->set_isGang();
      }
      if (gangClause->v) {
        const Fortran::parser::AccGangArgList &x = *gangClause->v;
        int numArgs{0};
        for (const Fortran::parser::AccGangArg &gangArg : x.v) {
          CHECK(numArgs <= 1 && "expecting 0 or 1 gang dim args");
          if (const auto *dim{
                  std::get_if<Fortran::parser::AccGangArg::Dim>(&gangArg.u)}) {
            if (const auto v{EvaluateInt64(context_, dim->v)}) {
              for (auto &device : currentDevices) {
                device->set_gangDim(*v);
              }
            }
          }
          numArgs++;
        }
      }
    } else if (const auto *bindClause{
                   std::get_if<Fortran::parser::AccClause::Bind>(&clause.u)}) {
      if (const auto *name{
              std::get_if<Fortran::parser::Name>(&bindClause->v.u)}) {
        if (Symbol * sym{ResolveFctName(*name)}) {
          Symbol &ultimate{sym->GetUltimate()};
          for (auto &device : currentDevices) {
            device->set_bindName(SymbolRef{ultimate});
          }
        } else {
          context_.Say((*name).source,
              "No function or subroutine declared for '%s'"_err_en_US,
              (*name).source);
        }
      } else if (const auto charExpr{
                     std::get_if<Fortran::parser::ScalarDefaultCharExpr>(
                         &bindClause->v.u)}) {
        auto *charConst{
            Fortran::parser::Unwrap<Fortran::parser::CharLiteralConstant>(
                *charExpr)};
        std::string str{std::get<std::string>(charConst->t)};
        for (auto &device : currentDevices) {
          device->set_bindName(std::string(str));
        }
      }
    }
  }
  if (symbol.has<SubprogramDetails>())
    symbol.get<SubprogramDetails>().add_openACCRoutineInfo(info);
  else
    symbol.get<ProcEntityDetails>().add_openACCRoutineInfo(info);
}

bool AccAttributeVisitor::Pre(const parser::OpenACCRoutineConstruct &x) {
  const auto &verbatim{std::get<parser::Verbatim>(x.t)};
  if (topScope_) {
    PushContext(
        verbatim.source, llvm::acc::Directive::ACCD_routine, *topScope_);
  } else {
    PushContext(verbatim.source, llvm::acc::Directive::ACCD_routine);
  }
  if (const auto &optName{std::get<std::optional<parser::Name>>(x.t)}) {
    if (Symbol * sym{ResolveFctName(*optName)}) {
      Symbol &ultimate{sym->GetUltimate()};
      AddRoutineInfoToSymbol(ultimate, x);
    } else {
      context_.Say((*optName).source,
          "No function or subroutine declared for '%s'"_err_en_US,
          (*optName).source);
    }
  } else {
    if (currScope().symbol()) {
      AddRoutineInfoToSymbol(*currScope().symbol(), x);
    }
  }
  return true;
}

bool AccAttributeVisitor::Pre(const parser::AccBindClause &x) {
  if (const auto *name{std::get_if<parser::Name>(&x.u)}) {
    if (!ResolveFctName(*name)) {
      context_.Say(name->source,
          "No function or subroutine declared for '%s'"_err_en_US,
          name->source);
    }
  }
  return true;
}

bool AccAttributeVisitor::Pre(const parser::OpenACCCombinedConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginCombinedDirective>(x.t)};
  const auto &combinedDir{
      std::get<parser::AccCombinedDirective>(beginBlockDir.t)};
  switch (combinedDir.v) {
  case llvm::acc::Directive::ACCD_kernels_loop:
  case llvm::acc::Directive::ACCD_parallel_loop:
  case llvm::acc::Directive::ACCD_serial_loop:
    PushContext(x.source, combinedDir.v);
    break;
  default:
    break;
  }
  const auto &clauseList{std::get<parser::AccClauseList>(beginBlockDir.t)};
  SetContextAssociatedLoopLevel(GetAssociatedLoopLevelFromClauses(clauseList));
  const auto &outer{std::get<std::optional<parser::DoConstruct>>(x.t)};
  CheckAssociatedLoop(*outer, HasForceCollapseModifier(clauseList));
  ClearDataSharingAttributeObjects();
  return true;
}

static bool IsLastNameArray(const parser::Designator &designator) {
  const auto &name{GetLastName(designator)};
  const evaluate::DataRef dataRef{*(name.symbol)};
  return common::visit( //
      common::visitors{
          [](const evaluate::SymbolRef &ref) {
            return ref->Rank() > 0 ||
                ref->GetType()->category() == DeclTypeSpec::Numeric;
          },
          [](const evaluate::ArrayRef &aref) {
            return aref.base().IsSymbol() ||
                aref.base().GetComponent().base().Rank() == 0;
          },
          [](const auto &) { return false; },
      },
      dataRef.u);
}

void AccAttributeVisitor::AllowOnlyArrayAndSubArray(
    const parser::AccObjectList &objectList) {
  for (const auto &accObject : objectList.v) {
    common::visit(
        common::visitors{
            [&](const parser::Designator &designator) {
              if (!IsLastNameArray(designator)) {
                context_.Say(designator.source,
                    "Only array element or subarray are allowed in %s directive"_err_en_US,
                    parser::ToUpperCaseLetters(
                        llvm::acc::getOpenACCDirectiveName(
                            GetContext().directive)
                            .str()));
              }
            },
            [&](const auto &name) {
              context_.Say(name.source,
                  "Only array element or subarray are allowed in %s directive"_err_en_US,
                  parser::ToUpperCaseLetters(
                      llvm::acc::getOpenACCDirectiveName(GetContext().directive)
                          .str()));
            },
        },
        accObject.u);
  }
}

void AccAttributeVisitor::DoNotAllowAssumedSizedArray(
    const parser::AccObjectList &objectList) {
  for (const auto &accObject : objectList.v) {
    common::visit(
        common::visitors{
            [&](const parser::Designator &designator) {
              const auto &name{GetLastName(designator)};
              if (name.symbol && semantics::IsAssumedSizeArray(*name.symbol)) {
                context_.Say(designator.source,
                    "Assumed-size dummy arrays may not appear on the %s "
                    "directive"_err_en_US,
                    parser::ToUpperCaseLetters(
                        llvm::acc::getOpenACCDirectiveName(
                            GetContext().directive)
                            .str()));
              }
            },
            [&](const auto &name) {

            },
        },
        accObject.u);
  }
}

void AccAttributeVisitor::AllowOnlyVariable(const parser::AccObject &object) {
  common::visit(
      common::visitors{
          [&](const parser::Designator &designator) {
            const auto &name{GetLastName(designator)};
            if (name.symbol && !semantics::IsVariableName(*name.symbol) &&
                !semantics::IsNamedConstant(*name.symbol)) {
              context_.Say(designator.source,
                  "Only variables are allowed in data clauses on the %s "
                  "directive"_err_en_US,
                  parser::ToUpperCaseLetters(
                      llvm::acc::getOpenACCDirectiveName(GetContext().directive)
                          .str()));
            }
          },
          [&](const auto &name) {},
      },
      object.u);
}

bool AccAttributeVisitor::Pre(const parser::OpenACCWaitConstruct &x) {
  const auto &verbatim{std::get<parser::Verbatim>(x.t)};
  PushContext(verbatim.source, llvm::acc::Directive::ACCD_wait);
  ClearDataSharingAttributeObjects();
  return true;
}

bool AccAttributeVisitor::Pre(const parser::OpenACCAtomicConstruct &x) {
  const auto &verbatimSource = common::visit(
      common::visitors{
          [&](const parser::AccAtomicUpdate &atomic) {
            const auto &optVerbatim =
                std::get<std::optional<parser::Verbatim>>(atomic.t);
            return optVerbatim ? optVerbatim->source : x.source;
          },
          [&](const auto &atomic) {
            return std::get<parser::Verbatim>(atomic.t).source;
          },
      },
      x.u);
  PushContext(verbatimSource, llvm::acc::Directive::ACCD_atomic);
  ClearDataSharingAttributeObjects();
  return true;
}

bool AccAttributeVisitor::Pre(const parser::OpenACCCacheConstruct &x) {
  const auto &verbatim{std::get<parser::Verbatim>(x.t)};
  PushContext(verbatim.source, llvm::acc::Directive::ACCD_cache);
  ClearDataSharingAttributeObjects();

  const auto &objectListWithModifier =
      std::get<parser::AccObjectListWithModifier>(x.t);
  const auto &objectList =
      std::get<Fortran::parser::AccObjectList>(objectListWithModifier.t);

  // 2.10 Cache directive restriction: A var in a cache directive must be a
  // single array element or a simple subarray.
  AllowOnlyArrayAndSubArray(objectList);

  return true;
}

bool AccAttributeVisitor::HasForceCollapseModifier(
    const parser::AccClauseList &x) {
  for (const auto &clause : x.v) {
    if (const auto *collapseClause{
            std::get_if<parser::AccClause::Collapse>(&clause.u)}) {
      const parser::AccCollapseArg &arg = collapseClause->v;
      return std::get<bool>(arg.t);
    }
  }
  return false;
}

std::int64_t AccAttributeVisitor::GetAssociatedLoopLevelFromClauses(
    const parser::AccClauseList &x) {
  std::int64_t collapseLevel{0};
  for (const auto &clause : x.v) {
    if (const auto *collapseClause{
            std::get_if<parser::AccClause::Collapse>(&clause.u)}) {
      const parser::AccCollapseArg &arg = collapseClause->v;
      const auto &collapseValue{std::get<parser::ScalarIntConstantExpr>(arg.t)};
      if (const auto v{EvaluateInt64(context_, collapseValue)}) {
        collapseLevel = *v;
      }
    }
  }

  if (collapseLevel) {
    return collapseLevel;
  }
  return 1; // default is outermost loop
}

void AccAttributeVisitor::CheckAssociatedLoop(
    const parser::DoConstruct &outerDoConstruct, bool forceCollapsed) {
  std::int64_t level{GetContext().associatedLoopLevel};
  if (level <= 0) { // collapse value was negative or 0
    return;
  }

  const auto getNextDoConstruct =
      [this, forceCollapsed](const parser::Block &block,
          std::int64_t &level) -> const parser::DoConstruct * {
    for (const auto &entry : block) {
      if (const auto *doConstruct = GetDoConstructIf(entry)) {
        return doConstruct;
      } else if (parser::Unwrap<parser::CompilerDirective>(entry)) {
        // It is allowed to have a compiler directive associated with the loop.
        continue;
      } else if (const auto &accLoop{
                     parser::Unwrap<parser::OpenACCLoopConstruct>(entry)}) {
        if (level == 0)
          break;
        const auto &beginDir{
            std::get<parser::AccBeginLoopDirective>(accLoop->t)};
        context_.Say(beginDir.source,
            "LOOP directive not expected in COLLAPSE loop nest"_err_en_US);
        level = 0;
      } else {
        if (!forceCollapsed) {
          break;
        }
      }
    }
    return nullptr;
  };

  auto checkExprHasSymbols = [&](llvm::SmallVector<Symbol *> &ivs,
                                 semantics::UnorderedSymbolSet &symbols) {
    for (auto iv : ivs) {
      if (symbols.count(*iv) != 0) {
        context_.Say(GetContext().directiveSource,
            "Trip count must be computable and invariant"_err_en_US);
      }
    }
  };

  Symbol::Flag flag = Symbol::Flag::AccPrivate;
  llvm::SmallVector<Symbol *> ivs;

  // Iterate the index variables of one DoConstruct, calling fn(name, lower,
  // upper) for each: once for a regular do loop, once per control variable for
  // a do concurrent loop.  Null pointers signal a loop without valid bounds
  // (e.g. do while); the level must still be consumed.
  auto forEachIndex = [this](const parser::DoConstruct &loop, auto &&fn) {
    if (loop.IsDoConcurrent()) {
      const auto &loopControl{*loop.GetLoopControl()};
      const auto &concurrent{
          std::get<parser::LoopControl::Concurrent>(loopControl.u)};
      const auto &header{std::get<parser::ConcurrentHeader>(concurrent.t)};
      for (const auto &control :
          std::get<std::list<parser::ConcurrentControl>>(header.t)) {
        fn(&std::get<parser::Name>(control.t),
            &parser::UnwrapRef<parser::Expr>(std::get<1>(control.t)),
            &parser::UnwrapRef<parser::Expr>(std::get<2>(control.t)));
      }
    } else {
      auto bounds{GetLoopBounds(loop)};
      const parser::ScalarExpr *lower{std::get<1>(bounds)};
      const parser::ScalarExpr *upper{std::get<2>(bounds)};
      fn(std::get<0>(bounds),
          lower ? &parser::UnwrapRef<parser::Expr>(*lower) : nullptr,
          upper ? &parser::UnwrapRef<parser::Expr>(*upper) : nullptr);
    }
  };

  for (const parser::DoConstruct *loop{&outerDoConstruct}; loop && level > 0;) {
    forEachIndex(*loop,
        [&](const parser::Name *ivName, const parser::Expr *lower,
            const parser::Expr *upper) {
          if (level <= 0)
            return;
          if (ivName && lower && upper) {
            if (auto *symbol{ResolveAcc(*ivName, flag, currScope())}) {
              if (auto lowerExpr{semantics::AnalyzeExpr(context_, *lower)}) {
                semantics::UnorderedSymbolSet lowerSyms =
                    evaluate::CollectSymbols(*lowerExpr);
                checkExprHasSymbols(ivs, lowerSyms);
              }
              if (auto upperExpr{semantics::AnalyzeExpr(context_, *upper)}) {
                semantics::UnorderedSymbolSet upperSyms =
                    evaluate::CollectSymbols(*upperExpr);
                checkExprHasSymbols(ivs, upperSyms);
              }
              ivs.push_back(symbol);
            }
          }
          --level;
        });

    const auto &block{std::get<parser::Block>(loop->t)};
    loop = getNextDoConstruct(block, level);
  }

  if (level != 0) {
    context_.Say(GetContext().directiveSource,
        "Not enough %s for COLLAPSE(%jd) clause, found %jd, expected %jd more"_err_en_US,
        forceCollapsed ? "nested loops" : "perfectly nested loops",
        GetContext().associatedLoopLevel,
        GetContext().associatedLoopLevel - level, level);
  }
}

void AccAttributeVisitor::EnsureAllocatableOrPointer(
    const llvm::acc::Clause clause, const parser::AccObjectList &objectList) {
  for (const auto &accObject : objectList.v) {
    common::visit(
        common::visitors{
            [&](const parser::Designator &designator) {
              const auto &lastName{GetLastName(designator)};
              if (!IsAllocatableOrObjectPointer(lastName.symbol)) {
                context_.Say(designator.source,
                    "Argument `%s` on the %s clause must be a variable or "
                    "array with the POINTER or ALLOCATABLE attribute"_err_en_US,
                    lastName.symbol->name(),
                    parser::ToUpperCaseLetters(
                        llvm::acc::getOpenACCClauseName(clause).str()));
              }
            },
            [&](const auto &name) {
              context_.Say(name.source,
                  "Argument on the %s clause must be a variable or "
                  "array with the POINTER or ALLOCATABLE attribute"_err_en_US,
                  parser::ToUpperCaseLetters(
                      llvm::acc::getOpenACCClauseName(clause).str()));
            },
        },
        accObject.u);
  }
}

bool AccAttributeVisitor::Pre(const parser::AccClause::Attach &x) {
  // Restriction - line 1708-1709
  EnsureAllocatableOrPointer(llvm::acc::Clause::ACCC_attach, x.v);
  return true;
}

bool AccAttributeVisitor::Pre(const parser::AccClause::Detach &x) {
  // Restriction - line 1715-1717
  EnsureAllocatableOrPointer(llvm::acc::Clause::ACCC_detach, x.v);
  return true;
}

void AccAttributeVisitor::Post(const parser::AccDefaultClause &x) {
  if (!dirContext_.empty()) {
    switch (x.v) {
    case llvm::acc::DefaultValue::ACC_Default_present:
      SetContextDefaultDSA(Symbol::Flag::AccPresent);
      break;
    case llvm::acc::DefaultValue::ACC_Default_none:
      SetContextDefaultDSA(Symbol::Flag::AccNone);
      break;
    }
  }
}

void AccAttributeVisitor::Post(const parser::Name &name) {
  if (name.symbol && WithinConstruct()) {
    const Symbol &symbol{name.symbol->GetUltimate()};
    if (!symbol.owner().IsDerivedType() && !symbol.has<ProcEntityDetails>() &&
        !symbol.has<SubprogramDetails>() && !IsObjectWithVisibleDSA(symbol) &&
        !symbol.has<AssocEntityDetails>() && !symbol.has<MiscDetails>()) {
      if (Symbol * found{currScope().FindSymbol(name.source)}) {
        if (&symbol != found) {
          // adjust the symbol within the region
          // TODO: why didn't name resolution set the right name originally?
          name.symbol = found;
        } else if (GetContext().defaultDSA == Symbol::Flag::AccNone) {
          // 2.5.14.
          context_.Say(name.source,
              "The DEFAULT(NONE) clause requires that '%s' must be listed in a data-mapping clause"_err_en_US,
              symbol.name());
        }
      } else {
        // TODO: assertion here?  or clear name.symbol?
      }
    }
  }
}

Symbol *AccAttributeVisitor::ResolveAccCommonBlockName(
    const parser::Name *name) {
  if (name) {
    if (Symbol *
        cb{GetContext().scope.FindCommonBlockInVisibleScopes(name->source)}) {
      name->symbol = cb;
      return cb;
    }
  }
  return nullptr;
}

void AccAttributeVisitor::AddUseDeviceObject(
    const Symbol &object, const parser::Name &name) {
  if (!useDeviceObjects_.insert(object).second) {
    context_.Say(name.source,
        "'%s' appears in more than one USE_DEVICE clause on the same HOST_DATA directive"_err_en_US,
        name.ToString());
  }
}

void AccAttributeVisitor::ResolveAccObjectList(
    const parser::AccObjectList &accObjectList, Symbol::Flag accFlag) {
  for (const auto &accObject : accObjectList.v) {
    AllowOnlyVariable(accObject);
    ResolveAccObject(accObject, accFlag);
  }
}

void AccAttributeVisitor::ResolveAccObject(
    const parser::AccObject &accObject, Symbol::Flag accFlag) {
  common::visit(
      common::visitors{
          [&](const parser::Designator &designator) {
            if (const auto *name{
                    parser::GetDesignatorNameIfDataRef(designator)}) {
              if (auto *symbol{ResolveAcc(*name, accFlag, currScope())}) {
                AddToContextObjectWithDSA(*symbol, accFlag);
                if (dataSharingAttributeFlags.test(accFlag)) {
                  CheckMultipleAppearances(*name, *symbol, accFlag);
                }
              }
            } else {
              // Array sections to be changed to substrings as needed
              if (AnalyzeExpr(context_, designator)) {
                if (std::holds_alternative<parser::Substring>(designator.u)) {
                  context_.Say(designator.source,
                      "Substrings are not allowed on OpenACC "
                      "directives or clauses"_err_en_US);
                }
              }
              // other checks, more TBD
            }
          },
          [&](const parser::Name &name) { // common block
            if (auto *symbol{ResolveAccCommonBlockName(&name)}) {
              CheckMultipleAppearances(
                  name, *symbol, Symbol::Flag::AccCommonBlock);
              for (auto &object : symbol->get<CommonBlockDetails>().objects()) {
                if (auto *resolvedObject{
                        ResolveAcc(*object, accFlag, currScope())}) {
                  AddToContextObjectWithDSA(*resolvedObject, accFlag);
                }
              }
            } else {
              context_.Say(name.source,
                  "Could not find COMMON block '%s' used in OpenACC directive"_err_en_US,
                  name.ToString());
            }
          },
      },
      accObject.u);
}

Symbol *AccAttributeVisitor::ResolveAcc(
    const parser::Name &name, Symbol::Flag accFlag, Scope &scope) {
  return DeclareOrMarkOtherAccessEntity(name, accFlag);
}

Symbol *AccAttributeVisitor::ResolveAcc(
    Symbol &symbol, Symbol::Flag accFlag, Scope &scope) {
  return DeclareOrMarkOtherAccessEntity(symbol, accFlag);
}

Symbol *AccAttributeVisitor::DeclareOrMarkOtherAccessEntity(
    const parser::Name &name, Symbol::Flag accFlag) {
  if (name.symbol) {
    return DeclareOrMarkOtherAccessEntity(*name.symbol, accFlag);
  } else {
    return nullptr;
  }
}

Symbol *AccAttributeVisitor::DeclareOrMarkOtherAccessEntity(
    Symbol &object, Symbol::Flag accFlag) {
  if (accFlagsRequireMark.test(accFlag)) {
    if (GetContext().directive == llvm::acc::ACCD_declare) {
      object.set(Symbol::Flag::AccDeclare);
      object.set(accFlag);
    }
  }
  return &object;
}

static bool WithMultipleAppearancesAccException(
    const Symbol &symbol, Symbol::Flag flag) {
  return false; // Place holder
}

void AccAttributeVisitor::CheckMultipleAppearances(
    const parser::Name &name, const Symbol &symbol, Symbol::Flag accFlag) {
  const auto *target{&symbol};
  if (HasDataSharingAttributeObject(*target) &&
      !WithMultipleAppearancesAccException(symbol, accFlag)) {
    context_.Say(name.source,
        "'%s' appears in more than one data-sharing clause "
        "on the same OpenACC directive"_err_en_US,
        name.ToString());
  } else {
    AddDataSharingAttributeObject(*target);
  }
}

#ifndef NDEBUG

#define DEBUG_TYPE "omp"

static llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const Symbol::Flags &flags);

namespace dbg {
static void DumpAssocSymbols(llvm::raw_ostream &os, const Symbol &sym);
static std::string ScopeSourcePos(const Fortran::semantics::Scope &scope);
} // namespace dbg

#endif

bool OmpAttributeVisitor::Pre(const parser::OmpBlockConstruct &x) {
  const parser::OmpDirectiveSpecification &dirSpec{x.BeginDir()};
  llvm::omp::Directive dirId{dirSpec.DirId()};
  PushContext(dirSpec.source, dirId);
  ClearDataSharingAttributeObjects();
  return true;
}

void OmpAttributeVisitor::Post(const parser::OmpBlockConstruct &x) {
  PopContext();
}

bool OmpAttributeVisitor::Pre(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  const auto &standaloneDir{std::get<parser::OmpDirectiveName>(x.v.t)};
  PushContext(standaloneDir.source, standaloneDir.v);
  ClearDataSharingAttributeObjects();
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPLoopConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};
  const parser::OmpDirectiveName &beginName{beginSpec.DirName()};
  PushContext(beginName.source, beginName.v);
  ClearDataSharingAttributeObjects();

  if (beginName.v == llvm::omp::Directive::OMPD_do) {
    if (const parser::DoConstruct *doConstruct{x.GetNestedLoop()}) {
      if (doConstruct->IsDoWhile()) {
        return true;
      }
    }
  }

  PrivatizeAssociatedLoopIndex(x);
  return true;
}

void OmpAttributeVisitor::ResolveSeqLoopIndexInParallelOrTaskConstruct(
    const parser::Name &iv) {
  unsigned version{context_.langOptions().OpenMPVersion};
  // Find the parallel, teams or task generating construct enclosing the
  // sequential loop.
  auto targetIt{dirContext_.rbegin()};
  for (;; ++targetIt) {
    if (targetIt == dirContext_.rend()) {
      return;
    }
    if (llvm::omp::allParallelSet.test(targetIt->directive) ||
        llvm::omp::taskGeneratingSet.test(targetIt->directive)) {
      break;
    }
    if (version >= 52) {
      if (llvm::omp::allTeamsSet.test(targetIt->directive)) {
        break;
      }
    }
  }
  if (IsLocalInsideScope(*iv.symbol, targetIt->scope)) {
    return;
  }
  // If this symbol already has a data-sharing attribute then there is nothing
  // to do here.
  if (const Symbol *symbol{iv.symbol}) {
    for (auto symMap : targetIt->objectWithDSA) {
      if (symMap.first->name() == symbol->name()) {
        return;
      }
    }
  }
  // If this symbol already has an explicit data-sharing attribute in the
  // enclosing OpenMP parallel or task then there is nothing to do here.
  if (auto *symbol{targetIt->scope.FindSymbol(iv.source)}) {
    if (symbol->owner() == targetIt->scope) {
      if (symbol->test(Symbol::Flag::OmpExplicit) &&
          (symbol->flags() & dataSharingAttributeFlags).any()) {
        return;
      }
    }
  }
  // Otherwise find the symbol and make it Private for the entire enclosing
  // parallel or task
  if (auto *symbol{ResolveOmp(iv, Symbol::Flag::OmpPrivate, targetIt->scope)}) {
    targetIt++;
    SetSymbolDSA(
        *symbol, {Symbol::Flag::OmpPreDetermined, Symbol::Flag::OmpPrivate});
    iv.symbol = symbol; // adjust the symbol within region
    for (auto it{dirContext_.rbegin()}; it != targetIt; ++it) {
      AddToContextObjectWithDSA(*symbol, Symbol::Flag::OmpPrivate, *it);
    }
  }
}

// [OMP-4.5]2.15.1.1 Data-sharing Attribute Rules - Predetermined
//   - A loop iteration variable for a sequential loop in a parallel
//     or task generating construct is private in the innermost such
//     construct that encloses the loop
// Loop iteration variables are not well defined for DO WHILE loop.
// Use of DO CONCURRENT inside OpenMP construct is unspecified behavior
// till OpenMP-5.0 standard.
// In above both cases we skip the privatization of iteration variables.
bool OmpAttributeVisitor::Pre(const parser::DoConstruct &x) {
  if (WithinConstruct()) {
    llvm::SmallVector<const parser::Name *> ivs;
    if (x.IsDoNormal()) {
      const parser::Name *iv{GetLoopIndex(x)};
      if (iv && iv->symbol)
        ivs.push_back(iv);
    }
    for (auto iv : ivs) {
      if (!iv->symbol->test(Symbol::Flag::OmpPreDetermined)) {
        ResolveSeqLoopIndexInParallelOrTaskConstruct(*iv);
      } else {
        // TODO: conflict checks with explicitly determined DSA
      }
    }
  }
  return true;
}

// 2.15.1.1 Data-sharing Attribute Rules - Predetermined
//   - The loop iteration variable(s) in the associated do-loop(s) of a do,
//     parallel do, taskloop, or distribute construct is (are) private.
//   - The loop iteration variable in the associated do-loop of a simd construct
//     with just one associated do-loop is linear with a linear-step that is the
//     increment of the associated do-loop (only for OpenMP versions <= 4.5)
//   - The loop iteration variables in the associated do-loops of a simd
//     construct with multiple associated do-loops are lastprivate.
void OmpAttributeVisitor::PrivatizeAssociatedLoopIndex(
    const parser::OpenMPLoopConstruct &x) {
  const parser::OmpDirectiveSpecification &spec{x.BeginDir()};
  unsigned version{context_.langOptions().OpenMPVersion};

  auto [depth, _]{
      omp::GetAffectedNestDepthWithReason(spec, version, &context_)};

  // If depth is absent, then there is some issue. Leave it alone here,
  // and let the semantic checks diagnose the problem.
  if (!depth || *depth.value <= 0) {
    return;
  }

  int64_t level{*depth.value};
  Symbol::Flag ivDSA;
  if (!llvm::omp::allSimdSet.test(GetContext().directive)) {
    ivDSA = Symbol::Flag::OmpPrivate;
  } else if (level == 1 && version < 60) {
    ivDSA = Symbol::Flag::OmpLinear;
  } else {
    ivDSA = Symbol::Flag::OmpLastPrivate;
  }

  Scope &scope{currScope()};

  if (auto doLoops{omp::CollectAffectedDoLoops(x, version, &context_)}) {
    for (const parser::DoConstruct *loop : *doLoops) {
      const parser::Name *iv{GetLoopIndex(*loop)};
      if (!iv || (iv->symbol && IsLocalInsideScope(*iv->symbol, scope))) {
        continue;
      }
      if (auto *symbol{ResolveOmp(*iv, ivDSA, scope)}) {
        SetSymbolDSA(*symbol, {Symbol::Flag::OmpPreDetermined, ivDSA});
        iv->symbol = symbol; // adjust the symbol within region
        AddToContextObjectWithDSA(*symbol, ivDSA);
      }
    }
  }
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPGroupprivate &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_groupprivate);
  for (const parser::OmpArgument &arg : x.v.Arguments().v) {
    if (auto *object{parser::omp::GetArgumentObject(arg)}) {
      ResolveOmpObject(*object, Symbol::Flag::OmpGroupPrivate);
    }
  }
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPSectionsConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};
  const parser::OmpDirectiveName &beginName{beginSpec.DirName()};
  switch (beginName.v) {
  case llvm::omp::Directive::OMPD_parallel_sections:
  case llvm::omp::Directive::OMPD_sections:
    PushContext(beginName.source, beginName.v);
    GetContext().withinConstruct = true;
    break;
  default:
    break;
  }
  ClearDataSharingAttributeObjects();
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPSectionConstruct &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_section);
  GetContext().withinConstruct = true;
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPCriticalConstruct &x) {
  const parser::OmpBeginDirective &beginSpec{x.BeginDir()};
  PushContext(beginSpec.DirName().source, beginSpec.DirId());
  GetContext().withinConstruct = true;
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OmpDeclareTargetDirective &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_declare_target);

  for (const parser::OmpArgument &arg : x.v.Arguments().v) {
    if (auto *object{parser::omp::GetArgumentObject(arg)}) {
      ResolveOmpObject(*object, Symbol::Flag::OmpDeclareTarget);
    }
  }

  for (const parser::OmpClause &clause : x.v.Clauses().v) {
    if (auto *objects{parser::omp::GetOmpObjectList(clause)}) {
      for (const parser::OmpObject &object : objects->v) {
        ResolveOmpObject(object, Symbol::Flag::OmpDeclareTarget);
      }
    }
  }
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OmpDeclareMapperDirective &x) {
  const parser::OmpDirectiveName &dirName{x.v.DirName()};
  PushContext(dirName.source, dirName.v);
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OmpDeclareReductionDirective &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_declare_reduction);
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPThreadprivate &x) {
  const parser::OmpDirectiveName &dirName{x.v.DirName()};
  PushContext(dirName.source, dirName.v);

  for (const parser::OmpArgument &arg : x.v.Arguments().v) {
    if (auto *object{parser::omp::GetArgumentObject(arg)}) {
      ResolveOmpObject(*object, Symbol::Flag::OmpThreadprivate);
    }
  }
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OmpAllocateDirective &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_allocate);
  assert(!partStack_.empty() && "Misplaced directive");

  auto ompFlag{partStack_.back() == PartKind::SpecificationPart
          ? Symbol::Flag::OmpDeclarativeAllocateDirective
          : Symbol::Flag::OmpExecutableAllocateDirective};

  parser::omp::OmpAllocateInfo info{parser::omp::SplitOmpAllocate(x)};
  for (const parser::OmpAllocateDirective *ad : info.dirs) {
    for (const parser::OmpArgument &arg : ad->BeginDir().Arguments().v) {
      if (auto *object{parser::omp::GetArgumentObject(arg)}) {
        ResolveOmpObject(*object, ompFlag);
      }
    }
  }

  PopContext();
  return false;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPAssumeConstruct &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_assume);
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPAtomicConstruct &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_atomic);
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPDispatchConstruct &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_dispatch);
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPAllocatorsConstruct &x) {
  const parser::OmpDirectiveSpecification &dirSpec{x.BeginDir()};
  PushContext(x.source, dirSpec.DirId());

  for (const auto &clause : dirSpec.Clauses().v) {
    if (std::get_if<parser::OmpClause::Allocate>(&clause.u)) {
      ResolveOmpObjectList(*parser::omp::GetOmpObjectList(clause),
          Symbol::Flag::OmpExecutableAllocateDirective);
    }
  }
  return true;
}

void OmpAttributeVisitor::Post(const parser::OmpClause::Defaultmap &x) {
  using ImplicitBehavior = parser::OmpDefaultmapClause::ImplicitBehavior;
  using VariableCategory = parser::OmpVariableCategory;

  VariableCategory::Value varCategory;
  ImplicitBehavior impBehavior;

  if (!dirContext_.empty()) {
    impBehavior = std::get<ImplicitBehavior>(x.v.t);

    auto &modifiers{OmpGetModifiers(x.v)};
    auto *maybeCategory{
        OmpGetUniqueModifier<parser::OmpVariableCategory>(modifiers)};
    if (maybeCategory)
      varCategory = maybeCategory->v;
    else
      varCategory = VariableCategory::Value::All;

    AddContextDefaultmapBehaviour(varCategory, impBehavior);
  }
}

void OmpAttributeVisitor::Post(const parser::OmpDefaultClause &x) {
  // The DEFAULT clause may also be used on METADIRECTIVE. In that case
  // there is nothing to do.
  using DataSharingAttribute = parser::OmpDefaultClause::DataSharingAttribute;
  if (auto *dsa{std::get_if<DataSharingAttribute>(&x.u)}) {
    if (!dirContext_.empty()) {
      switch (*dsa) {
      case DataSharingAttribute::Private:
        SetContextDefaultDSA(Symbol::Flag::OmpPrivate);
        break;
      case DataSharingAttribute::Firstprivate:
        SetContextDefaultDSA(Symbol::Flag::OmpFirstPrivate);
        break;
      case DataSharingAttribute::Shared:
        SetContextDefaultDSA(Symbol::Flag::OmpShared);
        break;
      case DataSharingAttribute::None:
        SetContextDefaultDSA(Symbol::Flag::OmpNone);
        break;
      }
    }
  }
}

bool OmpAttributeVisitor::IsNestedInDirective(llvm::omp::Directive directive) {
  if (dirContext_.size() >= 1) {
    for (std::size_t i = dirContext_.size() - 1; i > 0; --i) {
      if (dirContext_[i - 1].directive == directive) {
        return true;
      }
    }
  }
  return false;
}

void OmpAttributeVisitor::Post(const parser::OpenMPAllocatorsConstruct &x) {
  PopContext();
}

static bool IsTargetCaptureImplicitlyFirstprivatizeable(const Symbol &symbol,
    const Symbol::Flags &dsa, const Symbol::Flags &dataSharingAttributeFlags,
    const Symbol::Flags &dataMappingAttributeFlags,
    std::map<parser::OmpVariableCategory::Value,
        parser::OmpDefaultmapClause::ImplicitBehavior>
        defaultMap) {
  // If a Defaultmap clause is present for the current target scope, and it has
  // specified behaviour other than Firstprivate for scalars then we exit early,
  // as it overrides the implicit Firstprivatization of scalars OpenMP rule.
  if (!defaultMap.empty()) {
    if (llvm::is_contained(
            defaultMap, parser::OmpVariableCategory::Value::All) &&
        defaultMap[parser::OmpVariableCategory::Value::All] !=
            parser::OmpDefaultmapClause::ImplicitBehavior::Firstprivate) {
      return false;
    }

    if (llvm::is_contained(
            defaultMap, parser::OmpVariableCategory::Value::Scalar) &&
        defaultMap[parser::OmpVariableCategory::Value::Scalar] !=
            parser::OmpDefaultmapClause::ImplicitBehavior::Firstprivate) {
      return false;
    }
  }

  auto checkSymbol = [&](const Symbol &checkSym) {
    // if we're associated with any other flags we skip implicit privitization
    // for now. If we're an allocatable, pointer or declare target, we're not
    // implicitly firstprivitizeable under OpenMP restrictions.
    // TODO: Relax restriction as we progress privitization and further
    // investigate the flags we can intermix with.
    if (!(dsa & (dataSharingAttributeFlags | dataMappingAttributeFlags))
            .none() ||
        !checkSym.flags().none() || IsAssumedShape(checkSym) ||
        semantics::IsAllocatableOrPointer(checkSym)) {
      return false;
    }

    // It is default firstprivatizeable as far as the OpenMP specification is
    // concerned if it is a non-array scalar type that has been implicitly
    // captured in a target region
    const auto *type{checkSym.GetType()};
    if ((!checkSym.GetShape() || checkSym.GetShape()->empty()) &&
        (type->category() ==
                Fortran::semantics::DeclTypeSpec::Category::Numeric ||
            type->category() ==
                Fortran::semantics::DeclTypeSpec::Category::Logical ||
            type->category() ==
                Fortran::semantics::DeclTypeSpec::Category::Character)) {
      return true;
    }
    return false;
  };

  return common::visit(
      common::visitors{
          [&](const UseDetails &x) -> bool { return checkSymbol(x.symbol()); },
          [&](const HostAssocDetails &x) -> bool {
            return checkSymbol(x.symbol());
          },
          [&](const auto &) -> bool { return checkSymbol(symbol); },
      },
      symbol.details());
}

void OmpAttributeVisitor::CreateImplicitSymbols(
    const parser::Name &name, const Symbol *symbol) {
  if (!omp::IsPrivatizable(*symbol)) {
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "CreateImplicitSymbols: " << *symbol << '\n');

  // Implicitly determined DSAs
  // OMP 5.2 5.1.1 - Variables Referenced in a Construct
  Symbol *lastDeclSymbol = nullptr;
  Symbol::Flags prevDSA;
  bool checkDefaultNone = false;
  for (int dirDepth{0}; dirDepth < (int)dirContext_.size(); ++dirDepth) {
    DirContext &dirContext = dirContext_[dirDepth];
    Symbol::Flags dsa;

    Scope &scope{context_.FindScope(dirContext.directiveSource)};

    auto initSymbolDSA = [&](const Symbol *sym, Symbol::Flags &dsa) {
      auto it{scope.find(sym->name())};
      if (it != scope.end()) {
        // There is already a symbol in the current scope, use its DSA.
        dsa = GetSymbolDSA(*it->second);
      } else {
        for (auto symMap : dirContext.objectWithDSA) {
          if (symMap.first->name() == sym->name()) {
            // `sym` already has a data-sharing attribute in the current
            // context, use it.
            dsa.set(symMap.second);
            break;
          }
        }
      }
    };
    initSymbolDSA(symbol, dsa);

    // When handling each implicit rule for a given symbol, one of the
    // following actions may be taken:
    // 1. Declare a new private or shared symbol.
    // 2. Use the last declared symbol, by inserting a new symbol in the
    //    scope being processed, associated with it.
    //    If no symbol was declared previously, then no association is needed
    //    and the symbol from the enclosing scope will be inherited by the
    //    current one.
    //
    // Because of how symbols are collected in lowering, not inserting a new
    // symbol in the second case could lead to the conclusion that a symbol
    // from an enclosing construct was declared in the current construct,
    // which would result in wrong privatization code being generated.
    // Consider the following example:
    //
    // !$omp parallel default(private)              ! p1
    //   !$omp parallel default(private) shared(x)  ! p2
    //     x = 10
    //   !$omp end parallel
    // !$omp end parallel
    //
    // If a new x symbol was not inserted in the inner parallel construct
    // (p2), it would use the x symbol definition from the enclosing scope.
    // Then, when p2's default symbols were collected in lowering, the x
    // symbol from the outer parallel construct (p1) would be collected, as
    // it would have the private flag set.
    // This would make x appear to be defined in p2, causing it to be
    // privatized in p2 and its privatization in p1 to be skipped.
    auto makeSymbol = [&](Symbol::Flags flags) {
      const Symbol *hostSymbol =
          lastDeclSymbol ? lastDeclSymbol : &symbol->GetUltimate();
      assert(flags.LeastElement());
      Symbol::Flag flag = *flags.LeastElement();
      lastDeclSymbol = DeclareNewAccessEntity(
          *hostSymbol, flag, context_.FindScope(dirContext.directiveSource));
      lastDeclSymbol->flags() |= flags;
      return lastDeclSymbol;
    };
    auto useLastDeclSymbol = [&]() {
      if (lastDeclSymbol) {
        const Symbol *hostSymbol =
            lastDeclSymbol ? lastDeclSymbol : &symbol->GetUltimate();
        MakeAssocSymbol(symbol->name(), *hostSymbol,
            context_.FindScope(dirContext.directiveSource));
      }
    };

#ifndef NDEBUG
    auto printImplicitRule = [&](const char *id) {
      LLVM_DEBUG(llvm::dbgs() << "\t" << id << ": dsa: " << dsa << '\n');
      LLVM_DEBUG(
          llvm::dbgs() << "\t\tScope: " << dbg::ScopeSourcePos(scope) << '\n');
    };
#define PRINT_IMPLICIT_RULE(id) printImplicitRule(id)
#else
#define PRINT_IMPLICIT_RULE(id)
#endif

    bool taskGenDir = llvm::omp::taskGeneratingSet.test(dirContext.directive);
    bool targetDir = llvm::omp::allTargetSet.test(dirContext.directive);
    bool parallelDir = llvm::omp::topParallelSet.test(dirContext.directive);
    bool teamsDir = llvm::omp::allTeamsSet.test(dirContext.directive);
    bool isStaticStorageDuration = HasStaticStorageDuration(*symbol);
    LLVM_DEBUG(llvm::dbgs()
        << "HasStaticStorageDuration(" << symbol->name() << "):\n");

    const Symbol *crayPtr = nullptr;
    Symbol::Flags crayPtrDSA;
    if (symbol->GetUltimate().test(Symbol::Flag::CrayPointee)) {
      crayPtr =
          currScope().FindSymbol(semantics::GetCrayPointer(*symbol).name());
      if (crayPtr) {
        initSymbolDSA(crayPtr, crayPtrDSA);
      }
    }
    if (dsa.none() && crayPtrDSA.none() &&
        dirContext.defaultDSA == Symbol::Flag::OmpNone) {
      checkDefaultNone = true;
    }
    if (checkDefaultNone) {
      auto defaultNoneError = [&](parser::CharBlock loc, const Symbol *sym) {
        if (crayPtr) {
          context_.Say(loc,
              "The DEFAULT(NONE) clause requires that the Cray Pointer '%s' must be listed in a data-sharing attribute clause"_err_en_US,
              crayPtr->name());
        } else {
          context_.Say(loc,
              "The DEFAULT(NONE) clause requires that '%s' must be listed in a data-sharing attribute clause"_err_en_US,
              sym->name());
        }
      };
      if (dsa.test(Symbol::Flag::OmpPrivate) ||
          crayPtrDSA.test(Symbol::Flag::OmpPrivate)) {
        checkDefaultNone = false;
      } else if (dsa.any() || crayPtrDSA.any()) {
        defaultNoneError(dirContext.directiveSource, symbol);
      } else if (dirDepth == (int)dirContext_.size() - 1) {
        defaultNoneError(name.source, symbol);
      }
    }

    if (dsa.any()) {
      if (parallelDir || taskGenDir || teamsDir) {
        Symbol *prevDeclSymbol{lastDeclSymbol};
        // NOTE As `dsa` will match that of the symbol in the current scope
        //      (if any), we won't override the DSA of any existing symbol.
        if ((dsa & dataSharingAttributeFlags).any()) {
          makeSymbol(dsa);
        }
        // Fix host association of explicit symbols, as they can be created
        // before implicit ones in enclosing scope.
        if (prevDeclSymbol && prevDeclSymbol != lastDeclSymbol &&
            lastDeclSymbol->test(Symbol::Flag::OmpExplicit)) {
          const auto *hostAssoc{lastDeclSymbol->detailsIf<HostAssocDetails>()};
          if (hostAssoc && hostAssoc->symbol() != *prevDeclSymbol) {
            lastDeclSymbol->set_details(HostAssocDetails{*prevDeclSymbol});
          }
        }
      }
      prevDSA = dsa;
      PRINT_IMPLICIT_RULE("0) already has DSA");
      continue;
    }

    // NOTE Because of how lowering uses OmpImplicit flag, we can only set it
    //      for symbols with private DSA.
    //      Also, as the default clause is handled separately in lowering,
    //      don't mark its symbols with OmpImplicit either.
    //      Ideally, lowering should be changed and all implicit symbols
    //      should be marked with OmpImplicit.

    if (dirContext.defaultDSA == Symbol::Flag::OmpPrivate ||
        dirContext.defaultDSA == Symbol::Flag::OmpFirstPrivate ||
        dirContext.defaultDSA == Symbol::Flag::OmpShared) {
      // 1) default
      // Allowed only with parallel, teams and task generating constructs.
      if (!parallelDir && !taskGenDir && !teamsDir) {
        return;
      }
      dsa = {dirContext.defaultDSA};
      makeSymbol(dsa);
      PRINT_IMPLICIT_RULE("1) default");
    } else if (parallelDir) {
      // 2) parallel -> shared
      dsa = {Symbol::Flag::OmpShared};
      makeSymbol(dsa);
      PRINT_IMPLICIT_RULE("2) parallel");
    } else if (!taskGenDir && !targetDir) {
      // 3) enclosing context
      dsa = prevDSA;
      useLastDeclSymbol();
      PRINT_IMPLICIT_RULE("3) enclosing context");
    } else if (targetDir) {
      // 4) not mapped target variable  -> firstprivate
      //    - i.e. implicit, but meets OpenMP specification rules for
      //    firstprivate "promotion"
      if (enableDelayedPrivatizationStaging &&
          IsTargetCaptureImplicitlyFirstprivatizeable(*symbol, prevDSA,
              dataSharingAttributeFlags, dataMappingAttributeFlags,
              dirContext.defaultMap)) {
        prevDSA.set(Symbol::Flag::OmpImplicit);
        prevDSA.set(Symbol::Flag::OmpFirstPrivate);
        makeSymbol(prevDSA);
      }
      dsa = prevDSA;
      PRINT_IMPLICIT_RULE("4) not mapped target variable  -> firstprivate");
    } else if (taskGenDir) {
      // TODO 5) dummy arg in orphaned taskgen construct -> firstprivate
      if (prevDSA.test(Symbol::Flag::OmpShared) ||
          (isStaticStorageDuration &&
              (prevDSA & dataSharingAttributeFlags).none())) {
        // 6) shared in enclosing context -> shared
        dsa = {Symbol::Flag::OmpShared};
        makeSymbol(dsa);
        PRINT_IMPLICIT_RULE("6) taskgen: shared");
      } else {
        // 7) firstprivate
        dsa = {Symbol::Flag::OmpFirstPrivate};
        makeSymbol(dsa)->set(Symbol::Flag::OmpImplicit);
        PRINT_IMPLICIT_RULE("7) taskgen: firstprivate");
      }
    }
    prevDSA = dsa;
  }
}

static bool IsOpenMPPointer(const Symbol &symbol) {
  if (IsPointer(symbol) || IsBuiltinCPtr(symbol))
    return true;
  return false;
}

static bool IsOpenMPAggregate(const Symbol &symbol) {
  if (IsAllocatable(symbol) || IsOpenMPPointer(symbol))
    return false;

  const auto *type{symbol.GetType()};
  // OpenMP categorizes Fortran characters as aggregates.
  if (type->category() == Fortran::semantics::DeclTypeSpec::Category::Character)
    return true;

  if (const auto *det{symbol.GetUltimate()
              .detailsIf<Fortran::semantics::ObjectEntityDetails>()})
    if (det->IsArray())
      return true;

  if (type->AsDerived())
    return true;

  if (IsDeferredShape(symbol) || IsAssumedRank(symbol) ||
      IsAssumedShape(symbol))
    return true;
  return false;
}

static bool IsOpenMPScalar(const Symbol &symbol) {
  if (IsOpenMPAggregate(symbol) || IsOpenMPPointer(symbol) ||
      IsAllocatable(symbol))
    return false;
  const auto *type{symbol.GetType()};
  if ((!symbol.GetShape() || symbol.GetShape()->empty()) &&
      (type->category() ==
              Fortran::semantics::DeclTypeSpec::Category::Numeric ||
          type->category() ==
              Fortran::semantics::DeclTypeSpec::Category::Logical))
    return true;
  return false;
}

static bool DefaultMapCategoryMatchesSymbol(
    parser::OmpVariableCategory::Value category, const Symbol &symbol) {
  using VarCat = parser::OmpVariableCategory::Value;
  switch (category) {
  case VarCat::Scalar:
    return IsOpenMPScalar(symbol);
  case VarCat::Allocatable:
    return IsAllocatable(symbol);
  case VarCat::Aggregate:
    return IsOpenMPAggregate(symbol);
  case VarCat::Pointer:
    return IsOpenMPPointer(symbol);
  case VarCat::All:
    return true;
  }
  return false;
}

// For OpenMP constructs, check all the data-refs within the constructs
// and adjust the symbol for each Name if necessary
void OmpAttributeVisitor::Post(const parser::Name &name) {
  auto *symbol{name.symbol};

  if (symbol && WithinConstruct()) {
    if (omp::IsPrivatizable(*symbol) && !IsObjectWithDSA(*symbol) &&
        !IsLocalInsideScope(*symbol, currScope())) {
      // TODO: create a separate function to go through the rules for
      //       predetermined, explicitly determined, and implicitly
      //       determined data-sharing attributes (2.15.1.1).
      if (Symbol * found{currScope().FindSymbol(name.source)}) {
        if (symbol != found) {
          name.symbol = found; // adjust the symbol within region
        }
      }
    }

    // TODO: handle case where default and defaultmap are present on the same
    // construct and conflict, defaultmap should supersede default if they
    // conflict.
    if (!GetContext().defaultMap.empty()) {
      // Checked before implicit data sharing attributes as this rule ignores
      // them and expects explicit predetermined/specified attributes to be in
      // place for the types specified.
      if (Symbol * found{currScope().FindSymbol(name.source)}) {
        // If the variable has declare target applied to it (enter or link) it
        // is exempt from defaultmap(none) restrictions.
        // We also exempt procedures and named constants from defaultmap(none)
        // checking.
        if (!symbol->GetUltimate().test(Symbol::Flag::OmpDeclareTarget) &&
            !(IsProcedure(*symbol) &&
                !semantics::IsProcedurePointer(*symbol)) &&
            !IsNamedConstant(*symbol)) {
          auto &dMap = GetContext().defaultMap;
          for (auto defaults : dMap) {
            if (defaults.second ==
                parser::OmpDefaultmapClause::ImplicitBehavior::None) {
              if (DefaultMapCategoryMatchesSymbol(defaults.first, *found)) {
                if (!IsObjectWithDSA(*symbol)) {
                  context_.Say(name.source,
                      "The DEFAULTMAP(NONE) clause requires that '%s' must be "
                      "listed in a "
                      "data-sharing attribute, data-mapping attribute, or is_device_ptr clause"_err_en_US,
                      symbol->name());
                }
              }
            }
          }
        }
      }
    }

    if (Symbol * found{currScope().FindSymbol(name.source)}) {
      if (found->GetUltimate().test(semantics::Symbol::Flag::OmpThreadprivate))
        return;
    }

    // We should only create any additional symbols, if the one mentioned
    // in the source code was declared outside of the construct. This was
    // always the case before Fortran 2008. F2008 introduced the BLOCK
    // construct, and allowed local variable declarations.
    // In OpenMP local (non-static) variables are always private in a given
    // construct, if they are declared inside the construct. In those cases
    // we don't need to do anything here (i.e. no flags are needed or
    // anything else).
    if (!IsLocalInsideScope(*symbol, currScope())) {
      CreateImplicitSymbols(name, symbol);
    }
  } // within OpenMP construct
}

Symbol *OmpAttributeVisitor::ResolveName(const parser::Name *name) {
  // TODO: why is the symbol not properly resolved by name resolution?
  if (auto *resolvedSymbol{
          name ? GetContext().scope.FindSymbol(name->source) : nullptr}) {
    name->symbol = resolvedSymbol;
    return resolvedSymbol;
  } else {
    return nullptr;
  }
}

void OmpAttributeVisitor::ResolveOmpName(
    const parser::Name &name, Symbol::Flag ompFlag) {
  if (ResolveName(&name)) {
    if (auto *resolvedSymbol{ResolveOmp(name, ompFlag, currScope())}) {
      if (dataSharingAttributeFlags.test(ompFlag)) {
        AddToContextObjectWithExplicitDSA(*resolvedSymbol, ompFlag);
      }
    }
  } else if (ompFlag == Symbol::Flag::OmpCriticalLock) {
    const auto pair{
        GetContext().scope.try_emplace(name.source, Attrs{}, UnknownDetails{})};
    CHECK(pair.second);
    name.symbol = &pair.first->second.get();
  }
}

void OmpAttributeVisitor::ResolveOmpNameList(
    const std::list<parser::Name> &nameList, Symbol::Flag ompFlag) {
  for (const auto &name : nameList) {
    ResolveOmpName(name, ompFlag);
  }
}

Symbol *OmpAttributeVisitor::ResolveOmpCommonBlockName(
    const parser::Name *name) {
  if (!name) {
    return nullptr;
  }
  if (auto *cb{GetProgramUnitOrBlockConstructContaining(GetContext().scope)
              .FindCommonBlock(name->source)}) {
    name->symbol = cb;
    return cb;
  }
  return nullptr;
}

void OmpAttributeVisitor::ResolveOmpObjectList(
    const parser::OmpObjectList &ompObjectList, Symbol::Flag ompFlag) {
  for (const auto &ompObject : ompObjectList.v) {
    ResolveOmpObject(ompObject, ompFlag);
  }
}

/// True if either symbol is in a namelist or some other symbol in the same
/// equivalence set as symbol is in a namelist.
static bool SymbolOrEquivalentIsInNamelist(const Symbol &symbol) {
  auto isInNamelist{[](const Symbol &sym) {
    const Symbol &ultimate{sym.GetUltimate()};
    return ultimate.test(Symbol::Flag::InNamelist);
  }};

  const EquivalenceSet *eqv{FindEquivalenceSet(symbol)};
  if (!eqv) {
    return isInNamelist(symbol);
  }

  return llvm::any_of(*eqv, [isInNamelist](const EquivalenceObject &obj) {
    return isInNamelist(obj.symbol);
  });
}

void OmpAttributeVisitor::ResolveOmpDesignator(
    const parser::Designator &designator, Symbol::Flag ompFlag) {
  unsigned version{context_.langOptions().OpenMPVersion};
  llvm::omp::Directive directive{GetContext().directive};

  const auto *name{parser::GetDesignatorNameIfDataRef(designator)};
  if (!name) {
    // Array sections to be changed to substrings as needed
    if (AnalyzeExpr(context_, designator)) {
      if (std::holds_alternative<parser::Substring>(designator.u)) {
        context_.Say(designator.source,
            "Substrings are not allowed on OpenMP directives or clauses"_err_en_US);
      }
    }
    // other checks, more TBD
    return;
  }

  if (auto *symbol{ResolveOmp(*name, ompFlag, currScope())}) {
    auto checkExclusivelists{//
        [&](const Symbol *symbol1, Symbol::Flag firstOmpFlag,
            const Symbol *symbol2, Symbol::Flag secondOmpFlag) {
          if ((symbol1->test(firstOmpFlag) && symbol2->test(secondOmpFlag)) ||
              (symbol1->test(secondOmpFlag) && symbol2->test(firstOmpFlag))) {
            context_.Say(designator.source,
                "Variable '%s' may not appear on both %s and %s clauses on a %s construct"_err_en_US,
                symbol2->name(), Symbol::OmpFlagToClauseName(firstOmpFlag),
                Symbol::OmpFlagToClauseName(secondOmpFlag),
                parser::omp::GetUpperName(directive, version));
          }
        }};
    if (dataCopyingAttributeFlags.test(ompFlag)) {
      CheckDataCopyingClause(*name, *symbol, ompFlag);
    } else {
      AddToContextObjectWithExplicitDSA(*symbol, ompFlag);
      if (dataSharingAttributeFlags.test(ompFlag)) {
        CheckMultipleAppearances(*name, *symbol, ompFlag);
      }
      if (privateDataSharingAttributeFlags.test(ompFlag)) {
        CheckObjectIsPrivatizable(*name, *symbol, ompFlag);
      }
    }
    // Save the original symbol. For privatizing clauses, ensure enclosing
    // constructs properly capture the variable.
    const Symbol *origSymbol{name->symbol};
    if (origSymbol && privateDataSharingAttributeFlags.test(ompFlag)) {
      CreateImplicitSymbols(*name, origSymbol);
    }
    if (ompFlag == Symbol::Flag::OmpReduction) {
      // Using variables inside of a namelist in OpenMP reductions
      // is allowed by the standard, but is not allowed for
      // privatisation. This looks like an oversight. If the
      // namelist is hoisted to a global, we cannot apply the
      // mapping for the reduction variable: resulting in incorrect
      // results. Disabling this hoisting could make some real
      // production code go slower. See discussion in #109303
      if (SymbolOrEquivalentIsInNamelist(*symbol)) {
        context_.Say(name->source,
            "Variable '%s' in NAMELIST cannot be in a REDUCTION clause"_err_en_US,
            name->ToString());
      }
    }
    if (ompFlag == Symbol::Flag::OmpInclusiveScan ||
        ompFlag == Symbol::Flag::OmpExclusiveScan) {
      if (!symbol->test(Symbol::Flag::OmpInScanReduction)) {
        context_.Say(name->source,
            "List item %s must appear in REDUCTION clause with the INSCAN modifier of the parent directive"_err_en_US,
            name->ToString());
      }
    }
    if (ompFlag == Symbol::Flag::OmpDeclareTarget) {
      if (symbol->IsFuncResult()) {
        if (Symbol * func{currScope().symbol()}) {
          CHECK(func->IsSubprogram());
          func->set(ompFlag);
          name->symbol = func;
        }
      }
    }
    if (directive == llvm::omp::Directive::OMPD_target_data) {
      checkExclusivelists(symbol, Symbol::Flag::OmpUseDevicePtr, symbol,
          Symbol::Flag::OmpUseDeviceAddr);
    }
    if (llvm::omp::allDistributeSet.test(directive)) {
      checkExclusivelists(symbol, Symbol::Flag::OmpFirstPrivate, symbol,
          Symbol::Flag::OmpLastPrivate);
    }
    if (llvm::omp::allTargetSet.test(directive)) {
      checkExclusivelists(symbol, Symbol::Flag::OmpIsDevicePtr, symbol,
          Symbol::Flag::OmpHasDeviceAddr);
      const auto *hostAssocSym{symbol};
      if (!symbol->test(Symbol::Flag::OmpIsDevicePtr) &&
          !symbol->test(Symbol::Flag::OmpHasDeviceAddr)) {
        if (const auto *details{symbol->detailsIf<HostAssocDetails>()}) {
          hostAssocSym = &details->symbol();
        }
      }
      static Symbol::Flag dataMappingAttributeFlags[] = {//
          Symbol::Flag::OmpMapTo, Symbol::Flag::OmpMapFrom,
          Symbol::Flag::OmpMapToFrom, Symbol::Flag::OmpMapStorage,
          Symbol::Flag::OmpMapDelete, Symbol::Flag::OmpIsDevicePtr,
          Symbol::Flag::OmpHasDeviceAddr};

      static Symbol::Flag dataSharingAttributeFlags[] = {//
          Symbol::Flag::OmpPrivate, Symbol::Flag::OmpFirstPrivate,
          Symbol::Flag::OmpLastPrivate, Symbol::Flag::OmpShared,
          Symbol::Flag::OmpLinear};

      // For OMP TARGET TEAMS directive some sharing attribute
      // flags and mapping attribute flags can co-exist.
      if (!llvm::omp::allTeamsSet.test(directive) &&
          !llvm::omp::allParallelSet.test(directive)) {
        for (Symbol::Flag ompFlag1 : dataMappingAttributeFlags) {
          for (Symbol::Flag ompFlag2 : dataSharingAttributeFlags) {
            if ((hostAssocSym->test(ompFlag2) &&
                    hostAssocSym->test(Symbol::Flag::OmpExplicit)) ||
                (symbol->test(ompFlag2) &&
                    symbol->test(Symbol::Flag::OmpExplicit))) {
              checkExclusivelists(hostAssocSym, ompFlag1, symbol, ompFlag2);
            }
          }
        }
      }
    }
  }
}

void OmpAttributeVisitor::PropagateOmpFlagToEquivalenceSet(
    const Symbol &symbol, Symbol::Flag ompFlag) {
  // Find the equivalence set containing this symbol
  if (const EquivalenceSet *eqSet{FindEquivalenceSet(symbol)}) {
    // Propagate the flag to all symbols in the equivalence set
    for (const EquivalenceObject &eqObj : *eqSet) {
      Symbol &eqSymbol{eqObj.symbol};

      // Skip the symbol itself (already has the flag)
      if (&eqSymbol == &symbol) {
        continue;
      }

      // Set the OpenMP flag on the equivalenced symbol
      if (Symbol * resolvedSymbol{ResolveOmp(eqSymbol, ompFlag, currScope())}) {
        // Also add to the context if needed
        if (ompFlagsRequireMark.test(ompFlag)) {
          AddToContextObjectWithExplicitDSA(*resolvedSymbol, ompFlag);
        }
      }
    }
  }
}

void OmpAttributeVisitor::ResolveOmpCommonBlock(
    const parser::Name &name, Symbol::Flag ompFlag) {
  if (auto *symbol{ResolveOmpCommonBlockName(&name)}) {
    if (!dataCopyingAttributeFlags.test(ompFlag)) {
      CheckMultipleAppearances(name, *symbol, Symbol::Flag::OmpCommonBlock);
    }
    // 2.15.3 When a named common block appears in a list, it has the
    // same meaning as if every explicit member of the common block
    // appeared in the list
    auto &details{symbol->get<CommonBlockDetails>()};
    for (auto [index, object] : llvm::enumerate(details.objects())) {
      if (auto *resolvedObject{ResolveOmp(*object, ompFlag, currScope())}) {
        if (dataCopyingAttributeFlags.test(ompFlag)) {
          CheckDataCopyingClause(name, *resolvedObject, ompFlag);
        } else {
          AddToContextObjectWithExplicitDSA(*resolvedObject, ompFlag);
        }
        details.replace_object(*resolvedObject, index);

        // Propagate the flag to symbols in the equivalence set
        if (ompFlag == Symbol::Flag::OmpThreadprivate) {
          PropagateOmpFlagToEquivalenceSet(*resolvedObject, ompFlag);
        }
      }
    }
  } else {
    context_.Say(name.source, // 2.15.3
        "COMMON block must be declared in the same scoping unit in which the OpenMP directive or clause appears"_err_en_US);
  }
}

void OmpAttributeVisitor::ResolveOmpObject(
    const parser::OmpObject &ompObject, Symbol::Flag ompFlag) {
  common::visit( //
      common::visitors{
          [&](const parser::Designator &designator) {
            ResolveOmpDesignator(designator, ompFlag);
          },
          [&](const parser::Name &name) { // common block
            ResolveOmpCommonBlock(name, ompFlag);
          },
          [&](const parser::OmpObject::Invalid &invalid) {
            switch (invalid.v) {
              SWITCH_COVERS_ALL_CASES
            case parser::OmpObject::Invalid::Kind::BlankCommonBlock:
              context_.Say(invalid.source,
                  "Blank common blocks are not allowed as directive or clause arguments"_err_en_US);
              break;
            }
          },
      },
      ompObject.u);
}

Symbol *OmpAttributeVisitor::ResolveOmp(
    const parser::Name &name, Symbol::Flag ompFlag, Scope &scope) {
  if (ompFlagsRequireNewSymbol.test(ompFlag)) {
    return DeclareAccessEntity(name, ompFlag, scope);
  } else {
    return DeclareOrMarkOtherAccessEntity(name, ompFlag);
  }
}

Symbol *OmpAttributeVisitor::ResolveOmp(
    Symbol &symbol, Symbol::Flag ompFlag, Scope &scope) {
  if (ompFlagsRequireNewSymbol.test(ompFlag)) {
    return DeclareAccessEntity(symbol, ompFlag, scope);
  } else {
    return DeclareOrMarkOtherAccessEntity(symbol, ompFlag);
  }
}

Symbol *OmpAttributeVisitor::DeclareOrMarkOtherAccessEntity(
    const parser::Name &name, Symbol::Flag ompFlag) {
  Symbol *prev{currScope().FindSymbol(name.source)};
  if (!name.symbol || !prev) {
    return nullptr;
  } else if (prev != name.symbol) {
    name.symbol = prev;
  }
  return DeclareOrMarkOtherAccessEntity(*prev, ompFlag);
}

Symbol *OmpAttributeVisitor::DeclareOrMarkOtherAccessEntity(
    Symbol &object, Symbol::Flag ompFlag) {
  if (ompFlagsRequireMark.test(ompFlag)) {
    object.set(ompFlag);
  }
  return &object;
}

static bool WithMultipleAppearancesOmpException(
    const Symbol &symbol, Symbol::Flag flag) {
  return (flag == Symbol::Flag::OmpFirstPrivate &&
             symbol.test(Symbol::Flag::OmpLastPrivate)) ||
      (flag == Symbol::Flag::OmpLastPrivate &&
          symbol.test(Symbol::Flag::OmpFirstPrivate));
}

void OmpAttributeVisitor::CheckMultipleAppearances(
    const parser::Name &name, const Symbol &symbol, Symbol::Flag ompFlag) {
  const auto *target{&symbol};
  if (ompFlagsRequireNewSymbol.test(ompFlag)) {
    if (const auto *details{symbol.detailsIf<HostAssocDetails>()}) {
      target = &details->symbol();
    }
  }
  if (HasDataSharingAttributeObject(target->GetUltimate()) &&
      !WithMultipleAppearancesOmpException(symbol, ompFlag)) {
    context_.Say(name.source,
        "'%s' appears in more than one data-sharing clause "
        "on the same OpenMP directive"_err_en_US,
        name.ToString());
  } else {
    AddDataSharingAttributeObject(target->GetUltimate());
  }
}

static bool IsSymbolThreadprivate(const Symbol &symbol) {
  const Symbol &ultimate{symbol.GetUltimate()};

  return ultimate.test(Symbol::Flag::OmpThreadprivate);
}

static bool IsSymbolPrivate(const Symbol &symbol) {
  LLVM_DEBUG(llvm::dbgs() << "IsSymbolPrivate(" << symbol.name() << "):\n");
  LLVM_DEBUG(dbg::DumpAssocSymbols(llvm::dbgs(), symbol));

  if (Symbol::Flags dsa{GetSymbolDSA(symbol)}; dsa.any()) {
    if (dsa.test(Symbol::Flag::OmpShared)) {
      return false;
    }
    return true;
  }

  // A symbol that has not gone through constructs that may privatize the
  // original symbol may be predetermined as private.
  // (OMP 5.2 5.1.1 - Variables Referenced in a Construct)
  if (symbol == symbol.GetUltimate()) {
    switch (symbol.owner().kind()) {
    case Scope::Kind::MainProgram:
    case Scope::Kind::Subprogram:
    case Scope::Kind::BlockConstruct:
      return !symbol.attrs().test(Attr::SAVE) &&
          !symbol.attrs().test(Attr::PARAMETER) && !IsAssumedShape(symbol) &&
          !symbol.flags().test(Symbol::Flag::InCommonBlock);
    default:
      return false;
    }
  }
  return false;
}

void OmpAttributeVisitor::CheckDataCopyingClause(
    const parser::Name &name, const Symbol &symbol, Symbol::Flag ompFlag) {
  if (ompFlag == Symbol::Flag::OmpCopyIn) {
    // List of items/objects that can appear in a 'copyin' clause must be
    // 'threadprivate'
    if (!IsSymbolThreadprivate(symbol)) {
      context_.Say(name.source,
          "Non-THREADPRIVATE object '%s' in COPYIN clause"_err_en_US,
          symbol.name());
    }
  } else if (ompFlag == Symbol::Flag::OmpCopyPrivate &&
      GetContext().directive == llvm::omp::Directive::OMPD_single) {
    // A list item that appears in a 'copyprivate' clause may not appear on a
    // 'private' or 'firstprivate' clause on a single construct
    if (IsObjectWithDSA(symbol) &&
        (symbol.test(Symbol::Flag::OmpPrivate) ||
            symbol.test(Symbol::Flag::OmpFirstPrivate))) {
      context_.Say(name.source,
          "COPYPRIVATE variable '%s' may not appear on a PRIVATE or "
          "FIRSTPRIVATE clause on a SINGLE construct"_err_en_US,
          symbol.name());
    } else if (!IsSymbolThreadprivate(symbol) && !IsSymbolPrivate(symbol)) {
      // List of items/objects that can appear in a 'copyprivate' clause must be
      // either 'private' or 'threadprivate' in enclosing context.
      context_.Say(name.source,
          "COPYPRIVATE variable '%s' is not PRIVATE or THREADPRIVATE in "
          "outer context"_err_en_US,
          symbol.name());
    }
  }
}

void OmpAttributeVisitor::CheckObjectIsPrivatizable(
    const parser::Name &name, const Symbol &symbol, Symbol::Flag ompFlag) {
  const auto &ultimateSymbol{symbol.GetUltimate()};
  llvm::StringRef clauseName{"PRIVATE"};
  if (ompFlag == Symbol::Flag::OmpFirstPrivate) {
    clauseName = "FIRSTPRIVATE";
  } else if (ompFlag == Symbol::Flag::OmpLastPrivate) {
    clauseName = "LASTPRIVATE";
  }

  if (SymbolOrEquivalentIsInNamelist(symbol)) {
    context_.Say(name.source,
        "Variable '%s' in NAMELIST cannot be in a %s clause"_err_en_US,
        name.ToString(), clauseName.str());
  }

  if (ultimateSymbol.has<AssocEntityDetails>()) {
    context_.Say(name.source,
        "Variable '%s' in ASSOCIATE cannot be in a %s clause"_err_en_US,
        name.ToString(), clauseName.str());
  }

  if (stmtFunctionExprSymbols_.find(ultimateSymbol) !=
      stmtFunctionExprSymbols_.end()) {
    context_.Say(name.source,
        "Variable '%s' in statement function expression cannot be in a "
        "%s clause"_err_en_US,
        name.ToString(), clauseName.str());
  }
}

void OmpAttributeVisitor::AddOmpRequiresToScope(Scope &scope,
    const WithOmpDeclarative::RequiresClauses *reqs,
    const common::OmpMemoryOrderType *memOrder) {
  const Scope &programUnit{omp::GetProgramUnit(scope)};
  using RequiresClauses = WithOmpDeclarative::RequiresClauses;
  RequiresClauses combinedReqs{reqs ? *reqs : RequiresClauses{}};

  if (auto *symbol{const_cast<Symbol *>(programUnit.symbol())}) {
    common::visit(
        [&](auto &details) {
          if constexpr (std::is_convertible_v<decltype(&details),
                            WithOmpDeclarative *>) {
            if (combinedReqs.any()) {
              if (const RequiresClauses *otherReqs{details.ompRequires()}) {
                combinedReqs |= *otherReqs;
              }
              details.set_ompRequires(combinedReqs);
            }
            if (memOrder) {
              if (details.has_ompAtomicDefaultMemOrder() &&
                  *details.ompAtomicDefaultMemOrder() != *memOrder) {
                unsigned version{context_.langOptions().OpenMPVersion};
                context_.Say(programUnit.sourceRange(),
                    "Conflicting '%s' REQUIRES clauses found in compilation "
                    "unit"_err_en_US,
                    parser::omp::GetUpperName(
                        llvm::omp::Clause::OMPC_atomic_default_mem_order,
                        version));
              }
              details.set_ompAtomicDefaultMemOrder(*memOrder);
            }
          }
        },
        symbol->details());
  }
}

#ifndef NDEBUG

static llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const Symbol::Flags &flags) {
  flags.Dump(os, Symbol::EnumToString);
  return os;
}

namespace dbg {

static llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, std::optional<parser::SourcePosition> srcPos) {
  if (srcPos) {
    os << *srcPos.value().path << ":" << srcPos.value().line << ": ";
  }
  return os;
}

static std::optional<parser::SourcePosition> GetSourcePosition(
    const Fortran::semantics::Scope &scope,
    const Fortran::parser::CharBlock &src) {
  parser::AllCookedSources &allCookedSources{
      scope.context().allCookedSources()};
  if (std::optional<parser::ProvenanceRange> prange{
          allCookedSources.GetProvenanceRange(src)}) {
    return allCookedSources.allSources().GetSourcePosition(prange->start());
  }
  return std::nullopt;
}

// Returns a string containing the source location of `scope` followed by
// its first source line.
static std::string ScopeSourcePos(const Fortran::semantics::Scope &scope) {
  const parser::CharBlock &sourceRange{scope.sourceRange()};
  std::string src{sourceRange.ToString()};
  size_t nl{src.find('\n')};
  std::string str;
  llvm::raw_string_ostream ss{str};

  ss << GetSourcePosition(scope, sourceRange) << src.substr(0, nl);
  return str;
}

static void DumpAssocSymbols(llvm::raw_ostream &os, const Symbol &sym) {
  os << '\t' << sym << '\n';
  os << "\t\tOwner: " << ScopeSourcePos(sym.owner()) << '\n';
  if (const auto *details{sym.detailsIf<HostAssocDetails>()}) {
    DumpAssocSymbols(os, details->symbol());
  }
}

} // namespace dbg

#endif

} // namespace Fortran::semantics
