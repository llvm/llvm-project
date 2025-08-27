//===-- lib/Semantics/check-omp-structure.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"

#include "check-directive-structure.h"
#include "definable.h"
#include "resolve-names-utils.h"

#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Common/visit.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/message.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/openmp-modifiers.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "flang/Support/Fortran-features.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace Fortran::semantics {

using namespace Fortran::semantics::omp;
using namespace Fortran::parser::omp;

// Use when clause falls under 'struct OmpClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &) { \
    CheckAllowedClause(llvm::omp::Clause::Y); \
  }

#define CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &c) { \
    CheckAllowedClause(llvm::omp::Clause::Y); \
    RequiresConstantPositiveParameter(llvm::omp::Clause::Y, c.v); \
  }

#define CHECK_REQ_SCALAR_INT_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &c) { \
    CheckAllowedClause(llvm::omp::Clause::Y); \
    RequiresPositiveParameter(llvm::omp::Clause::Y, c.v); \
  }

// Use when clause don't falls under 'struct OmpClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_PARSER_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::X &) { \
    CheckAllowedClause(llvm::omp::Y); \
  }

// 'OmpWorkshareBlockChecker' is used to check the validity of the assignment
// statements and the expressions enclosed in an OpenMP Workshare construct
class OmpWorkshareBlockChecker {
public:
  OmpWorkshareBlockChecker(SemanticsContext &context, parser::CharBlock source)
      : context_{context}, source_{source} {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::AssignmentStmt &assignment) {
    const auto &var{std::get<parser::Variable>(assignment.t)};
    const auto &expr{std::get<parser::Expr>(assignment.t)};
    const auto *lhs{GetExpr(context_, var)};
    const auto *rhs{GetExpr(context_, expr)};
    if (lhs && rhs) {
      Tristate isDefined{semantics::IsDefinedAssignment(
          lhs->GetType(), lhs->Rank(), rhs->GetType(), rhs->Rank())};
      if (isDefined == Tristate::Yes) {
        context_.Say(expr.source,
            "Defined assignment statement is not "
            "allowed in a WORKSHARE construct"_err_en_US);
      }
    }
    return true;
  }

  bool Pre(const parser::Expr &expr) {
    if (const auto *e{GetExpr(context_, expr)}) {
      for (const Symbol &symbol : evaluate::CollectSymbols(*e)) {
        const Symbol &root{GetAssociationRoot(symbol)};
        if (IsFunction(root)) {
          std::string attrs{""};
          if (!IsElementalProcedure(root)) {
            attrs = " non-ELEMENTAL";
          }
          if (root.attrs().test(Attr::IMPURE)) {
            if (attrs != "") {
              attrs = "," + attrs;
            }
            attrs = " IMPURE" + attrs;
          }
          if (attrs != "") {
            context_.Say(expr.source,
                "User defined%s function '%s' is not allowed in a "
                "WORKSHARE construct"_err_en_US,
                attrs, root.name());
          }
        }
      }
    }
    return false;
  }

private:
  SemanticsContext &context_;
  parser::CharBlock source_;
};

// 'OmpWorkdistributeBlockChecker' is used to check the validity of the
// assignment statements and the expressions enclosed in an OpenMP
// WORKDISTRIBUTE construct
class OmpWorkdistributeBlockChecker {
public:
  OmpWorkdistributeBlockChecker(
      SemanticsContext &context, parser::CharBlock source)
      : context_{context}, source_{source} {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::AssignmentStmt &assignment) {
    const auto &var{std::get<parser::Variable>(assignment.t)};
    const auto &expr{std::get<parser::Expr>(assignment.t)};
    const auto *lhs{GetExpr(context_, var)};
    const auto *rhs{GetExpr(context_, expr)};
    if (lhs && rhs) {
      Tristate isDefined{semantics::IsDefinedAssignment(
          lhs->GetType(), lhs->Rank(), rhs->GetType(), rhs->Rank())};
      if (isDefined == Tristate::Yes) {
        context_.Say(expr.source,
            "Defined assignment statement is not allowed in a WORKDISTRIBUTE construct"_err_en_US);
      }
    }
    return true;
  }

  bool Pre(const parser::Expr &expr) {
    if (const auto *e{GetExpr(context_, expr)}) {
      if (!e)
        return false;
      for (const Symbol &symbol : evaluate::CollectSymbols(*e)) {
        const Symbol &root{GetAssociationRoot(symbol)};
        if (IsFunction(root)) {
          std::vector<std::string> attrs;
          if (!IsElementalProcedure(root)) {
            attrs.push_back("non-ELEMENTAL");
          }
          if (root.attrs().test(Attr::IMPURE)) {
            attrs.push_back("IMPURE");
          }
          std::string attrsStr =
              attrs.empty() ? "" : " " + llvm::join(attrs, ", ");
          context_.Say(expr.source,
              "User defined%s function '%s' is not allowed in a WORKDISTRIBUTE construct"_err_en_US,
              attrsStr, root.name());
        }
      }
    }
    return false;
  }

private:
  SemanticsContext &context_;
  parser::CharBlock source_;
};

// `OmpUnitedTaskDesignatorChecker` is used to check if the designator
// can appear within the TASK construct
class OmpUnitedTaskDesignatorChecker {
public:
  OmpUnitedTaskDesignatorChecker(SemanticsContext &context)
      : context_{context} {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::Name &name) {
    if (name.symbol->test(Symbol::Flag::OmpThreadprivate)) {
      // OpenMP 5.2: 5.2 threadprivate directive restriction
      context_.Say(name.source,
          "A THREADPRIVATE variable `%s` cannot appear in an UNTIED TASK region"_err_en_US,
          name.source);
    }
    return true;
  }

private:
  SemanticsContext &context_;
};

bool OmpStructureChecker::CheckAllowedClause(llvmOmpClause clause) {
  // Do not do clause checks while processing METADIRECTIVE.
  // Context selectors can contain clauses that are not given as a part
  // of a construct, but as trait properties. Testing whether they are
  // valid or not is deferred to the checks of the context selectors.
  // As it stands now, these clauses would appear as if they were present
  // on METADIRECTIVE, leading to incorrect diagnostics.
  if (GetDirectiveNest(ContextSelectorNest) > 0) {
    return true;
  }

  unsigned version{context_.langOptions().OpenMPVersion};
  DirectiveContext &dirCtx = GetContext();
  llvm::omp::Directive dir{dirCtx.directive};

  if (!llvm::omp::isAllowedClauseForDirective(dir, clause, version)) {
    unsigned allowedInVersion{[&] {
      for (unsigned v : llvm::omp::getOpenMPVersions()) {
        if (v <= version) {
          continue;
        }
        if (llvm::omp::isAllowedClauseForDirective(dir, clause, v)) {
          return v;
        }
      }
      return 0u;
    }()};

    // Only report it if there is a later version that allows it.
    // If it's not allowed at all, it will be reported by CheckAllowed.
    if (allowedInVersion != 0) {
      auto clauseName{parser::ToUpperCaseLetters(getClauseName(clause).str())};
      auto dirName{parser::ToUpperCaseLetters(getDirectiveName(dir).str())};

      context_.Say(dirCtx.clauseSource,
          "%s clause is not allowed on directive %s in %s, %s"_err_en_US,
          clauseName, dirName, ThisVersion(version),
          TryVersion(allowedInVersion));
    }
  }
  return CheckAllowed(clause);
}

void OmpStructureChecker::AnalyzeObject(const parser::OmpObject &object) {
  if (std::holds_alternative<parser::Name>(object.u)) {
    // Do not analyze common block names. The analyzer will flag an error
    // on those.
    return;
  }
  if (auto *symbol{GetObjectSymbol(object)}) {
    // Eliminate certain kinds of symbols before running the analyzer to
    // avoid confusing error messages. The analyzer assumes that the context
    // of the object use is an expression, and some diagnostics are tailored
    // to that.
    if (symbol->has<DerivedTypeDetails>() || symbol->has<MiscDetails>()) {
      // Type names, construct names, etc.
      return;
    }
    if (auto *typeSpec{symbol->GetType()}) {
      if (typeSpec->category() == DeclTypeSpec::Category::Character) {
        // Don't pass character objects to the analyzer, it can emit somewhat
        // cryptic errors (e.g. "'obj' is not an array"). Substrings are
        // checked elsewhere in OmpStructureChecker.
        return;
      }
    }
  }
  evaluate::ExpressionAnalyzer ea{context_};
  auto restore{ea.AllowWholeAssumedSizeArray(true)};
  common::visit([&](auto &&s) { ea.Analyze(s); }, object.u);
}

void OmpStructureChecker::AnalyzeObjects(const parser::OmpObjectList &objects) {
  for (const parser::OmpObject &object : objects.v) {
    AnalyzeObject(object);
  }
}

bool OmpStructureChecker::IsCloselyNestedRegion(const OmpDirectiveSet &set) {
  // Definition of close nesting:
  //
  // `A region nested inside another region with no parallel region nested
  // between them`
  //
  // Examples:
  //   non-parallel construct 1
  //    non-parallel construct 2
  //      parallel construct
  //        construct 3
  // In the above example, construct 3 is NOT closely nested inside construct 1
  // or 2
  //
  //   non-parallel construct 1
  //    non-parallel construct 2
  //        construct 3
  // In the above example, construct 3 is closely nested inside BOTH construct 1
  // and 2
  //
  // Algorithm:
  // Starting from the parent context, Check in a bottom-up fashion, each level
  // of the context stack. If we have a match for one of the (supplied)
  // violating directives, `close nesting` is satisfied. If no match is there in
  // the entire stack, `close nesting` is not satisfied. If at any level, a
  // `parallel` region is found, `close nesting` is not satisfied.

  if (CurrentDirectiveIsNested()) {
    int index = dirContext_.size() - 2;
    while (index != -1) {
      if (set.test(dirContext_[index].directive)) {
        return true;
      } else if (llvm::omp::allParallelSet.test(dirContext_[index].directive)) {
        return false;
      }
      index--;
    }
  }
  return false;
}

void OmpStructureChecker::CheckVariableListItem(
    const SymbolSourceMap &symbols) {
  for (auto &[symbol, source] : symbols) {
    if (!IsVariableListItem(*symbol)) {
      context_.SayWithDecl(
          *symbol, source, "'%s' must be a variable"_err_en_US, symbol->name());
    }
  }
}

void OmpStructureChecker::CheckDirectiveSpelling(
    parser::CharBlock spelling, llvm::omp::Directive id) {
  // Directive names that contain spaces can be spelled in the source without
  // any of the spaces. Because of that getOpenMPKind* is not guaranteed to
  // work with the source spelling as the argument.
  //
  // To verify the source spellings, we have to get the spelling for a given
  // version, remove spaces and compare it with the source spelling (also
  // with spaces removed).
  auto removeSpaces = [](llvm::StringRef s) {
    std::string n{s.str()};
    for (size_t idx{n.size()}; idx > 0; --idx) {
      if (isspace(n[idx - 1])) {
        n.erase(idx - 1, 1);
      }
    }
    return n;
  };

  std::string lowerNoWS{removeSpaces(
      parser::ToLowerCaseLetters({spelling.begin(), spelling.size()}))};
  llvm::StringRef ref(lowerNoWS);
  if (ref.starts_with("end")) {
    ref = ref.drop_front(3);
  }

  unsigned version{context_.langOptions().OpenMPVersion};

  // For every "future" version v, check if the check if the corresponding
  // spelling of id was introduced later than the current version. If so,
  // and if that spelling matches the source spelling, issue a warning.
  for (unsigned v : llvm::omp::getOpenMPVersions()) {
    if (v <= version) {
      continue;
    }
    llvm::StringRef name{llvm::omp::getOpenMPDirectiveName(id, v)};
    auto [kind, versions]{llvm::omp::getOpenMPDirectiveKindAndVersions(name)};
    assert(kind == id && "Directive kind mismatch");

    if (static_cast<int>(version) >= versions.Min) {
      continue;
    }
    if (ref == removeSpaces(name)) {
      context_.Say(spelling,
          "Directive spelling '%s' is introduced in a later OpenMP version, %s"_warn_en_US,
          parser::ToUpperCaseLetters(ref), TryVersion(versions.Min));
      break;
    }
  }
}

void OmpStructureChecker::CheckMultipleOccurrence(
    semantics::UnorderedSymbolSet &listVars,
    const std::list<parser::Name> &nameList, const parser::CharBlock &item,
    const std::string &clauseName) {
  for (auto const &var : nameList) {
    if (llvm::is_contained(listVars, *(var.symbol))) {
      context_.Say(item,
          "List item '%s' present at multiple %s clauses"_err_en_US,
          var.ToString(), clauseName);
    }
    listVars.insert(*(var.symbol));
  }
}

void OmpStructureChecker::CheckMultListItems() {
  semantics::UnorderedSymbolSet listVars;

  // Aligned clause
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_aligned)) {
    const auto &alignedClause{std::get<parser::OmpClause::Aligned>(clause->u)};
    const auto &alignedList{std::get<0>(alignedClause.v.t)};
    std::list<parser::Name> alignedNameList;
    for (const auto &ompObject : alignedList.v) {
      if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
        if (name->symbol) {
          if (FindCommonBlockContaining(*(name->symbol))) {
            context_.Say(clause->source,
                "'%s' is a common block name and can not appear in an "
                "ALIGNED clause"_err_en_US,
                name->ToString());
          } else if (!(IsBuiltinCPtr(*(name->symbol)) ||
                         IsAllocatableOrObjectPointer(
                             &name->symbol->GetUltimate()))) {
            context_.Say(clause->source,
                "'%s' in ALIGNED clause must be of type C_PTR, POINTER or "
                "ALLOCATABLE"_err_en_US,
                name->ToString());
          } else {
            alignedNameList.push_back(*name);
          }
        } else {
          // The symbol is null, return early
          return;
        }
      }
    }
    CheckMultipleOccurrence(
        listVars, alignedNameList, clause->source, "ALIGNED");
  }

  // Nontemporal clause
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_nontemporal)) {
    const auto &nontempClause{
        std::get<parser::OmpClause::Nontemporal>(clause->u)};
    const auto &nontempNameList{nontempClause.v};
    CheckMultipleOccurrence(
        listVars, nontempNameList, clause->source, "NONTEMPORAL");
  }

  // Linear clause
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_linear)) {
    auto &linearClause{std::get<parser::OmpClause::Linear>(clause->u)};
    std::list<parser::Name> nameList;
    SymbolSourceMap symbols;
    GetSymbolsInObjectList(
        std::get<parser::OmpObjectList>(linearClause.v.t), symbols);
    llvm::transform(symbols, std::back_inserter(nameList), [&](auto &&pair) {
      return parser::Name{pair.second, const_cast<Symbol *>(pair.first)};
    });
    CheckMultipleOccurrence(listVars, nameList, clause->source, "LINEAR");
  }
}

bool OmpStructureChecker::HasInvalidWorksharingNesting(
    const parser::CharBlock &source, const OmpDirectiveSet &set) {
  // set contains all the invalid closely nested directives
  // for the given directive (`source` here)
  if (IsCloselyNestedRegion(set)) {
    context_.Say(source,
        "A worksharing region may not be closely nested inside a "
        "worksharing, explicit task, taskloop, critical, ordered, atomic, or "
        "master region"_err_en_US);
    return true;
  }
  return false;
}

void OmpStructureChecker::HasInvalidTeamsNesting(
    const llvm::omp::Directive &dir, const parser::CharBlock &source) {
  if (!llvm::omp::nestedTeamsAllowedSet.test(dir)) {
    context_.Say(source,
        "Only `DISTRIBUTE`, `PARALLEL`, or `LOOP` regions are allowed to be "
        "strictly nested inside `TEAMS` region."_err_en_US);
  }
}

void OmpStructureChecker::CheckPredefinedAllocatorRestriction(
    const parser::CharBlock &source, const parser::Name &name) {
  if (const auto *symbol{name.symbol}) {
    const auto *commonBlock{FindCommonBlockContaining(*symbol)};
    const auto &scope{context_.FindScope(symbol->name())};
    const Scope &containingScope{GetProgramUnitContaining(scope)};
    if (!isPredefinedAllocator &&
        (IsSaved(*symbol) || commonBlock ||
            containingScope.kind() == Scope::Kind::Module)) {
      context_.Say(source,
          "If list items within the %s directive have the "
          "SAVE attribute, are a common block name, or are "
          "declared in the scope of a module, then only "
          "predefined memory allocator parameters can be used "
          "in the allocator clause"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}

void OmpStructureChecker::CheckPredefinedAllocatorRestriction(
    const parser::CharBlock &source,
    const parser::OmpObjectList &ompObjectList) {
  for (const auto &ompObject : ompObjectList.v) {
    common::visit(
        common::visitors{
            [&](const parser::Designator &designator) {
              if (const auto *dataRef{
                      std::get_if<parser::DataRef>(&designator.u)}) {
                if (const auto *name{std::get_if<parser::Name>(&dataRef->u)}) {
                  CheckPredefinedAllocatorRestriction(source, *name);
                }
              }
            },
            [&](const parser::Name &name) {
              CheckPredefinedAllocatorRestriction(source, name);
            },
        },
        ompObject.u);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Hint &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_hint);
  auto &dirCtx{GetContext()};

  if (std::optional<int64_t> maybeVal{GetIntValue(x.v.v)}) {
    int64_t val{*maybeVal};
    if (val >= 0) {
      // Check contradictory values.
      if ((val & 0xC) == 0xC || // omp_sync_hint_speculative and nonspeculative
          (val & 0x3) == 0x3) { // omp_sync_hint_contended and uncontended
        context_.Say(dirCtx.clauseSource,
            "The synchronization hint is not valid"_err_en_US);
      }
    } else {
      context_.Say(dirCtx.clauseSource,
          "Synchronization hint must be non-negative"_err_en_US);
    }
  } else {
    context_.Say(dirCtx.clauseSource,
        "Synchronization hint must be a constant integer value"_err_en_US);
  }
}

void OmpStructureChecker::Enter(const parser::OmpDirectiveSpecification &x) {
  // OmpDirectiveSpecification exists on its own only in METADIRECTIVE.
  // In other cases it's a part of other constructs that handle directive
  // context stack by themselves.
  if (GetDirectiveNest(MetadirectiveNest)) {
    PushContextAndClauseSets(
        std::get<parser::OmpDirectiveName>(x.t).source, x.DirId());
  }
}

void OmpStructureChecker::Leave(const parser::OmpDirectiveSpecification &) {
  if (GetDirectiveNest(MetadirectiveNest)) {
    dirContext_.pop_back();
  }
}

template <typename Checker> struct DirectiveSpellingVisitor {
  using Directive = llvm::omp::Directive;

  DirectiveSpellingVisitor(Checker &&checker) : checker_(std::move(checker)) {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  template <typename... Ts>
  static const parser::OmpDirectiveName &GetDirName(
      const std::tuple<Ts...> &t) {
    return std::get<parser::OmpBeginDirective>(t).DirName();
  }

  bool Pre(const parser::OmpSectionsDirective &x) {
    checker_(x.source, x.v);
    return false;
  }
  bool Pre(const parser::OpenMPDeclarativeAllocate &x) {
    checker_(std::get<parser::Verbatim>(x.t).source, Directive::OMPD_allocate);
    return false;
  }
  bool Pre(const parser::OpenMPDispatchConstruct &x) {
    checker_(GetDirName(x.t).source, Directive::OMPD_dispatch);
    return false;
  }
  bool Pre(const parser::OmpErrorDirective &x) {
    checker_(std::get<parser::Verbatim>(x.t).source, Directive::OMPD_error);
    return false;
  }
  bool Pre(const parser::OmpNothingDirective &x) {
    checker_(x.source, Directive::OMPD_nothing);
    return false;
  }
  bool Pre(const parser::OpenMPExecutableAllocate &x) {
    checker_(std::get<parser::Verbatim>(x.t).source, Directive::OMPD_allocate);
    return false;
  }
  bool Pre(const parser::OpenMPAllocatorsConstruct &x) {
    checker_(GetDirName(x.t).source, Directive::OMPD_allocators);
    return false;
  }
  bool Pre(const parser::OmpMetadirectiveDirective &x) {
    checker_(
        std::get<parser::Verbatim>(x.t).source, Directive::OMPD_metadirective);
    return false;
  }
  bool Pre(const parser::OpenMPDeclarativeAssumes &x) {
    checker_(std::get<parser::Verbatim>(x.t).source, Directive::OMPD_assumes);
    return false;
  }
  bool Pre(const parser::OpenMPDeclareMapperConstruct &x) {
    checker_(
        std::get<parser::Verbatim>(x.t).source, Directive::OMPD_declare_mapper);
    return false;
  }
  bool Pre(const parser::OpenMPDeclareReductionConstruct &x) {
    checker_(std::get<parser::Verbatim>(x.t).source,
        Directive::OMPD_declare_reduction);
    return false;
  }
  bool Pre(const parser::OpenMPDeclareSimdConstruct &x) {
    checker_(
        std::get<parser::Verbatim>(x.t).source, Directive::OMPD_declare_simd);
    return false;
  }
  bool Pre(const parser::OpenMPDeclareTargetConstruct &x) {
    checker_(
        std::get<parser::Verbatim>(x.t).source, Directive::OMPD_declare_target);
    return false;
  }
  bool Pre(const parser::OmpDeclareVariantDirective &x) {
    checker_(std::get<parser::Verbatim>(x.t).source,
        Directive::OMPD_declare_variant);
    return false;
  }
  bool Pre(const parser::OpenMPGroupprivate &x) {
    checker_(x.v.DirName().source, Directive::OMPD_groupprivate);
    return false;
  }
  bool Pre(const parser::OpenMPThreadprivate &x) {
    checker_(
        std::get<parser::Verbatim>(x.t).source, Directive::OMPD_threadprivate);
    return false;
  }
  bool Pre(const parser::OpenMPRequiresConstruct &x) {
    checker_(std::get<parser::Verbatim>(x.t).source, Directive::OMPD_requires);
    return false;
  }
  bool Pre(const parser::OmpBeginDirective &x) {
    checker_(x.DirName().source, x.DirId());
    return false;
  }
  bool Pre(const parser::OmpEndDirective &x) {
    checker_(x.DirName().source, x.DirId());
    return false;
  }
  bool Pre(const parser::OmpLoopDirective &x) {
    checker_(x.source, x.v);
    return false;
  }

  bool Pre(const parser::OmpDirectiveSpecification &x) {
    auto &name = std::get<parser::OmpDirectiveName>(x.t);
    checker_(name.source, name.v);
    return false;
  }

private:
  Checker checker_;
};

template <typename T>
DirectiveSpellingVisitor(T &&) -> DirectiveSpellingVisitor<T>;

void OmpStructureChecker::Enter(const parser::OpenMPConstruct &x) {
  DirectiveSpellingVisitor visitor(
      [this](parser::CharBlock source, llvm::omp::Directive id) {
        return CheckDirectiveSpelling(source, id);
      });
  parser::Walk(x, visitor);

  // Simd Construct with Ordered Construct Nesting check
  // We cannot use CurrentDirectiveIsNested() here because
  // PushContextAndClauseSets() has not been called yet, it is
  // called individually for each construct.  Therefore a
  // dirContext_ size `1` means the current construct is nested
  if (dirContext_.size() >= 1) {
    if (GetDirectiveNest(SIMDNest) > 0) {
      CheckSIMDNest(x);
    }
    if (GetDirectiveNest(TargetNest) > 0) {
      CheckTargetNest(x);
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPConstruct &) {
  for (const auto &[sym, source] : deferredNonVariables_) {
    context_.SayWithDecl(
        *sym, source, "'%s' must be a variable"_err_en_US, sym->name());
  }
  deferredNonVariables_.clear();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclarativeConstruct &x) {
  DirectiveSpellingVisitor visitor(
      [this](parser::CharBlock source, llvm::omp::Directive id) {
        return CheckDirectiveSpelling(source, id);
      });
  parser::Walk(x, visitor);

  EnterDirectiveNest(DeclarativeNest);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclarativeConstruct &x) {
  ExitDirectiveNest(DeclarativeNest);
}

void OmpStructureChecker::AddEndDirectiveClauses(
    const parser::OmpClauseList &clauses) {
  for (const parser::OmpClause &clause : clauses.v) {
    GetContext().endDirectiveClauses.push_back(clause.Id());
  }
}

void OmpStructureChecker::CheckIteratorRange(
    const parser::OmpIteratorSpecifier &x) {
  // Check:
  // 1. Whether begin/end are present.
  // 2. Whether the step value is non-zero.
  // 3. If the step has a known sign, whether the lower/upper bounds form
  // a proper interval.
  const auto &[begin, end, step]{std::get<parser::SubscriptTriplet>(x.t).t};
  if (!begin || !end) {
    context_.Say(x.source,
        "The begin and end expressions in iterator range-specification are "
        "mandatory"_err_en_US);
  }
  // [5.2:67:19] In a range-specification, if the step is not specified its
  // value is implicitly defined to be 1.
  if (auto stepv{step ? GetIntValue(*step) : std::optional<int64_t>{1}}) {
    if (*stepv == 0) {
      context_.Say(
          x.source, "The step value in the iterator range is 0"_warn_en_US);
    } else if (begin && end) {
      std::optional<int64_t> beginv{GetIntValue(*begin)};
      std::optional<int64_t> endv{GetIntValue(*end)};
      if (beginv && endv) {
        if (*stepv > 0 && *beginv > *endv) {
          context_.Say(x.source,
              "The begin value is greater than the end value in iterator "
              "range-specification with a positive step"_warn_en_US);
        } else if (*stepv < 0 && *beginv < *endv) {
          context_.Say(x.source,
              "The begin value is less than the end value in iterator "
              "range-specification with a negative step"_warn_en_US);
        }
      }
    }
  }
}

void OmpStructureChecker::CheckIteratorModifier(const parser::OmpIterator &x) {
  // Check if all iterator variables have integer type.
  for (auto &&iterSpec : x.v) {
    bool isInteger{true};
    auto &typeDecl{std::get<parser::TypeDeclarationStmt>(iterSpec.t)};
    auto &typeSpec{std::get<parser::DeclarationTypeSpec>(typeDecl.t)};
    if (!std::holds_alternative<parser::IntrinsicTypeSpec>(typeSpec.u)) {
      isInteger = false;
    } else {
      auto &intrinType{std::get<parser::IntrinsicTypeSpec>(typeSpec.u)};
      if (!std::holds_alternative<parser::IntegerTypeSpec>(intrinType.u)) {
        isInteger = false;
      }
    }
    if (!isInteger) {
      context_.Say(iterSpec.source,
          "The iterator variable must be of integer type"_err_en_US);
    }
    CheckIteratorRange(iterSpec);
  }
}

void OmpStructureChecker::CheckTargetNest(const parser::OpenMPConstruct &c) {
  // 2.12.5 Target Construct Restriction
  bool eligibleTarget{true};
  llvm::omp::Directive ineligibleTargetDir;
  parser::CharBlock source;
  common::visit(
      common::visitors{
          [&](const parser::OpenMPBlockConstruct &c) {
            const parser::OmpDirectiveSpecification &beginSpec{c.BeginDir()};
            source = beginSpec.DirName().source;
            if (beginSpec.DirId() == llvm::omp::Directive::OMPD_target_data) {
              eligibleTarget = false;
              ineligibleTargetDir = beginSpec.DirId();
            }
          },
          [&](const parser::OpenMPStandaloneConstruct &c) {
            common::visit(
                common::visitors{
                    [&](const parser::OpenMPSimpleStandaloneConstruct &c) {
                      source = c.v.DirName().source;
                      switch (llvm::omp::Directive dirId{c.v.DirId()}) {
                      case llvm::omp::Directive::OMPD_target_update:
                      case llvm::omp::Directive::OMPD_target_enter_data:
                      case llvm::omp::Directive::OMPD_target_exit_data:
                        eligibleTarget = false;
                        ineligibleTargetDir = dirId;
                        break;
                      default:
                        break;
                      }
                    },
                    [&](const auto &c) {},
                },
                c.u);
          },
          [&](const parser::OpenMPLoopConstruct &c) {
            const auto &beginLoopDir{
                std::get<parser::OmpBeginLoopDirective>(c.t)};
            const auto &beginDir{
                std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
            source = beginLoopDir.source;
            if (llvm::omp::allTargetSet.test(beginDir.v)) {
              eligibleTarget = false;
              ineligibleTargetDir = beginDir.v;
            }
          },
          [&](const auto &c) {},
      },
      c.u);
  if (!eligibleTarget) {
    context_.Warn(common::UsageWarning::OpenMPUsage, source,
        "If %s directive is nested inside TARGET region, the behaviour is unspecified"_port_en_US,
        parser::ToUpperCaseLetters(
            getDirectiveName(ineligibleTargetDir).str()));
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};
  const std::optional<parser::OmpEndDirective> &endSpec{x.EndDir()};
  const parser::Block &block{std::get<parser::Block>(x.t)};

  PushContextAndClauseSets(beginSpec.DirName().source, beginSpec.DirId());

  // Missing mandatory end block: this is checked in semantics because that
  // makes it easier to control the error messages.
  // The end block is mandatory when the construct is not applied to a strictly
  // structured block (aka it is applied to a loosely structured block). In
  // other words, the body doesn't contain exactly one parser::BlockConstruct.
  auto isStrictlyStructuredBlock{[](const parser::Block &block) -> bool {
    if (block.size() != 1) {
      return false;
    }
    const parser::ExecutionPartConstruct &contents{block.front()};
    auto *executableConstruct{
        std::get_if<parser::ExecutableConstruct>(&contents.u)};
    if (!executableConstruct) {
      return false;
    }
    return std::holds_alternative<common::Indirection<parser::BlockConstruct>>(
        executableConstruct->u);
  }};
  if (!endSpec && !isStrictlyStructuredBlock(block)) {
    context_.Say(
        x.BeginDir().source, "Expected OpenMP end directive"_err_en_US);
  }

  if (llvm::omp::allTargetSet.test(GetContext().directive)) {
    EnterDirectiveNest(TargetNest);
  }

  if (CurrentDirectiveIsNested()) {
    if (llvm::omp::bottomTeamsSet.test(GetContextParent().directive)) {
      HasInvalidTeamsNesting(beginSpec.DirId(), beginSpec.source);
    }
    if (GetContext().directive == llvm::omp::Directive::OMPD_master) {
      CheckMasterNesting(x);
    }
    // A teams region can only be strictly nested within the implicit parallel
    // region or a target region.
    if (GetContext().directive == llvm::omp::Directive::OMPD_teams &&
        GetContextParent().directive != llvm::omp::Directive::OMPD_target) {
      context_.Say(x.BeginDir().DirName().source,
          "%s region can only be strictly nested within the implicit parallel "
          "region or TARGET region"_err_en_US,
          ContextDirectiveAsFortran());
    }
    // If a teams construct is nested within a target construct, that target
    // construct must contain no statements, declarations or directives outside
    // of the teams construct.
    if (GetContext().directive == llvm::omp::Directive::OMPD_teams &&
        GetContextParent().directive == llvm::omp::Directive::OMPD_target &&
        !GetDirectiveNest(TargetBlockOnlyTeams)) {
      context_.Say(GetContextParent().directiveSource,
          "TARGET construct with nested TEAMS region contains statements or "
          "directives outside of the TEAMS construct"_err_en_US);
    }
    if (GetContext().directive == llvm::omp::Directive::OMPD_workdistribute &&
        GetContextParent().directive != llvm::omp::Directive::OMPD_teams) {
      context_.Say(x.BeginDir().DirName().source,
          "%s region can only be strictly nested within TEAMS region"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }

  CheckNoBranching(block, beginSpec.DirId(), beginSpec.source);

  // Target block constructs are target device constructs. Keep track of
  // whether any such construct has been visited to later check that REQUIRES
  // directives for target-related options don't appear after them.
  if (llvm::omp::allTargetSet.test(beginSpec.DirId())) {
    deviceConstructFound_ = true;
  }

  if (GetContext().directive == llvm::omp::Directive::OMPD_single) {
    std::set<Symbol *> singleCopyprivateSyms;
    std::set<Symbol *> endSingleCopyprivateSyms;
    bool foundNowait{false};
    parser::CharBlock NowaitSource;

    auto catchCopyPrivateNowaitClauses = [&](const auto &dirSpec, bool isEnd) {
      for (auto &clause : dirSpec.Clauses().v) {
        if (clause.Id() == llvm::omp::Clause::OMPC_copyprivate) {
          for (const auto &ompObject : GetOmpObjectList(clause)->v) {
            const auto *name{parser::Unwrap<parser::Name>(ompObject)};
            if (Symbol * symbol{name->symbol}) {
              if (singleCopyprivateSyms.count(symbol)) {
                if (isEnd) {
                  context_.Warn(common::UsageWarning::OpenMPUsage, name->source,
                      "The COPYPRIVATE clause with '%s' is already used on the SINGLE directive"_warn_en_US,
                      name->ToString());
                } else {
                  context_.Say(name->source,
                      "'%s' appears in more than one COPYPRIVATE clause on the SINGLE directive"_err_en_US,
                      name->ToString());
                }
              } else if (endSingleCopyprivateSyms.count(symbol)) {
                context_.Say(name->source,
                    "'%s' appears in more than one COPYPRIVATE clause on the END SINGLE directive"_err_en_US,
                    name->ToString());
              } else {
                if (isEnd) {
                  endSingleCopyprivateSyms.insert(symbol);
                } else {
                  singleCopyprivateSyms.insert(symbol);
                }
              }
            }
          }
        } else if (clause.Id() == llvm::omp::Clause::OMPC_nowait) {
          if (foundNowait) {
            context_.Say(clause.source,
                "At most one NOWAIT clause can appear on the SINGLE directive"_err_en_US);
          } else {
            foundNowait = !isEnd;
          }
          if (!NowaitSource.ToString().size()) {
            NowaitSource = clause.source;
          }
        }
      }
    };
    catchCopyPrivateNowaitClauses(beginSpec, false);
    if (endSpec) {
      catchCopyPrivateNowaitClauses(*endSpec, true);
    }
    unsigned version{context_.langOptions().OpenMPVersion};
    if (version <= 52 && NowaitSource.ToString().size() &&
        (singleCopyprivateSyms.size() || endSingleCopyprivateSyms.size())) {
      context_.Say(NowaitSource,
          "NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive"_err_en_US);
    }
  }

  switch (beginSpec.DirId()) {
  case llvm::omp::Directive::OMPD_target:
    if (CheckTargetBlockOnlyTeams(block)) {
      EnterDirectiveNest(TargetBlockOnlyTeams);
    }
    break;
  case llvm::omp::OMPD_workshare:
  case llvm::omp::OMPD_parallel_workshare:
    CheckWorkshareBlockStmts(block, beginSpec.source);
    HasInvalidWorksharingNesting(
        beginSpec.source, llvm::omp::nestedWorkshareErrSet);
    break;
  case llvm::omp::OMPD_workdistribute:
    if (!CurrentDirectiveIsNested()) {
      context_.Say(beginSpec.source,
          "A WORKDISTRIBUTE region must be nested inside TEAMS region only."_err_en_US);
    }
    CheckWorkdistributeBlockStmts(block, beginSpec.source);
    break;
  case llvm::omp::OMPD_teams_workdistribute:
  case llvm::omp::OMPD_target_teams_workdistribute:
    CheckWorkdistributeBlockStmts(block, beginSpec.source);
    break;
  case llvm::omp::Directive::OMPD_scope:
  case llvm::omp::Directive::OMPD_single:
    // TODO: This check needs to be extended while implementing nesting of
    // regions checks.
    HasInvalidWorksharingNesting(
        beginSpec.source, llvm::omp::nestedWorkshareErrSet);
    break;
  case llvm::omp::Directive::OMPD_task:
    for (const auto &clause : beginSpec.Clauses().v) {
      if (std::get_if<parser::OmpClause::Untied>(&clause.u)) {
        OmpUnitedTaskDesignatorChecker check{context_};
        parser::Walk(block, check);
      }
    }
    break;
  default:
    break;
  }
}

void OmpStructureChecker::CheckMasterNesting(
    const parser::OpenMPBlockConstruct &x) {
  // A MASTER region may not be `closely nested` inside a worksharing, loop,
  // task, taskloop, or atomic region.
  // TODO:  Expand the check to include `LOOP` construct as well when it is
  // supported.
  if (IsCloselyNestedRegion(llvm::omp::nestedMasterErrSet)) {
    context_.Say(x.BeginDir().source,
        "`MASTER` region may not be closely nested inside of `WORKSHARING`, "
        "`LOOP`, `TASK`, `TASKLOOP`,"
        " or `ATOMIC` region."_err_en_US);
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPAssumeConstruct &x) {
  PushContextAndClauseSets(x.source, llvm::omp::Directive::OMPD_assume);
}

void OmpStructureChecker::Leave(const parser::OpenMPAssumeConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclarativeAssumes &x) {
  PushContextAndClauseSets(x.source, llvm::omp::Directive::OMPD_assumes);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclarativeAssumes &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Leave(const parser::OpenMPBlockConstruct &) {
  if (GetDirectiveNest(TargetBlockOnlyTeams)) {
    ExitDirectiveNest(TargetBlockOnlyTeams);
  }
  if (llvm::omp::allTargetSet.test(GetContext().directive)) {
    ExitDirectiveNest(TargetNest);
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::ChecksOnOrderedAsBlock() {
  if (FindClause(llvm::omp::Clause::OMPC_depend)) {
    context_.Say(GetContext().clauseSource,
        "DEPEND clauses are not allowed when ORDERED construct is a block construct with an ORDERED region"_err_en_US);
    return;
  }

  bool isNestedInDo{false};
  bool isNestedInDoSIMD{false};
  bool isNestedInSIMD{false};
  bool noOrderedClause{false};
  bool isOrderedClauseWithPara{false};
  bool isCloselyNestedRegion{true};
  if (CurrentDirectiveIsNested()) {
    for (int i = (int)dirContext_.size() - 2; i >= 0; i--) {
      if (llvm::omp::nestedOrderedErrSet.test(dirContext_[i].directive)) {
        context_.Say(GetContext().directiveSource,
            "`ORDERED` region may not be closely nested inside of `CRITICAL`, "
            "`ORDERED`, explicit `TASK` or `TASKLOOP` region."_err_en_US);
        break;
      } else if (llvm::omp::allDoSet.test(dirContext_[i].directive)) {
        isNestedInDo = true;
        isNestedInDoSIMD =
            llvm::omp::allDoSimdSet.test(dirContext_[i].directive);
        if (const auto *clause{
                FindClause(dirContext_[i], llvm::omp::Clause::OMPC_ordered)}) {
          const auto &orderedClause{
              std::get<parser::OmpClause::Ordered>(clause->u)};
          const auto orderedValue{GetIntValue(orderedClause.v)};
          isOrderedClauseWithPara = orderedValue > 0;
        } else {
          noOrderedClause = true;
        }
        break;
      } else if (llvm::omp::allSimdSet.test(dirContext_[i].directive)) {
        isNestedInSIMD = true;
        break;
      } else if (llvm::omp::nestedOrderedParallelErrSet.test(
                     dirContext_[i].directive)) {
        isCloselyNestedRegion = false;
        break;
      }
    }
  }

  if (!isCloselyNestedRegion) {
    context_.Say(GetContext().directiveSource,
        "An ORDERED directive without the DEPEND clause must be closely nested "
        "in a SIMD, worksharing-loop, or worksharing-loop SIMD "
        "region"_err_en_US);
  } else {
    if (CurrentDirectiveIsNested() &&
        FindClause(llvm::omp::Clause::OMPC_simd) &&
        (!isNestedInDoSIMD && !isNestedInSIMD)) {
      context_.Say(GetContext().directiveSource,
          "An ORDERED directive with SIMD clause must be closely nested in a "
          "SIMD or worksharing-loop SIMD region"_err_en_US);
    }
    if (isNestedInDo && (noOrderedClause || isOrderedClauseWithPara)) {
      context_.Say(GetContext().directiveSource,
          "An ORDERED directive without the DEPEND clause must be closely "
          "nested in a worksharing-loop (or worksharing-loop SIMD) region with "
          "ORDERED clause without the parameter"_err_en_US);
    }
  }
}

void OmpStructureChecker::Leave(const parser::OmpBeginDirective &) {
  switch (GetContext().directive) {
  case llvm::omp::Directive::OMPD_ordered:
    // [5.1] 2.19.9 Ordered Construct Restriction
    ChecksOnOrderedAsBlock();
    break;
  default:
    break;
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPSectionsConstruct &x) {
  const auto &beginSectionsDir{
      std::get<parser::OmpBeginSectionsDirective>(x.t)};
  const auto &endSectionsDir{
      std::get<std::optional<parser::OmpEndSectionsDirective>>(x.t)};
  const auto &beginDir{
      std::get<parser::OmpSectionsDirective>(beginSectionsDir.t)};
  PushContextAndClauseSets(beginDir.source, beginDir.v);

  if (!endSectionsDir) {
    context_.Say(beginSectionsDir.source,
        "Expected OpenMP END SECTIONS directive"_err_en_US);
    // Following code assumes the option is present.
    return;
  }

  const auto &endDir{std::get<parser::OmpSectionsDirective>(endSectionsDir->t)};
  CheckMatching<parser::OmpSectionsDirective>(beginDir, endDir);

  AddEndDirectiveClauses(std::get<parser::OmpClauseList>(endSectionsDir->t));

  const auto &sectionBlocks{std::get<std::list<parser::OpenMPConstruct>>(x.t)};
  for (const parser::OpenMPConstruct &construct : sectionBlocks) {
    auto &section{std::get<parser::OpenMPSectionConstruct>(construct.u)};
    CheckNoBranching(
        std::get<parser::Block>(section.t), beginDir.v, beginDir.source);
  }
  HasInvalidWorksharingNesting(
      beginDir.source, llvm::omp::nestedWorkshareErrSet);
}

void OmpStructureChecker::Leave(const parser::OpenMPSectionsConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndSectionsDirective &x) {
  const auto &dir{std::get<parser::OmpSectionsDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
    // 2.7.2 end-sections -> END SECTIONS [nowait-clause]
  case llvm::omp::Directive::OMPD_sections:
    PushContextAndClauseSets(
        dir.source, llvm::omp::Directive::OMPD_end_sections);
    break;
  default:
    // no clauses are allowed
    break;
  }
}

// TODO: Verify the popping of dirContext requirement after nowait
// implementation, as there is an implicit barrier at the end of the worksharing
// constructs unless a nowait clause is specified. Only OMPD_end_sections is
// popped becuase it is pushed while entering the EndSectionsDirective.
void OmpStructureChecker::Leave(const parser::OmpEndSectionsDirective &x) {
  if (GetContext().directive == llvm::omp::Directive::OMPD_end_sections) {
    dirContext_.pop_back();
  }
}

void OmpStructureChecker::CheckThreadprivateOrDeclareTargetVar(
    const parser::Designator &designator) {
  auto *name{parser::Unwrap<parser::Name>(designator)};
  // If the symbol is null, return early, CheckSymbolNames
  // should have already reported the missing symbol as a
  // diagnostic error
  if (!name || !name->symbol) {
    return;
  }

  llvm::omp::Directive directive{GetContext().directive};

  if (name->symbol->GetUltimate().IsSubprogram()) {
    if (directive == llvm::omp::Directive::OMPD_threadprivate)
      context_.Say(name->source,
          "The procedure name cannot be in a %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    // TODO: Check for procedure name in declare target directive.
  } else if (name->symbol->attrs().test(Attr::PARAMETER)) {
    if (directive == llvm::omp::Directive::OMPD_threadprivate)
      context_.Say(name->source,
          "The entity with PARAMETER attribute cannot be in a %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    else if (directive == llvm::omp::Directive::OMPD_declare_target)
      context_.Warn(common::UsageWarning::OpenMPUsage, name->source,
          "The entity with PARAMETER attribute is used in a %s directive"_warn_en_US,
          ContextDirectiveAsFortran());
  } else if (FindCommonBlockContaining(*name->symbol)) {
    context_.Say(name->source,
        "A variable in a %s directive cannot be an element of a common block"_err_en_US,
        ContextDirectiveAsFortran());
  } else if (FindEquivalenceSet(*name->symbol)) {
    context_.Say(name->source,
        "A variable in a %s directive cannot appear in an EQUIVALENCE statement"_err_en_US,
        ContextDirectiveAsFortran());
  } else if (name->symbol->test(Symbol::Flag::OmpThreadprivate) &&
      directive == llvm::omp::Directive::OMPD_declare_target) {
    context_.Say(name->source,
        "A THREADPRIVATE variable cannot appear in a %s directive"_err_en_US,
        ContextDirectiveAsFortran());
  } else {
    const semantics::Scope &useScope{
        context_.FindScope(GetContext().directiveSource)};
    const semantics::Scope &curScope = name->symbol->GetUltimate().owner();
    if (!curScope.IsTopLevel()) {
      const semantics::Scope &declScope =
          GetProgramUnitOrBlockConstructContaining(curScope);
      const semantics::Symbol *sym{
          declScope.parent().FindSymbol(name->symbol->name())};
      if (sym &&
          (sym->has<MainProgramDetails>() || sym->has<ModuleDetails>())) {
        context_.Say(name->source,
            "The module name cannot be in a %s directive"_err_en_US,
            ContextDirectiveAsFortran());
      } else if (!IsSaved(*name->symbol) &&
          declScope.kind() != Scope::Kind::MainProgram &&
          declScope.kind() != Scope::Kind::Module) {
        context_.Say(name->source,
            "A variable that appears in a %s directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly"_err_en_US,
            ContextDirectiveAsFortran());
      } else if (useScope != declScope) {
        context_.Say(name->source,
            "The %s directive and the common block or variable in it must appear in the same declaration section of a scoping unit"_err_en_US,
            ContextDirectiveAsFortran());
      }
    }
  }
}

void OmpStructureChecker::CheckThreadprivateOrDeclareTargetVar(
    const parser::Name &name) {
  if (!name.symbol) {
    return;
  }

  if (auto *cb{name.symbol->detailsIf<CommonBlockDetails>()}) {
    for (const auto &obj : cb->objects()) {
      if (FindEquivalenceSet(*obj)) {
        context_.Say(name.source,
            "A variable in a %s directive cannot appear in an EQUIVALENCE statement (variable '%s' from common block '/%s/')"_err_en_US,
            ContextDirectiveAsFortran(), obj->name(), name.symbol->name());
      }
    }
  }
}

void OmpStructureChecker::CheckThreadprivateOrDeclareTargetVar(
    const parser::OmpObjectList &objList) {
  for (const auto &ompObject : objList.v) {
    common::visit([&](auto &&s) { CheckThreadprivateOrDeclareTargetVar(s); },
        ompObject.u);
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPGroupprivate &x) {
  PushContextAndClauseSets(
      x.v.DirName().source, llvm::omp::Directive::OMPD_groupprivate);

  for (const parser::OmpArgument &arg : x.v.Arguments().v) {
    auto *locator{std::get_if<parser::OmpLocator>(&arg.u)};
    const Symbol *sym{GetArgumentSymbol(arg)};

    if (!locator || !sym ||
        (!IsVariableListItem(*sym) && !IsCommonBlock(*sym))) {
      context_.Say(arg.source,
          "GROUPPRIVATE argument should be a variable or a named common block"_err_en_US);
      continue;
    }

    if (sym->has<AssocEntityDetails>()) {
      context_.SayWithDecl(*sym, arg.source,
          "GROUPPRIVATE argument cannot be an ASSOCIATE name"_err_en_US);
      continue;
    }
    if (auto *obj{sym->detailsIf<ObjectEntityDetails>()}) {
      if (obj->IsCoarray()) {
        context_.Say(
            arg.source, "GROUPPRIVATE argument cannot be a coarray"_err_en_US);
        continue;
      }
      if (obj->init()) {
        context_.SayWithDecl(*sym, arg.source,
            "GROUPPRIVATE argument cannot be declared with an initializer"_err_en_US);
        continue;
      }
    }
    if (sym->test(Symbol::Flag::InCommonBlock)) {
      context_.Say(arg.source,
          "GROUPPRIVATE argument cannot be a member of a common block"_err_en_US);
      continue;
    }
    if (!IsCommonBlock(*sym)) {
      const Scope &thisScope{context_.FindScope(x.v.source)};
      if (thisScope != sym->owner()) {
        context_.SayWithDecl(*sym, arg.source,
            "GROUPPRIVATE argument variable must be declared in the same scope as the construct on which it appears"_err_en_US);
        continue;
      } else if (!thisScope.IsModule() && !sym->attrs().test(Attr::SAVE)) {
        context_.SayWithDecl(*sym, arg.source,
            "GROUPPRIVATE argument variable must be declared in the module scope or have SAVE attribute"_err_en_US);
        continue;
      }
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPGroupprivate &x) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPThreadprivate &c) {
  const auto &dir{std::get<parser::Verbatim>(c.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_threadprivate);
}

void OmpStructureChecker::Leave(const parser::OpenMPThreadprivate &c) {
  const auto &dir{std::get<parser::Verbatim>(c.t)};
  const auto &objectList{std::get<parser::OmpObjectList>(c.t)};
  CheckSymbolNames(dir.source, objectList);
  CheckVarIsNotPartOfAnotherVar(dir.source, objectList);
  CheckThreadprivateOrDeclareTargetVar(objectList);
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareSimdConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_declare_simd);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareSimdConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpDeclareVariantDirective &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_declare_variant);
}

void OmpStructureChecker::Leave(const parser::OmpDeclareVariantDirective &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDepobjConstruct &x) {
  const auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};
  PushContextAndClauseSets(dirName.source, llvm::omp::Directive::OMPD_depobj);
  unsigned version{context_.langOptions().OpenMPVersion};

  const parser::OmpArgumentList &arguments{x.v.Arguments()};
  const parser::OmpClauseList &clauses{x.v.Clauses()};

  // Ref: [6.0:505-506]

  if (version < 60) {
    if (arguments.v.size() != 1) {
      parser::CharBlock source(
          arguments.v.empty() ? dirName.source : arguments.source);
      context_.Say(
          source, "The DEPOBJ directive requires a single argument"_err_en_US);
    }
  }
  if (clauses.v.size() != 1) {
    context_.Say(
        x.source, "The DEPOBJ construct requires a single clause"_err_en_US);
    return;
  }

  auto &clause{clauses.v.front()};

  if (version >= 60 && arguments.v.empty()) {
    context_.Say(x.source,
        "DEPOBJ syntax with no argument is not handled yet"_err_en_US);
    return;
  }

  // [5.2:73:27-28]
  // If the destroy clause appears on a depobj construct, destroy-var must
  // refer to the same depend object as the depobj argument of the construct.
  if (clause.Id() == llvm::omp::Clause::OMPC_destroy) {
    auto getObjSymbol{[&](const parser::OmpObject &obj) {
      return common::visit(
          [&](auto &&s) { return GetLastName(s).symbol; }, obj.u);
    }};
    auto getArgSymbol{[&](const parser::OmpArgument &arg) {
      if (auto *locator{std::get_if<parser::OmpLocator>(&arg.u)}) {
        if (auto *object{std::get_if<parser::OmpObject>(&locator->u)}) {
          return getObjSymbol(*object);
        }
      }
      return static_cast<Symbol *>(nullptr);
    }};

    auto &wrapper{std::get<parser::OmpClause::Destroy>(clause.u)};
    if (const std::optional<parser::OmpDestroyClause> &destroy{wrapper.v}) {
      const Symbol *constrSym{getArgSymbol(arguments.v.front())};
      const Symbol *clauseSym{getObjSymbol(destroy->v)};
      assert(constrSym && "Unresolved depobj construct symbol");
      assert(clauseSym && "Unresolved destroy symbol on depobj construct");
      if (constrSym != clauseSym) {
        context_.Say(x.source,
            "The DESTROY clause must refer to the same object as the "
            "DEPOBJ construct"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDepobjConstruct &x) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPRequiresConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_requires);

  if (visitedAtomicSource_.empty()) {
    return;
  }
  const auto &clauseList{std::get<parser::OmpClauseList>(x.t)};
  for (const parser::OmpClause &clause : clauseList.v) {
    llvm::omp::Clause id{clause.Id()};
    if (id == llvm::omp::Clause::OMPC_atomic_default_mem_order) {
      parser::MessageFormattedText txt(
          "REQUIRES directive with '%s' clause found lexically after atomic operation without a memory order clause"_err_en_US,
          parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(id)));
      parser::Message message(clause.source, txt);
      message.Attach(visitedAtomicSource_, "Previous atomic construct"_en_US);
      context_.Say(std::move(message));
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPRequiresConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::CheckAlignValue(const parser::OmpClause &clause) {
  if (auto *align{std::get_if<parser::OmpClause::Align>(&clause.u)}) {
    if (const auto &v{GetIntValue(align->v)}; !v || *v <= 0) {
      context_.Say(clause.source,
          "The alignment value should be a constant positive integer"_err_en_US);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclarativeAllocate &x) {
  isPredefinedAllocator = true;
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<parser::OmpObjectList>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_allocate);
  const auto &clauseList{std::get<parser::OmpClauseList>(x.t)};
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(objectList, currSymbols);
  for (auto &[symbol, source] : currSymbols) {
    if (IsPointer(*symbol)) {
      context_.Say(source,
          "List item '%s' in ALLOCATE directive must not have POINTER "
          "attribute"_err_en_US,
          source.ToString());
    }
    if (IsDummy(*symbol)) {
      context_.Say(source,
          "List item '%s' in ALLOCATE directive must not be a dummy "
          "argument"_err_en_US,
          source.ToString());
    }
    if (symbol->GetUltimate().has<AssocEntityDetails>()) {
      context_.Say(source,
          "List item '%s' in ALLOCATE directive must not be an associate "
          "name"_err_en_US,
          source.ToString());
    }
  }
  for (const auto &clause : clauseList.v) {
    CheckAlignValue(clause);
  }
  CheckVarIsNotPartOfAnotherVar(dir.source, objectList);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclarativeAllocate &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<parser::OmpObjectList>(x.t)};
  CheckPredefinedAllocatorRestriction(dir.source, objectList);
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpClause::Allocator &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_allocator);
  // Note: Predefined allocators are stored in ScalarExpr as numbers
  //   whereas custom allocators are stored as strings, so if the ScalarExpr
  //   actually has an int value, then it must be a predefined allocator
  isPredefinedAllocator = GetIntValue(x.v).has_value();
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_allocator, x.v);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Allocate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_allocate);
  if (OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_allocate, GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    if (auto *align{
            OmpGetUniqueModifier<parser::OmpAlignModifier>(modifiers)}) {
      if (const auto &v{GetIntValue(align->v)}; !v || *v <= 0) {
        context_.Say(OmpGetModifierSource(modifiers, align),
            "The alignment value should be a constant positive integer"_err_en_US);
      }
    }
    // The simple and complex modifiers have the same structure. They only
    // differ in their syntax.
    if (auto *alloc{OmpGetUniqueModifier<parser::OmpAllocatorComplexModifier>(
            modifiers)}) {
      isPredefinedAllocator = GetIntValue(alloc->v).has_value();
    }
    if (auto *alloc{OmpGetUniqueModifier<parser::OmpAllocatorSimpleModifier>(
            modifiers)}) {
      isPredefinedAllocator = GetIntValue(alloc->v).has_value();
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpDeclareTargetWithClause &x) {
  SetClauseSets(llvm::omp::Directive::OMPD_declare_target);
}

void OmpStructureChecker::Leave(const parser::OmpDeclareTargetWithClause &x) {
  if (x.v.v.size() > 0) {
    const parser::OmpClause *enterClause =
        FindClause(llvm::omp::Clause::OMPC_enter);
    const parser::OmpClause *toClause = FindClause(llvm::omp::Clause::OMPC_to);
    const parser::OmpClause *linkClause =
        FindClause(llvm::omp::Clause::OMPC_link);
    const parser::OmpClause *indirectClause =
        FindClause(llvm::omp::Clause::OMPC_indirect);
    if (!enterClause && !toClause && !linkClause) {
      context_.Say(x.source,
          "If the DECLARE TARGET directive has a clause, it must contain at least one ENTER clause or LINK clause"_err_en_US);
    }
    if (indirectClause && !enterClause) {
      context_.Say(x.source,
          "The INDIRECT clause cannot be used without the ENTER clause with the DECLARE TARGET directive."_err_en_US);
    }
    unsigned version{context_.langOptions().OpenMPVersion};
    if (toClause && version >= 52) {
      context_.Warn(common::UsageWarning::OpenMPUsage, toClause->source,
          "The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead."_warn_en_US);
    }
    if (indirectClause) {
      CheckAllowedClause(llvm::omp::Clause::OMPC_indirect);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareMapperConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_declare_mapper);
  const auto &spec{std::get<parser::OmpMapperSpecifier>(x.t)};
  const auto &type = std::get<parser::TypeSpec>(spec.t);
  if (!std::get_if<parser::DerivedTypeSpec>(&type.u)) {
    context_.Say(dir.source, "Type is not a derived type"_err_en_US);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareMapperConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OpenMPDeclareReductionConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_declare_reduction);
}

void OmpStructureChecker::Leave(
    const parser::OpenMPDeclareReductionConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareTargetConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, llvm::omp::Directive::OMPD_declare_target);
}

void OmpStructureChecker::Enter(const parser::OmpDeclareTargetWithList &x) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(x.v, symbols);
  for (auto &[symbol, source] : symbols) {
    const GenericDetails *genericDetails = symbol->detailsIf<GenericDetails>();
    if (genericDetails) {
      context_.Say(source,
          "The procedure '%s' in DECLARE TARGET construct cannot be a generic name."_err_en_US,
          symbol->name());
      genericDetails->specific();
    }
    if (IsProcedurePointer(*symbol)) {
      context_.Say(source,
          "The procedure '%s' in DECLARE TARGET construct cannot be a procedure pointer."_err_en_US,
          symbol->name());
    }
    const SubprogramDetails *entryDetails =
        symbol->detailsIf<SubprogramDetails>();
    if (entryDetails && entryDetails->entryScope()) {
      context_.Say(source,
          "The procedure '%s' in DECLARE TARGET construct cannot be an entry name."_err_en_US,
          symbol->name());
    }
    if (IsStmtFunction(*symbol)) {
      context_.Say(source,
          "The procedure '%s' in DECLARE TARGET construct cannot be a statement function."_err_en_US,
          symbol->name());
    }
  }
}

void OmpStructureChecker::CheckSymbolNames(
    const parser::CharBlock &source, const parser::OmpObjectList &objList) {
  for (const auto &ompObject : objList.v) {
    common::visit(
        common::visitors{
            [&](const parser::Designator &designator) {
              if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
                if (!name->symbol) {
                  context_.Say(source,
                      "The given %s directive clause has an invalid argument"_err_en_US,
                      ContextDirectiveAsFortran());
                }
              }
            },
            [&](const parser::Name &name) {
              if (!name.symbol) {
                context_.Say(source,
                    "The given %s directive clause has an invalid argument"_err_en_US,
                    ContextDirectiveAsFortran());
              }
            },
        },
        ompObject.u);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareTargetConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
  // Handle both forms of DECLARE TARGET.
  // - Extended list: It behaves as if there was an ENTER/TO clause with the
  //   list of objects as argument. It accepts no explicit clauses.
  // - With clauses.
  if (const auto *objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
    deviceConstructFound_ = true;
    CheckSymbolNames(dir.source, *objectList);
    CheckVarIsNotPartOfAnotherVar(dir.source, *objectList);
    CheckThreadprivateOrDeclareTargetVar(*objectList);
  } else if (const auto *clauseList{
                 parser::Unwrap<parser::OmpClauseList>(spec.u)}) {
    bool toClauseFound{false}, deviceTypeClauseFound{false},
        enterClauseFound{false};
    for (const auto &clause : clauseList->v) {
      common::visit(
          common::visitors{
              [&](const parser::OmpClause::To &toClause) {
                toClauseFound = true;
                auto &objList{std::get<parser::OmpObjectList>(toClause.v.t)};
                CheckSymbolNames(dir.source, objList);
                CheckVarIsNotPartOfAnotherVar(dir.source, objList);
                CheckThreadprivateOrDeclareTargetVar(objList);
              },
              [&](const parser::OmpClause::Link &linkClause) {
                CheckSymbolNames(dir.source, linkClause.v);
                CheckVarIsNotPartOfAnotherVar(dir.source, linkClause.v);
                CheckThreadprivateOrDeclareTargetVar(linkClause.v);
              },
              [&](const parser::OmpClause::Enter &enterClause) {
                enterClauseFound = true;
                auto &objList{std::get<parser::OmpObjectList>(enterClause.v.t)};
                CheckSymbolNames(dir.source, objList);
                CheckVarIsNotPartOfAnotherVar(dir.source, objList);
                CheckThreadprivateOrDeclareTargetVar(objList);
              },
              [&](const parser::OmpClause::DeviceType &deviceTypeClause) {
                deviceTypeClauseFound = true;
                if (deviceTypeClause.v.v !=
                    parser::OmpDeviceTypeClause::DeviceTypeDescription::Host) {
                  // Function / subroutine explicitly marked as runnable by the
                  // target device.
                  deviceConstructFound_ = true;
                }
              },
              [&](const auto &) {},
          },
          clause.u);

      if ((toClauseFound || enterClauseFound) && !deviceTypeClauseFound) {
        deviceConstructFound_ = true;
      }
    }
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpErrorDirective &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_error);
}

void OmpStructureChecker::Enter(const parser::OpenMPDispatchConstruct &x) {
  const parser::OmpDirectiveSpecification &dirSpec{x.BeginDir()};
  const auto &block{std::get<parser::Block>(x.t)};
  PushContextAndClauseSets(
      dirSpec.DirName().source, llvm::omp::Directive::OMPD_dispatch);

  if (block.empty()) {
    context_.Say(x.source,
        "The DISPATCH construct should contain a single function or subroutine call"_err_en_US);
    return;
  }

  bool passChecks{false};
  omp::SourcedActionStmt action{omp::GetActionStmt(block)};
  if (const auto *assignStmt{
          parser::Unwrap<parser::AssignmentStmt>(*action.stmt)}) {
    if (parser::Unwrap<parser::FunctionReference>(assignStmt->t)) {
      passChecks = true;
    }
  } else if (parser::Unwrap<parser::CallStmt>(*action.stmt)) {
    passChecks = true;
  }

  if (!passChecks) {
    context_.Say(action.source,
        "The body of the DISPATCH construct should be a function or a subroutine call"_err_en_US);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDispatchConstruct &x) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Leave(const parser::OmpErrorDirective &x) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpClause::At &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_at);
  if (GetDirectiveNest(DeclarativeNest) > 0) {
    if (x.v.v == parser::OmpAtClause::ActionTime::Execution) {
      context_.Say(GetContext().clauseSource,
          "The ERROR directive with AT(EXECUTION) cannot appear in the specification part"_err_en_US);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPExecutableAllocate &x) {
  isPredefinedAllocator = true;
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<std::optional<parser::OmpObjectList>>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_allocate);
  const auto &clauseList{std::get<parser::OmpClauseList>(x.t)};
  for (const auto &clause : clauseList.v) {
    CheckAlignValue(clause);
  }
  if (objectList) {
    CheckVarIsNotPartOfAnotherVar(dir.source, *objectList);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPExecutableAllocate &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<std::optional<parser::OmpObjectList>>(x.t)};
  if (objectList)
    CheckPredefinedAllocatorRestriction(dir.source, *objectList);
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPAllocatorsConstruct &x) {
  isPredefinedAllocator = true;

  const parser::OmpDirectiveSpecification &dirSpec{x.BeginDir()};
  auto &block{std::get<parser::Block>(x.t)};
  PushContextAndClauseSets(
      dirSpec.DirName().source, llvm::omp::Directive::OMPD_allocators);

  if (block.empty()) {
    context_.Say(dirSpec.source,
        "The ALLOCATORS construct should contain a single ALLOCATE statement"_err_en_US);
    return;
  }

  omp::SourcedActionStmt action{omp::GetActionStmt(block)};
  const auto *allocate{
      action ? parser::Unwrap<parser::AllocateStmt>(action.stmt) : nullptr};

  if (!allocate) {
    const parser::CharBlock &source = action ? action.source : x.source;
    context_.Say(source,
        "The body of the ALLOCATORS construct should be an ALLOCATE statement"_err_en_US);
  }

  for (const auto &clause : dirSpec.Clauses().v) {
    if (const auto *allocClause{
            parser::Unwrap<parser::OmpClause::Allocate>(clause)}) {
      CheckVarIsNotPartOfAnotherVar(
          dirSpec.source, std::get<parser::OmpObjectList>(allocClause->v.t));
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPAllocatorsConstruct &x) {
  const parser::OmpDirectiveSpecification &dirSpec{x.BeginDir()};

  for (const auto &clause : dirSpec.Clauses().v) {
    if (const auto *allocClause{
            std::get_if<parser::OmpClause::Allocate>(&clause.u)}) {
      CheckPredefinedAllocatorRestriction(
          dirSpec.source, std::get<parser::OmpObjectList>(allocClause->v.t));
    }
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::CheckScan(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  if (x.v.Clauses().v.size() != 1) {
    context_.Say(x.source,
        "Exactly one of EXCLUSIVE or INCLUSIVE clause is expected"_err_en_US);
  }
  if (!CurrentDirectiveIsNested() ||
      !llvm::omp::scanParentAllowedSet.test(GetContextParent().directive)) {
    context_.Say(x.source,
        "Orphaned SCAN directives are prohibited; perhaps you forgot "
        "to enclose the directive in to a WORKSHARING LOOP, a WORKSHARING "
        "LOOP SIMD or a SIMD directive."_err_en_US);
  }
}

void OmpStructureChecker::CheckBarrierNesting(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  // A barrier region may not be `closely nested` inside a worksharing, loop,
  // task, taskloop, critical, ordered, atomic, or master region.
  // TODO:  Expand the check to include `LOOP` construct as well when it is
  // supported.
  if (IsCloselyNestedRegion(llvm::omp::nestedBarrierErrSet)) {
    context_.Say(x.v.DirName().source,
        "`BARRIER` region may not be closely nested inside of `WORKSHARING`, "
        "`LOOP`, `TASK`, `TASKLOOP`,"
        "`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region."_err_en_US);
  }
}

void OmpStructureChecker::ChecksOnOrderedAsStandalone() {
  if (FindClause(llvm::omp::Clause::OMPC_threads) ||
      FindClause(llvm::omp::Clause::OMPC_simd)) {
    context_.Say(GetContext().clauseSource,
        "THREADS and SIMD clauses are not allowed when ORDERED construct is a standalone construct with no ORDERED region"_err_en_US);
  }

  int dependSinkCount{0}, dependSourceCount{0};
  bool exclusiveShown{false}, duplicateSourceShown{false};

  auto visitDoacross{[&](const parser::OmpDoacross &doa,
                         const parser::CharBlock &src) {
    common::visit(
        common::visitors{
            [&](const parser::OmpDoacross::Source &) { dependSourceCount++; },
            [&](const parser::OmpDoacross::Sink &) { dependSinkCount++; }},
        doa.u);
    if (!exclusiveShown && dependSinkCount > 0 && dependSourceCount > 0) {
      exclusiveShown = true;
      context_.Say(src,
          "The SINK and SOURCE dependence types are mutually exclusive"_err_en_US);
    }
    if (!duplicateSourceShown && dependSourceCount > 1) {
      duplicateSourceShown = true;
      context_.Say(src,
          "At most one SOURCE dependence type can appear on the ORDERED directive"_err_en_US);
    }
  }};

  // Visit the DEPEND and DOACROSS clauses.
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_depend)) {
    const auto &dependClause{std::get<parser::OmpClause::Depend>(clause->u)};
    if (auto *doAcross{std::get_if<parser::OmpDoacross>(&dependClause.v.u)}) {
      visitDoacross(*doAcross, clause->source);
    } else {
      context_.Say(clause->source,
          "Only SINK or SOURCE dependence types are allowed when ORDERED construct is a standalone construct with no ORDERED region"_err_en_US);
    }
  }
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_doacross)) {
    auto &doaClause{std::get<parser::OmpClause::Doacross>(clause->u)};
    visitDoacross(doaClause.v.v, clause->source);
  }

  bool isNestedInDoOrderedWithPara{false};
  if (CurrentDirectiveIsNested() &&
      llvm::omp::nestedOrderedDoAllowedSet.test(GetContextParent().directive)) {
    if (const auto *clause{
            FindClause(GetContextParent(), llvm::omp::Clause::OMPC_ordered)}) {
      const auto &orderedClause{
          std::get<parser::OmpClause::Ordered>(clause->u)};
      const auto orderedValue{GetIntValue(orderedClause.v)};
      if (orderedValue > 0) {
        isNestedInDoOrderedWithPara = true;
        CheckOrderedDependClause(orderedValue);
      }
    }
  }

  if (FindClause(llvm::omp::Clause::OMPC_depend) &&
      !isNestedInDoOrderedWithPara) {
    context_.Say(GetContext().clauseSource,
        "An ORDERED construct with the DEPEND clause must be closely nested "
        "in a worksharing-loop (or parallel worksharing-loop) construct with "
        "ORDERED clause with a parameter"_err_en_US);
  }
}

void OmpStructureChecker::CheckOrderedDependClause(
    std::optional<int64_t> orderedValue) {
  auto visitDoacross{[&](const parser::OmpDoacross &doa,
                         const parser::CharBlock &src) {
    if (auto *sinkVector{std::get_if<parser::OmpDoacross::Sink>(&doa.u)}) {
      int64_t numVar = sinkVector->v.v.size();
      if (orderedValue != numVar) {
        context_.Say(src,
            "The number of variables in the SINK iteration vector does not match the parameter specified in ORDERED clause"_err_en_US);
      }
    }
  }};
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_depend)) {
    auto &dependClause{std::get<parser::OmpClause::Depend>(clause->u)};
    if (auto *doAcross{std::get_if<parser::OmpDoacross>(&dependClause.v.u)}) {
      visitDoacross(*doAcross, clause->source);
    }
  }
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_doacross)) {
    auto &doaClause{std::get<parser::OmpClause::Doacross>(clause->u)};
    visitDoacross(doaClause.v.v, clause->source);
  }
}

void OmpStructureChecker::CheckTargetUpdate() {
  const parser::OmpClause *toWrapper{FindClause(llvm::omp::Clause::OMPC_to)};
  const parser::OmpClause *fromWrapper{
      FindClause(llvm::omp::Clause::OMPC_from)};
  if (!toWrapper && !fromWrapper) {
    context_.Say(GetContext().directiveSource,
        "At least one motion-clause (TO/FROM) must be specified on "
        "TARGET UPDATE construct."_err_en_US);
  }
  if (toWrapper && fromWrapper) {
    SymbolSourceMap toSymbols, fromSymbols;
    auto &fromClause{std::get<parser::OmpClause::From>(fromWrapper->u).v};
    auto &toClause{std::get<parser::OmpClause::To>(toWrapper->u).v};
    GetSymbolsInObjectList(
        std::get<parser::OmpObjectList>(fromClause.t), fromSymbols);
    GetSymbolsInObjectList(
        std::get<parser::OmpObjectList>(toClause.t), toSymbols);

    for (auto &[symbol, source] : toSymbols) {
      auto fromSymbol{fromSymbols.find(symbol)};
      if (fromSymbol != fromSymbols.end()) {
        context_.Say(source,
            "A list item ('%s') can only appear in a TO or FROM clause, but not in both."_err_en_US,
            symbol->name());
        context_.Say(source, "'%s' appears in the TO clause."_because_en_US,
            symbol->name());
        context_.Say(fromSymbol->second,
            "'%s' appears in the FROM clause."_because_en_US,
            fromSymbol->first->name());
      }
    }
  }
}

void OmpStructureChecker::CheckTaskDependenceType(
    const parser::OmpTaskDependenceType::Value &x) {
  // Common checks for task-dependence-type (DEPEND and UPDATE clauses).
  unsigned version{context_.langOptions().OpenMPVersion};
  unsigned since{0};

  switch (x) {
  case parser::OmpTaskDependenceType::Value::In:
  case parser::OmpTaskDependenceType::Value::Out:
  case parser::OmpTaskDependenceType::Value::Inout:
    break;
  case parser::OmpTaskDependenceType::Value::Mutexinoutset:
  case parser::OmpTaskDependenceType::Value::Depobj:
    since = 50;
    break;
  case parser::OmpTaskDependenceType::Value::Inoutset:
    since = 52;
    break;
  }

  if (version < since) {
    context_.Say(GetContext().clauseSource,
        "%s task dependence type is not supported in %s, %s"_warn_en_US,
        parser::ToUpperCaseLetters(
            parser::OmpTaskDependenceType::EnumToString(x)),
        ThisVersion(version), TryVersion(since));
  }
}

void OmpStructureChecker::CheckDependenceType(
    const parser::OmpDependenceType::Value &x) {
  // Common checks for dependence-type (DEPEND and UPDATE clauses).
  unsigned version{context_.langOptions().OpenMPVersion};
  unsigned deprecatedIn{~0u};

  switch (x) {
  case parser::OmpDependenceType::Value::Source:
  case parser::OmpDependenceType::Value::Sink:
    deprecatedIn = 52;
    break;
  }

  if (version >= deprecatedIn) {
    context_.Say(GetContext().clauseSource,
        "%s dependence type is deprecated in %s"_warn_en_US,
        parser::ToUpperCaseLetters(parser::OmpDependenceType::EnumToString(x)),
        ThisVersion(deprecatedIn));
  }
}

void OmpStructureChecker::Enter(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  const auto &dir{std::get<parser::OmpDirectiveName>(x.v.t)};
  PushContextAndClauseSets(dir.source, dir.v);
  switch (dir.v) {
  case llvm::omp::Directive::OMPD_barrier:
    CheckBarrierNesting(x);
    break;
  case llvm::omp::Directive::OMPD_scan:
    CheckScan(x);
    break;
  default:
    break;
  }
}

void OmpStructureChecker::Leave(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  switch (GetContext().directive) {
  case llvm::omp::Directive::OMPD_ordered:
    // [5.1] 2.19.9 Ordered Construct Restriction
    ChecksOnOrderedAsStandalone();
    break;
  case llvm::omp::Directive::OMPD_target_update:
    CheckTargetUpdate();
    break;
  default:
    break;
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPFlushConstruct &x) {
  const auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};
  PushContextAndClauseSets(dirName.source, llvm::omp::Directive::OMPD_flush);
}

void OmpStructureChecker::Leave(const parser::OpenMPFlushConstruct &x) {
  auto &flushList{std::get<std::optional<parser::OmpArgumentList>>(x.v.t)};

  auto isVariableListItemOrCommonBlock{[](const Symbol &sym) {
    return IsVariableListItem(sym) ||
        sym.detailsIf<semantics::CommonBlockDetails>();
  }};

  if (flushList) {
    for (const parser::OmpArgument &arg : flushList->v) {
      if (auto *sym{GetArgumentSymbol(arg)};
          sym && !isVariableListItemOrCommonBlock(*sym)) {
        context_.Say(arg.source,
            "FLUSH argument must be a variable list item"_err_en_US);
      }
    }

    if (FindClause(llvm::omp::Clause::OMPC_acquire) ||
        FindClause(llvm::omp::Clause::OMPC_release) ||
        FindClause(llvm::omp::Clause::OMPC_acq_rel)) {
      context_.Say(flushList->source,
          "If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items must not be specified on the FLUSH directive"_err_en_US);
    }
  }

  unsigned version{context_.langOptions().OpenMPVersion};
  if (version >= 52) {
    using Flags = parser::OmpDirectiveSpecification::Flags;
    if (std::get<Flags>(x.v.t) == Flags::DeprecatedSyntax) {
      context_.Say(x.source,
          "The syntax \"FLUSH clause (object, ...)\" has been deprecated, use \"FLUSH(object, ...) clause\" instead"_warn_en_US);
    }
  }

  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCancelConstruct &x) {
  auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};
  auto &maybeClauses{std::get<std::optional<parser::OmpClauseList>>(x.v.t)};
  PushContextAndClauseSets(dirName.source, llvm::omp::Directive::OMPD_cancel);

  if (auto maybeConstruct{GetCancelType(
          llvm::omp::Directive::OMPD_cancel, x.source, maybeClauses)}) {
    CheckCancellationNest(dirName.source, *maybeConstruct);

    if (CurrentDirectiveIsNested()) {
      // nowait can be put on the end directive rather than the start directive
      // so we need to check both
      auto getParentClauses{[&]() {
        const DirectiveContext &parent{GetContextParent()};
        return llvm::concat<const llvm::omp::Clause>(
            parent.actualClauses, parent.endDirectiveClauses);
      }};

      if (llvm::omp::nestedCancelDoAllowedSet.test(*maybeConstruct)) {
        for (llvm::omp::Clause clause : getParentClauses()) {
          if (clause == llvm::omp::Clause::OMPC_nowait) {
            context_.Say(dirName.source,
                "The CANCEL construct cannot be nested inside of a worksharing construct with the NOWAIT clause"_err_en_US);
          }
          if (clause == llvm::omp::Clause::OMPC_ordered) {
            context_.Say(dirName.source,
                "The CANCEL construct cannot be nested inside of a worksharing construct with the ORDERED clause"_err_en_US);
          }
        }
      } else if (llvm::omp::nestedCancelSectionsAllowedSet.test(
                     *maybeConstruct)) {
        for (llvm::omp::Clause clause : getParentClauses()) {
          if (clause == llvm::omp::Clause::OMPC_nowait) {
            context_.Say(dirName.source,
                "The CANCEL construct cannot be nested inside of a worksharing construct with the NOWAIT clause"_err_en_US);
          }
        }
      }
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPCancelConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCriticalConstruct &x) {
  const parser::OmpBeginDirective &beginSpec{x.BeginDir()};
  const std::optional<parser::OmpEndDirective> &endSpec{x.EndDir()};
  PushContextAndClauseSets(beginSpec.DirName().source, beginSpec.DirName().v);

  const auto &block{std::get<parser::Block>(x.t)};
  CheckNoBranching(
      block, llvm::omp::Directive::OMPD_critical, beginSpec.DirName().source);

  auto getNameFromArg{[](const parser::OmpArgument &arg) {
    if (auto *object{parser::Unwrap<parser::OmpObject>(arg.u)}) {
      if (auto *designator{omp::GetDesignatorFromObj(*object)}) {
        return getDesignatorNameIfDataRef(*designator);
      }
    }
    return static_cast<const parser::Name *>(nullptr);
  }};

  auto checkArgumentList{[&](const parser::OmpArgumentList &args) {
    if (args.v.size() > 1) {
      context_.Say(args.source,
          "Only a single argument is allowed in CRITICAL directive"_err_en_US);
    } else if (!args.v.empty()) {
      if (!getNameFromArg(args.v.front())) {
        context_.Say(args.v.front().source,
            "CRITICAL argument should be a name"_err_en_US);
      }
    }
  }};

  const parser::Name *beginName{nullptr};
  const parser::Name *endName{nullptr};

  auto &beginArgs{beginSpec.Arguments()};
  checkArgumentList(beginArgs);

  if (!beginArgs.v.empty()) {
    beginName = getNameFromArg(beginArgs.v.front());
  }

  if (endSpec) {
    auto &endArgs{endSpec->Arguments()};
    checkArgumentList(endArgs);

    if (beginArgs.v.empty() != endArgs.v.empty()) {
      parser::CharBlock source{
          beginArgs.v.empty() ? endArgs.source : beginArgs.source};
      context_.Say(source,
          "Either both CRITICAL and END CRITICAL should have an argument, or none of them should"_err_en_US);
    } else if (!beginArgs.v.empty()) {
      endName = getNameFromArg(endArgs.v.front());
      if (beginName && endName) {
        if (beginName->ToString() != endName->ToString()) {
          context_.Say(endName->source,
              "The names on CRITICAL and END CRITICAL must match"_err_en_US);
        }
      }
    }
  }

  for (auto &clause : beginSpec.Clauses().v) {
    auto *hint{std::get_if<parser::OmpClause::Hint>(&clause.u)};
    if (!hint) {
      continue;
    }
    const int64_t OmpSyncHintNone = 0; // omp_sync_hint_none
    std::optional<int64_t> hintValue{GetIntValue(hint->v.v)};
    if (hintValue && *hintValue != OmpSyncHintNone) {
      // Emit a diagnostic if the name is missing, and point to the directive
      // with a missing name.
      parser::CharBlock source;
      if (!beginName) {
        source = beginSpec.DirName().source;
      } else if (endSpec && !endName) {
        source = endSpec->DirName().source;
      }

      if (!source.empty()) {
        context_.Say(source,
            "When HINT other than 'omp_sync_hint_none' is present, CRITICAL directive should have a name"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPCriticalConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OmpClause::CancellationConstructType &x) {
  llvm::omp::Directive dir{GetContext().directive};
  auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};

  if (dir != llvm::omp::Directive::OMPD_cancel &&
      dir != llvm::omp::Directive::OMPD_cancellation_point) {
    // Do not call CheckAllowed/CheckAllowedClause, because in case of an error
    // it will print "CANCELLATION_CONSTRUCT_TYPE" as the clause name instead
    // of the contained construct name.
    context_.Say(dirName.source, "%s cannot follow %s"_err_en_US,
        parser::ToUpperCaseLetters(getDirectiveName(dirName.v)),
        parser::ToUpperCaseLetters(getDirectiveName(dir)));
  } else {
    switch (dirName.v) {
    case llvm::omp::Directive::OMPD_do:
    case llvm::omp::Directive::OMPD_parallel:
    case llvm::omp::Directive::OMPD_sections:
    case llvm::omp::Directive::OMPD_taskgroup:
      break;
    default:
      context_.Say(dirName.source,
          "%s is not a cancellable construct"_err_en_US,
          parser::ToUpperCaseLetters(getDirectiveName(dirName.v)));
      break;
    }
  }
}

void OmpStructureChecker::Enter(
    const parser::OpenMPCancellationPointConstruct &x) {
  auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};
  auto &maybeClauses{std::get<std::optional<parser::OmpClauseList>>(x.v.t)};
  PushContextAndClauseSets(
      dirName.source, llvm::omp::Directive::OMPD_cancellation_point);

  if (auto maybeConstruct{
          GetCancelType(llvm::omp::Directive::OMPD_cancellation_point, x.source,
              maybeClauses)}) {
    CheckCancellationNest(dirName.source, *maybeConstruct);
  }
}

void OmpStructureChecker::Leave(
    const parser::OpenMPCancellationPointConstruct &) {
  dirContext_.pop_back();
}

std::optional<llvm::omp::Directive> OmpStructureChecker::GetCancelType(
    llvm::omp::Directive cancelDir, const parser::CharBlock &cancelSource,
    const std::optional<parser::OmpClauseList> &maybeClauses) {
  if (!maybeClauses) {
    return std::nullopt;
  }
  // Given clauses from CANCEL or CANCELLATION_POINT, identify the construct
  // to which the cancellation applies.
  std::optional<llvm::omp::Directive> cancelee;
  llvm::StringRef cancelName{getDirectiveName(cancelDir)};

  for (const parser::OmpClause &clause : maybeClauses->v) {
    using CancellationConstructType =
        parser::OmpClause::CancellationConstructType;
    if (auto *cctype{std::get_if<CancellationConstructType>(&clause.u)}) {
      if (cancelee) {
        context_.Say(cancelSource,
            "Multiple cancel-directive-name clauses are not allowed on the %s construct"_err_en_US,
            parser::ToUpperCaseLetters(cancelName.str()));
        return std::nullopt;
      }
      cancelee = std::get<parser::OmpDirectiveName>(cctype->v.t).v;
    }
  }

  if (!cancelee) {
    context_.Say(cancelSource,
        "Missing cancel-directive-name clause on the %s construct"_err_en_US,
        parser::ToUpperCaseLetters(cancelName.str()));
    return std::nullopt;
  }

  return cancelee;
}

void OmpStructureChecker::CheckCancellationNest(
    const parser::CharBlock &source, llvm::omp::Directive type) {
  llvm::StringRef typeName{getDirectiveName(type)};

  if (CurrentDirectiveIsNested()) {
    // If construct-type-clause is taskgroup, the cancellation construct must be
    // closely nested inside a task or a taskloop construct and the cancellation
    // region must be closely nested inside a taskgroup region. If
    // construct-type-clause is sections, the cancellation construct must be
    // closely nested inside a sections or section construct. Otherwise, the
    // cancellation construct must be closely nested inside an OpenMP construct
    // that matches the type specified in construct-type-clause of the
    // cancellation construct.
    bool eligibleCancellation{false};

    switch (type) {
    case llvm::omp::Directive::OMPD_taskgroup:
      if (llvm::omp::nestedCancelTaskgroupAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
        if (dirContext_.size() >= 3) {
          // Check if the cancellation region is closely nested inside a
          // taskgroup region when there are more than two levels of directives
          // in the directive context stack.
          if (GetContextParent().directive == llvm::omp::Directive::OMPD_task ||
              FindClauseParent(llvm::omp::Clause::OMPC_nogroup)) {
            for (int i = dirContext_.size() - 3; i >= 0; i--) {
              if (dirContext_[i].directive ==
                  llvm::omp::Directive::OMPD_taskgroup) {
                break;
              }
              if (llvm::omp::nestedCancelParallelAllowedSet.test(
                      dirContext_[i].directive)) {
                eligibleCancellation = false;
                break;
              }
            }
          }
        }
      }
      if (!eligibleCancellation) {
        context_.Say(source,
            "With %s clause, %s construct must be closely nested inside TASK or TASKLOOP construct and %s region must be closely nested inside TASKGROUP region"_err_en_US,
            parser::ToUpperCaseLetters(typeName.str()),
            ContextDirectiveAsFortran(), ContextDirectiveAsFortran());
      }
      return;
    case llvm::omp::Directive::OMPD_sections:
      if (llvm::omp::nestedCancelSectionsAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    case llvm::omp::Directive::OMPD_do:
      if (llvm::omp::nestedCancelDoAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    case llvm::omp::Directive::OMPD_parallel:
      if (llvm::omp::nestedCancelParallelAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    default:
      // This is diagnosed later.
      return;
    }
    if (!eligibleCancellation) {
      context_.Say(source,
          "With %s clause, %s construct cannot be closely nested inside %s construct"_err_en_US,
          parser::ToUpperCaseLetters(typeName.str()),
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(
              getDirectiveName(GetContextParent().directive).str()));
    }
  } else {
    // The cancellation directive cannot be orphaned.
    switch (type) {
    case llvm::omp::Directive::OMPD_taskgroup:
      context_.Say(source,
          "%s %s directive is not closely nested inside TASK or TASKLOOP"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(typeName.str()));
      break;
    case llvm::omp::Directive::OMPD_sections:
      context_.Say(source,
          "%s %s directive is not closely nested inside SECTION or SECTIONS"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(typeName.str()));
      break;
    case llvm::omp::Directive::OMPD_do:
      context_.Say(source,
          "%s %s directive is not closely nested inside the construct that matches the DO clause type"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(typeName.str()));
      break;
    case llvm::omp::Directive::OMPD_parallel:
      context_.Say(source,
          "%s %s directive is not closely nested inside the construct that matches the PARALLEL clause type"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(typeName.str()));
      break;
    default:
      // This is diagnosed later.
      return;
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpEndDirective &x) {
  parser::CharBlock source{x.DirName().source};
  ResetPartialContext(source);
  switch (x.DirId()) {
  case llvm::omp::Directive::OMPD_scope:
    PushContextAndClauseSets(source, llvm::omp::Directive::OMPD_end_scope);
    break;
  // 2.7.3 end-single-clause -> copyprivate-clause |
  //                            nowait-clause
  case llvm::omp::Directive::OMPD_single:
    PushContextAndClauseSets(source, llvm::omp::Directive::OMPD_end_single);
    break;
  // 2.7.4 end-workshare -> END WORKSHARE [nowait-clause]
  case llvm::omp::Directive::OMPD_workshare:
    PushContextAndClauseSets(source, llvm::omp::Directive::OMPD_end_workshare);
    break;
  default:
    // no clauses are allowed
    break;
  }
}

// TODO: Verify the popping of dirContext requirement after nowait
// implementation, as there is an implicit barrier at the end of the worksharing
// constructs unless a nowait clause is specified. Only OMPD_end_single and
// end_workshareare popped as they are pushed while entering the
// EndBlockDirective.
void OmpStructureChecker::Leave(const parser::OmpEndDirective &x) {
  if ((GetContext().directive == llvm::omp::Directive::OMPD_end_scope) ||
      (GetContext().directive == llvm::omp::Directive::OMPD_end_single) ||
      (GetContext().directive == llvm::omp::Directive::OMPD_end_workshare)) {
    dirContext_.pop_back();
  }
}

// Clauses
// Mainly categorized as
// 1. Checks on 'OmpClauseList' from 'parse-tree.h'.
// 2. Checks on clauses which fall under 'struct OmpClause' from parse-tree.h.
// 3. Checks on clauses which are not in 'struct OmpClause' from parse-tree.h.

void OmpStructureChecker::Leave(const parser::OmpClauseList &) {
  // 2.7.1 Loop Construct Restriction
  if (llvm::omp::allDoSet.test(GetContext().directive)) {
    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_schedule)}) {
      // only one schedule clause is allowed
      const auto &schedClause{std::get<parser::OmpClause::Schedule>(clause->u)};
      auto &modifiers{OmpGetModifiers(schedClause.v)};
      auto *ordering{
          OmpGetUniqueModifier<parser::OmpOrderingModifier>(modifiers)};
      if (ordering &&
          ordering->v == parser::OmpOrderingModifier::Value::Nonmonotonic) {
        if (FindClause(llvm::omp::Clause::OMPC_ordered)) {
          context_.Say(clause->source,
              "The NONMONOTONIC modifier cannot be specified "
              "if an ORDERED clause is specified"_err_en_US);
        }
      }
    }

    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_ordered)}) {
      // only one ordered clause is allowed
      const auto &orderedClause{
          std::get<parser::OmpClause::Ordered>(clause->u)};

      if (orderedClause.v) {
        CheckNotAllowedIfClause(
            llvm::omp::Clause::OMPC_ordered, {llvm::omp::Clause::OMPC_linear});

        if (auto *clause2{FindClause(llvm::omp::Clause::OMPC_collapse)}) {
          const auto &collapseClause{
              std::get<parser::OmpClause::Collapse>(clause2->u)};
          // ordered and collapse both have parameters
          if (const auto orderedValue{GetIntValue(orderedClause.v)}) {
            if (const auto collapseValue{GetIntValue(collapseClause.v)}) {
              if (*orderedValue > 0 && *orderedValue < *collapseValue) {
                context_.Say(clause->source,
                    "The parameter of the ORDERED clause must be "
                    "greater than or equal to "
                    "the parameter of the COLLAPSE clause"_err_en_US);
              }
            }
          }
        }
      }

      // TODO: ordered region binding check (requires nesting implementation)
    }
  } // doSet

  // 2.8.1 Simd Construct Restriction
  if (llvm::omp::allSimdSet.test(GetContext().directive)) {
    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_simdlen)}) {
      if (auto *clause2{FindClause(llvm::omp::Clause::OMPC_safelen)}) {
        const auto &simdlenClause{
            std::get<parser::OmpClause::Simdlen>(clause->u)};
        const auto &safelenClause{
            std::get<parser::OmpClause::Safelen>(clause2->u)};
        // simdlen and safelen both have parameters
        if (const auto simdlenValue{GetIntValue(simdlenClause.v)}) {
          if (const auto safelenValue{GetIntValue(safelenClause.v)}) {
            if (*safelenValue > 0 && *simdlenValue > *safelenValue) {
              context_.Say(clause->source,
                  "The parameter of the SIMDLEN clause must be less than or "
                  "equal to the parameter of the SAFELEN clause"_err_en_US);
            }
          }
        }
      }
    }

    // 2.11.5 Simd construct restriction (OpenMP 5.1)
    if (auto *sl_clause{FindClause(llvm::omp::Clause::OMPC_safelen)}) {
      if (auto *o_clause{FindClause(llvm::omp::Clause::OMPC_order)}) {
        const auto &orderClause{
            std::get<parser::OmpClause::Order>(o_clause->u)};
        if (std::get<parser::OmpOrderClause::Ordering>(orderClause.v.t) ==
            parser::OmpOrderClause::Ordering::Concurrent) {
          context_.Say(sl_clause->source,
              "The `SAFELEN` clause cannot appear in the `SIMD` directive "
              "with `ORDER(CONCURRENT)` clause"_err_en_US);
        }
      }
    }
  } // SIMD

  // Semantic checks related to presence of multiple list items within the same
  // clause
  CheckMultListItems();

  if (GetContext().directive == llvm::omp::Directive::OMPD_task) {
    if (auto *detachClause{FindClause(llvm::omp::Clause::OMPC_detach)}) {
      unsigned version{context_.langOptions().OpenMPVersion};
      if (version == 50 || version == 51) {
        // OpenMP 5.0: 2.10.1 Task construct restrictions
        CheckNotAllowedIfClause(llvm::omp::Clause::OMPC_detach,
            {llvm::omp::Clause::OMPC_mergeable});
      } else if (version >= 52) {
        // OpenMP 5.2: 12.5.2 Detach construct restrictions
        if (FindClause(llvm::omp::Clause::OMPC_final)) {
          context_.Say(GetContext().clauseSource,
              "If a DETACH clause appears on a directive, then the encountering task must not be a FINAL task"_err_en_US);
        }

        const auto &detach{
            std::get<parser::OmpClause::Detach>(detachClause->u)};
        if (const auto *name{parser::Unwrap<parser::Name>(detach.v.v)}) {
          Symbol *eventHandleSym{name->symbol};
          auto checkVarAppearsInDataEnvClause = [&](const parser::OmpObjectList
                                                        &objs,
                                                    std::string clause) {
            for (const auto &obj : objs.v) {
              if (const parser::Name *
                  objName{parser::Unwrap<parser::Name>(obj)}) {
                if (&objName->symbol->GetUltimate() == eventHandleSym) {
                  context_.Say(GetContext().clauseSource,
                      "A variable: `%s` that appears in a DETACH clause cannot appear on %s clause on the same construct"_err_en_US,
                      objName->source, clause);
                }
              }
            }
          };
          if (auto *dataEnvClause{
                  FindClause(llvm::omp::Clause::OMPC_private)}) {
            const auto &pClause{
                std::get<parser::OmpClause::Private>(dataEnvClause->u)};
            checkVarAppearsInDataEnvClause(pClause.v, "PRIVATE");
          } else if (auto *dataEnvClause{
                         FindClause(llvm::omp::Clause::OMPC_shared)}) {
            const auto &sClause{
                std::get<parser::OmpClause::Shared>(dataEnvClause->u)};
            checkVarAppearsInDataEnvClause(sClause.v, "SHARED");
          } else if (auto *dataEnvClause{
                         FindClause(llvm::omp::Clause::OMPC_firstprivate)}) {
            const auto &fpClause{
                std::get<parser::OmpClause::Firstprivate>(dataEnvClause->u)};
            checkVarAppearsInDataEnvClause(fpClause.v, "FIRSTPRIVATE");
          } else if (auto *dataEnvClause{
                         FindClause(llvm::omp::Clause::OMPC_in_reduction)}) {
            const auto &irClause{
                std::get<parser::OmpClause::InReduction>(dataEnvClause->u)};
            checkVarAppearsInDataEnvClause(
                std::get<parser::OmpObjectList>(irClause.v.t), "IN_REDUCTION");
          }
        }
      }
    }
  }

  auto testThreadprivateVarErr = [&](Symbol sym, parser::Name name,
                                     llvmOmpClause clauseTy) {
    if (sym.test(Symbol::Flag::OmpThreadprivate))
      context_.Say(name.source,
          "A THREADPRIVATE variable cannot be in %s clause"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseTy).str()));
  };

  // [5.1] 2.21.2 Threadprivate Directive Restriction
  OmpClauseSet threadprivateAllowedSet{llvm::omp::Clause::OMPC_copyin,
      llvm::omp::Clause::OMPC_copyprivate, llvm::omp::Clause::OMPC_schedule,
      llvm::omp::Clause::OMPC_num_threads, llvm::omp::Clause::OMPC_thread_limit,
      llvm::omp::Clause::OMPC_if};
  for (auto it : GetContext().clauseInfo) {
    llvmOmpClause type = it.first;
    const auto *clause = it.second;
    if (!threadprivateAllowedSet.test(type)) {
      if (const auto *objList{GetOmpObjectList(*clause)}) {
        for (const auto &ompObject : objList->v) {
          common::visit(
              common::visitors{
                  [&](const parser::Designator &) {
                    if (const auto *name{
                            parser::Unwrap<parser::Name>(ompObject)}) {
                      if (name->symbol) {
                        testThreadprivateVarErr(
                            name->symbol->GetUltimate(), *name, type);
                      }
                    }
                  },
                  [&](const parser::Name &name) {
                    if (name.symbol) {
                      for (const auto &mem :
                          name.symbol->get<CommonBlockDetails>().objects()) {
                        testThreadprivateVarErr(mem->GetUltimate(), name, type);
                        break;
                      }
                    }
                  },
              },
              ompObject.u);
        }
      }
    }
  }

  CheckRequireAtLeastOneOf();
}

void OmpStructureChecker::Enter(const parser::OmpClause &x) {
  SetContextClause(x);

  llvm::omp::Clause id{x.Id()};
  // The visitors for these clauses do their own checks.
  switch (id) {
  case llvm::omp::Clause::OMPC_copyprivate:
  case llvm::omp::Clause::OMPC_enter:
  case llvm::omp::Clause::OMPC_lastprivate:
  case llvm::omp::Clause::OMPC_reduction:
  case llvm::omp::Clause::OMPC_to:
    return;
  default:
    break;
  }

  // Named constants are OK to be used within 'shared' and 'firstprivate'
  // clauses.  The check for this happens a few lines below.
  bool SharedOrFirstprivate = false;
  switch (id) {
  case llvm::omp::Clause::OMPC_shared:
  case llvm::omp::Clause::OMPC_firstprivate:
    SharedOrFirstprivate = true;
    break;
  default:
    break;
  }

  if (const parser::OmpObjectList *objList{GetOmpObjectList(x)}) {
    AnalyzeObjects(*objList);
    SymbolSourceMap symbols;
    GetSymbolsInObjectList(*objList, symbols);
    for (const auto &[symbol, source] : symbols) {
      if (!IsVariableListItem(*symbol) &&
          !(IsNamedConstant(*symbol) && SharedOrFirstprivate)) {
        deferredNonVariables_.insert({symbol, source});
      }
    }
  }
}

// Following clauses do not have a separate node in parse-tree.h.
CHECK_SIMPLE_CLAUSE(Absent, OMPC_absent)
CHECK_SIMPLE_CLAUSE(Affinity, OMPC_affinity)
CHECK_SIMPLE_CLAUSE(Capture, OMPC_capture)
CHECK_SIMPLE_CLAUSE(Contains, OMPC_contains)
CHECK_SIMPLE_CLAUSE(Default, OMPC_default)
CHECK_SIMPLE_CLAUSE(Depobj, OMPC_depobj)
CHECK_SIMPLE_CLAUSE(DeviceType, OMPC_device_type)
CHECK_SIMPLE_CLAUSE(DistSchedule, OMPC_dist_schedule)
CHECK_SIMPLE_CLAUSE(DynGroupprivate, OMPC_dyn_groupprivate)
CHECK_SIMPLE_CLAUSE(Exclusive, OMPC_exclusive)
CHECK_SIMPLE_CLAUSE(Final, OMPC_final)
CHECK_SIMPLE_CLAUSE(Flush, OMPC_flush)
CHECK_SIMPLE_CLAUSE(Full, OMPC_full)
CHECK_SIMPLE_CLAUSE(Grainsize, OMPC_grainsize)
CHECK_SIMPLE_CLAUSE(Holds, OMPC_holds)
CHECK_SIMPLE_CLAUSE(Inclusive, OMPC_inclusive)
CHECK_SIMPLE_CLAUSE(Initializer, OMPC_initializer)
CHECK_SIMPLE_CLAUSE(Match, OMPC_match)
CHECK_SIMPLE_CLAUSE(Nontemporal, OMPC_nontemporal)
CHECK_SIMPLE_CLAUSE(NumTasks, OMPC_num_tasks)
CHECK_SIMPLE_CLAUSE(Order, OMPC_order)
CHECK_SIMPLE_CLAUSE(Read, OMPC_read)
CHECK_SIMPLE_CLAUSE(Threadprivate, OMPC_threadprivate)
CHECK_SIMPLE_CLAUSE(Threads, OMPC_threads)
CHECK_SIMPLE_CLAUSE(Inbranch, OMPC_inbranch)
CHECK_SIMPLE_CLAUSE(Link, OMPC_link)
CHECK_SIMPLE_CLAUSE(Indirect, OMPC_indirect)
CHECK_SIMPLE_CLAUSE(Mergeable, OMPC_mergeable)
CHECK_SIMPLE_CLAUSE(NoOpenmp, OMPC_no_openmp)
CHECK_SIMPLE_CLAUSE(NoOpenmpRoutines, OMPC_no_openmp_routines)
CHECK_SIMPLE_CLAUSE(NoOpenmpConstructs, OMPC_no_openmp_constructs)
CHECK_SIMPLE_CLAUSE(NoParallelism, OMPC_no_parallelism)
CHECK_SIMPLE_CLAUSE(Nogroup, OMPC_nogroup)
CHECK_SIMPLE_CLAUSE(Notinbranch, OMPC_notinbranch)
CHECK_SIMPLE_CLAUSE(Partial, OMPC_partial)
CHECK_SIMPLE_CLAUSE(ProcBind, OMPC_proc_bind)
CHECK_SIMPLE_CLAUSE(Simd, OMPC_simd)
CHECK_SIMPLE_CLAUSE(Sizes, OMPC_sizes)
CHECK_SIMPLE_CLAUSE(Permutation, OMPC_permutation)
CHECK_SIMPLE_CLAUSE(Uniform, OMPC_uniform)
CHECK_SIMPLE_CLAUSE(Unknown, OMPC_unknown)
CHECK_SIMPLE_CLAUSE(Untied, OMPC_untied)
CHECK_SIMPLE_CLAUSE(UsesAllocators, OMPC_uses_allocators)
CHECK_SIMPLE_CLAUSE(Write, OMPC_write)
CHECK_SIMPLE_CLAUSE(Init, OMPC_init)
CHECK_SIMPLE_CLAUSE(Use, OMPC_use)
CHECK_SIMPLE_CLAUSE(Novariants, OMPC_novariants)
CHECK_SIMPLE_CLAUSE(Nocontext, OMPC_nocontext)
CHECK_SIMPLE_CLAUSE(Severity, OMPC_severity)
CHECK_SIMPLE_CLAUSE(Message, OMPC_message)
CHECK_SIMPLE_CLAUSE(Filter, OMPC_filter)
CHECK_SIMPLE_CLAUSE(Otherwise, OMPC_otherwise)
CHECK_SIMPLE_CLAUSE(AdjustArgs, OMPC_adjust_args)
CHECK_SIMPLE_CLAUSE(AppendArgs, OMPC_append_args)
CHECK_SIMPLE_CLAUSE(MemoryOrder, OMPC_memory_order)
CHECK_SIMPLE_CLAUSE(Bind, OMPC_bind)
CHECK_SIMPLE_CLAUSE(Align, OMPC_align)
CHECK_SIMPLE_CLAUSE(Compare, OMPC_compare)
CHECK_SIMPLE_CLAUSE(OmpxAttribute, OMPC_ompx_attribute)
CHECK_SIMPLE_CLAUSE(Weak, OMPC_weak)
CHECK_SIMPLE_CLAUSE(AcqRel, OMPC_acq_rel)
CHECK_SIMPLE_CLAUSE(Acquire, OMPC_acquire)
CHECK_SIMPLE_CLAUSE(Relaxed, OMPC_relaxed)
CHECK_SIMPLE_CLAUSE(Release, OMPC_release)
CHECK_SIMPLE_CLAUSE(SeqCst, OMPC_seq_cst)
CHECK_SIMPLE_CLAUSE(Fail, OMPC_fail)

CHECK_REQ_SCALAR_INT_CLAUSE(NumTeams, OMPC_num_teams)
CHECK_REQ_SCALAR_INT_CLAUSE(NumThreads, OMPC_num_threads)
CHECK_REQ_SCALAR_INT_CLAUSE(OmpxDynCgroupMem, OMPC_ompx_dyn_cgroup_mem)
CHECK_REQ_SCALAR_INT_CLAUSE(Priority, OMPC_priority)
CHECK_REQ_SCALAR_INT_CLAUSE(ThreadLimit, OMPC_thread_limit)

CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Collapse, OMPC_collapse)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Safelen, OMPC_safelen)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Simdlen, OMPC_simdlen)

// Restrictions specific to each clause are implemented apart from the
// generalized restrictions.

void OmpStructureChecker::Enter(const parser::OmpClause::Destroy &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_destroy);

  llvm::omp::Directive dir{GetContext().directive};
  unsigned version{context_.langOptions().OpenMPVersion};
  if (dir == llvm::omp::Directive::OMPD_depobj) {
    unsigned argSince{52}, noargDeprecatedIn{52};
    if (x.v) {
      if (version < argSince) {
        context_.Say(GetContext().clauseSource,
            "The object parameter in DESTROY clause on DEPOPJ construct is not allowed in %s, %s"_warn_en_US,
            ThisVersion(version), TryVersion(argSince));
      }
    } else {
      if (version >= noargDeprecatedIn) {
        context_.Say(GetContext().clauseSource,
            "The DESTROY clause without argument on DEPOBJ construct is deprecated in %s"_warn_en_US,
            ThisVersion(noargDeprecatedIn));
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Reduction &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_reduction);
  auto &objects{std::get<parser::OmpObjectList>(x.v.t)};

  if (OmpVerifyModifiers(x.v, llvm::omp::OMPC_reduction,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    const auto *ident{
        OmpGetUniqueModifier<parser::OmpReductionIdentifier>(modifiers)};
    assert(ident && "reduction-identifier is a required modifier");
    if (CheckReductionOperator(*ident, OmpGetModifierSource(modifiers, ident),
            llvm::omp::OMPC_reduction)) {
      CheckReductionObjectTypes(objects, *ident);
    }
    using ReductionModifier = parser::OmpReductionModifier;
    if (auto *modifier{OmpGetUniqueModifier<ReductionModifier>(modifiers)}) {
      CheckReductionModifier(*modifier);
    }
  }
  CheckReductionObjects(objects, llvm::omp::Clause::OMPC_reduction);

  // If this is a worksharing construct then ensure the reduction variable
  // is not private in the parallel region that it binds to.
  if (llvm::omp::nestedReduceWorkshareAllowedSet.test(GetContext().directive)) {
    CheckSharedBindingInOuterContext(objects);
  }

  if (GetContext().directive == llvm::omp::Directive::OMPD_loop) {
    for (auto clause : GetContext().clauseInfo) {
      if (const auto *bindClause{
              std::get_if<parser::OmpClause::Bind>(&clause.second->u)}) {
        if (bindClause->v.v == parser::OmpBindClause::Binding::Teams) {
          context_.Say(GetContext().clauseSource,
              "'REDUCTION' clause not allowed with '!$OMP LOOP BIND(TEAMS)'."_err_en_US);
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::InReduction &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_in_reduction);
  auto &objects{std::get<parser::OmpObjectList>(x.v.t)};

  if (OmpVerifyModifiers(x.v, llvm::omp::OMPC_in_reduction,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    const auto *ident{
        OmpGetUniqueModifier<parser::OmpReductionIdentifier>(modifiers)};
    assert(ident && "reduction-identifier is a required modifier");
    if (CheckReductionOperator(*ident, OmpGetModifierSource(modifiers, ident),
            llvm::omp::OMPC_in_reduction)) {
      CheckReductionObjectTypes(objects, *ident);
    }
  }
  CheckReductionObjects(objects, llvm::omp::Clause::OMPC_in_reduction);
}

void OmpStructureChecker::Enter(const parser::OmpClause::TaskReduction &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_task_reduction);
  auto &objects{std::get<parser::OmpObjectList>(x.v.t)};

  if (OmpVerifyModifiers(x.v, llvm::omp::OMPC_task_reduction,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    const auto *ident{
        OmpGetUniqueModifier<parser::OmpReductionIdentifier>(modifiers)};
    assert(ident && "reduction-identifier is a required modifier");
    if (CheckReductionOperator(*ident, OmpGetModifierSource(modifiers, ident),
            llvm::omp::OMPC_task_reduction)) {
      CheckReductionObjectTypes(objects, *ident);
    }
  }
  CheckReductionObjects(objects, llvm::omp::Clause::OMPC_task_reduction);
}

bool OmpStructureChecker::CheckReductionOperator(
    const parser::OmpReductionIdentifier &ident, parser::CharBlock source,
    llvm::omp::Clause clauseId) {
  auto visitOperator{[&](const parser::DefinedOperator &dOpr) {
    if (const auto *intrinsicOp{
            std::get_if<parser::DefinedOperator::IntrinsicOperator>(&dOpr.u)}) {
      switch (*intrinsicOp) {
      case parser::DefinedOperator::IntrinsicOperator::Add:
      case parser::DefinedOperator::IntrinsicOperator::Multiply:
      case parser::DefinedOperator::IntrinsicOperator::AND:
      case parser::DefinedOperator::IntrinsicOperator::OR:
      case parser::DefinedOperator::IntrinsicOperator::EQV:
      case parser::DefinedOperator::IntrinsicOperator::NEQV:
        return true;
      case parser::DefinedOperator::IntrinsicOperator::Subtract:
        context_.Say(GetContext().clauseSource,
            "The minus reduction operator is deprecated since OpenMP 5.2 and is not supported in the REDUCTION clause."_err_en_US,
            ContextDirectiveAsFortran());
        return false;
      default:
        break;
      }
    }
    // User-defined operators are OK if there has been a declared reduction
    // for that. We mangle those names to store the user details.
    if (const auto *definedOp{std::get_if<parser::DefinedOpName>(&dOpr.u)}) {
      std::string mangled{MangleDefinedOperator(definedOp->v.symbol->name())};
      const Scope &scope{definedOp->v.symbol->owner()};
      if (const Symbol *symbol{scope.FindSymbol(mangled)}) {
        if (symbol->detailsIf<UserReductionDetails>()) {
          return true;
        }
      }
    }
    context_.Say(source, "Invalid reduction operator in %s clause."_err_en_US,
        parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
    return false;
  }};

  auto visitDesignator{[&](const parser::ProcedureDesignator &procD) {
    const parser::Name *name{std::get_if<parser::Name>(&procD.u)};
    bool valid{false};
    if (name && name->symbol) {
      const SourceName &realName{name->symbol->GetUltimate().name()};
      valid =
          llvm::is_contained({"max", "min", "iand", "ior", "ieor"}, realName);
      if (!valid) {
        valid = name->symbol->detailsIf<UserReductionDetails>();
      }
    }
    if (!valid) {
      context_.Say(source,
          "Invalid reduction identifier in %s clause."_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
    }
    return valid;
  }};

  return common::visit(
      common::visitors{visitOperator, visitDesignator}, ident.u);
}

/// Check restrictions on objects that are common to all reduction clauses.
void OmpStructureChecker::CheckReductionObjects(
    const parser::OmpObjectList &objects, llvm::omp::Clause clauseId) {
  unsigned version{context_.langOptions().OpenMPVersion};
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objects, symbols);

  // Array sections must be a contiguous storage, have non-zero length.
  for (const parser::OmpObject &object : objects.v) {
    CheckIfContiguous(object);
  }
  CheckReductionArraySection(objects, clauseId);
  // An object must be definable.
  CheckDefinableObjects(symbols, clauseId);
  // Procedure pointers are not allowed.
  CheckProcedurePointer(symbols, clauseId);
  // Pointers must not have INTENT(IN).
  CheckIntentInPointer(symbols, clauseId);

  // Disallow common blocks.
  // Iterate on objects because `GetSymbolsInObjectList` expands common block
  // names into the lists of their members.
  for (const parser::OmpObject &object : objects.v) {
    auto *symbol{GetObjectSymbol(object)};
    if (symbol && IsCommonBlock(*symbol)) {
      auto source{GetObjectSource(object)};
      context_.Say(source ? *source : GetContext().clauseSource,
          "Common block names are not allowed in %s clause"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
    }
  }

  // Denied in all current versions of the standard because structure components
  // are not definable (i.e. they are expressions not variables).
  // Object cannot be a part of another object (except array elements).
  CheckStructureComponent(objects, clauseId);

  if (version >= 50) {
    // If object is an array section or element, the base expression must be
    // a language identifier.
    for (const parser::OmpObject &object : objects.v) {
      if (auto *elem{GetArrayElementFromObj(object)}) {
        const parser::DataRef &base = elem->base;
        if (!std::holds_alternative<parser::Name>(base.u)) {
          auto source{GetObjectSource(object)};
          context_.Say(source ? *source : GetContext().clauseSource,
              "The base expression of an array element or section in %s clause must be an identifier"_err_en_US,
              parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
        }
      }
    }
    // Type parameter inquiries are not allowed.
    for (const parser::OmpObject &object : objects.v) {
      if (auto *dataRef{GetDataRefFromObj(object)}) {
        if (IsDataRefTypeParamInquiry(dataRef)) {
          auto source{GetObjectSource(object)};
          context_.Say(source ? *source : GetContext().clauseSource,
              "Type parameter inquiry is not permitted in %s clause"_err_en_US,
              parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
        }
      }
    }
  }
}

static bool CheckSymbolSupportsType(const Scope &scope,
    const parser::CharBlock &name, const DeclTypeSpec &type) {
  if (const auto *symbol{scope.FindSymbol(name)}) {
    if (const auto *reductionDetails{
            symbol->detailsIf<UserReductionDetails>()}) {
      return reductionDetails->SupportsType(type);
    }
  }
  return false;
}

static bool IsReductionAllowedForType(
    const parser::OmpReductionIdentifier &ident, const DeclTypeSpec &type,
    bool cannotBeBuiltinReduction, const Scope &scope,
    SemanticsContext &context) {
  auto isLogical{[](const DeclTypeSpec &type) -> bool {
    return type.category() == DeclTypeSpec::Logical;
  }};
  auto isCharacter{[](const DeclTypeSpec &type) -> bool {
    return type.category() == DeclTypeSpec::Character;
  }};

  auto checkOperator{[&](const parser::DefinedOperator &dOpr) {
    if (const auto *intrinsicOp{
            std::get_if<parser::DefinedOperator::IntrinsicOperator>(&dOpr.u)}) {
      if (cannotBeBuiltinReduction) {
        return false;
      }

      // OMP5.2: The type [...] of a list item that appears in a
      // reduction clause must be valid for the combiner expression
      // See F2023: Table 10.2
      // .LT., .LE., .GT., .GE. are handled as procedure designators
      // below.
      switch (*intrinsicOp) {
      case parser::DefinedOperator::IntrinsicOperator::Multiply:
      case parser::DefinedOperator::IntrinsicOperator::Add:
      case parser::DefinedOperator::IntrinsicOperator::Subtract:
        if (type.IsNumeric(TypeCategory::Integer) ||
            type.IsNumeric(TypeCategory::Real) ||
            type.IsNumeric(TypeCategory::Complex))
          return true;
        break;

      case parser::DefinedOperator::IntrinsicOperator::AND:
      case parser::DefinedOperator::IntrinsicOperator::OR:
      case parser::DefinedOperator::IntrinsicOperator::EQV:
      case parser::DefinedOperator::IntrinsicOperator::NEQV:
        if (isLogical(type)) {
          return true;
        }
        break;

      // Reduction identifier is not in OMP5.2 Table 5.2
      default:
        DIE("This should have been caught in CheckIntrinsicOperator");
        return false;
      }
      parser::CharBlock name{MakeNameFromOperator(*intrinsicOp, context)};
      return CheckSymbolSupportsType(scope, name, type);
    } else if (const auto *definedOp{
                   std::get_if<parser::DefinedOpName>(&dOpr.u)}) {
      return CheckSymbolSupportsType(
          scope, MangleDefinedOperator(definedOp->v.symbol->name()), type);
    }
    llvm_unreachable(
        "A DefinedOperator is either a DefinedOpName or an IntrinsicOperator");
  }};

  auto checkDesignator{[&](const parser::ProcedureDesignator &procD) {
    const parser::Name *name{std::get_if<parser::Name>(&procD.u)};
    CHECK(name && name->symbol);
    if (name && name->symbol) {
      const SourceName &realName{name->symbol->GetUltimate().name()};
      // OMP5.2: The type [...] of a list item that appears in a
      // reduction clause must be valid for the combiner expression
      if (realName == "iand" || realName == "ior" || realName == "ieor") {
        // IAND: arguments must be integers: F2023 16.9.100
        // IEOR: arguments must be integers: F2023 16.9.106
        // IOR: arguments must be integers: F2023 16.9.111
        if (type.IsNumeric(TypeCategory::Integer) &&
            !cannotBeBuiltinReduction) {
          return true;
        }
      } else if (realName == "max" || realName == "min") {
        // MAX: arguments must be integer, real, or character:
        // F2023 16.9.135
        // MIN: arguments must be integer, real, or character:
        // F2023 16.9.141
        if ((type.IsNumeric(TypeCategory::Integer) ||
                type.IsNumeric(TypeCategory::Real) || isCharacter(type)) &&
            !cannotBeBuiltinReduction) {
          return true;
        }
      }

      // If we get here, it may be a user declared reduction, so check
      // if the symbol has UserReductionDetails, and if so, the type is
      // supported.
      if (const auto *reductionDetails{
              name->symbol->detailsIf<UserReductionDetails>()}) {
        return reductionDetails->SupportsType(type);
      }

      // We also need to check for mangled names (max, min, iand, ieor and ior)
      // and then check if the type is there.
      parser::CharBlock mangledName{MangleSpecialFunctions(name->source)};
      return CheckSymbolSupportsType(scope, mangledName, type);
    }
    // Everything else is "not matching type".
    return false;
  }};

  return common::visit(
      common::visitors{checkOperator, checkDesignator}, ident.u);
}

void OmpStructureChecker::CheckReductionObjectTypes(
    const parser::OmpObjectList &objects,
    const parser::OmpReductionIdentifier &ident) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objects, symbols);

  for (auto &[symbol, source] : symbols) {
    // Built in reductions require types which can be used in their initializer
    // and combiner expressions. For example, for +:
    // r = 0; r = r + r2
    // But it might be valid to use these with DECLARE REDUCTION.
    // Assumed size is already caught elsewhere.
    bool cannotBeBuiltinReduction{IsAssumedRank(*symbol)};
    if (auto *type{symbol->GetType()}) {
      const auto &scope{context_.FindScope(symbol->name())};
      if (!IsReductionAllowedForType(
              ident, *type, cannotBeBuiltinReduction, scope, context_)) {
        context_.Say(source,
            "The type of '%s' is incompatible with the reduction operator."_err_en_US,
            symbol->name());
      }
    } else {
      assert(IsProcedurePointer(*symbol) && "Unexpected symbol properties");
    }
  }
}

void OmpStructureChecker::CheckReductionModifier(
    const parser::OmpReductionModifier &modifier) {
  using ReductionModifier = parser::OmpReductionModifier;
  if (modifier.v == ReductionModifier::Value::Default) {
    // The default one is always ok.
    return;
  }
  const DirectiveContext &dirCtx{GetContext()};
  if (dirCtx.directive == llvm::omp::Directive::OMPD_loop ||
      dirCtx.directive == llvm::omp::Directive::OMPD_taskloop) {
    // [5.2:257:33-34]
    // If a reduction-modifier is specified in a reduction clause that
    // appears on the directive, then the reduction modifier must be
    // default.
    // [5.2:268:16]
    // The reduction-modifier must be default.
    context_.Say(GetContext().clauseSource,
        "REDUCTION modifier on %s directive must be DEFAULT"_err_en_US,
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    return;
  }
  if (modifier.v == ReductionModifier::Value::Task) {
    // "Task" is only allowed on worksharing or "parallel" directive.
    static llvm::omp::Directive worksharing[]{
        llvm::omp::Directive::OMPD_do, llvm::omp::Directive::OMPD_scope,
        llvm::omp::Directive::OMPD_sections,
        // There are more worksharing directives, but they do not apply:
        // "for" is C++ only,
        // "single" and "workshare" don't allow reduction clause,
        // "loop" has different restrictions (checked above).
    };
    if (dirCtx.directive != llvm::omp::Directive::OMPD_parallel &&
        !llvm::is_contained(worksharing, dirCtx.directive)) {
      context_.Say(GetContext().clauseSource,
          "Modifier 'TASK' on REDUCTION clause is only allowed with "
          "PARALLEL or worksharing directive"_err_en_US);
    }
  } else if (modifier.v == ReductionModifier::Value::Inscan) {
    // "Inscan" is only allowed on worksharing-loop, worksharing-loop simd,
    // or "simd" directive.
    // The worksharing-loop directives are OMPD_do and OMPD_for. Only the
    // former is allowed in Fortran.
    if (!llvm::omp::scanParentAllowedSet.test(dirCtx.directive)) {
      context_.Say(GetContext().clauseSource,
          "Modifier 'INSCAN' on REDUCTION clause is only allowed with "
          "WORKSHARING LOOP, WORKSHARING LOOP SIMD, "
          "or SIMD directive"_err_en_US);
    }
  } else {
    // Catch-all for potential future modifiers to make sure that this
    // function is up-to-date.
    context_.Say(GetContext().clauseSource,
        "Unexpected modifier on REDUCTION clause"_err_en_US);
  }
}

void OmpStructureChecker::CheckReductionArraySection(
    const parser::OmpObjectList &ompObjectList, llvm::omp::Clause clauseId) {
  for (const auto &ompObject : ompObjectList.v) {
    if (const auto *dataRef{parser::Unwrap<parser::DataRef>(ompObject)}) {
      if (const auto *arrayElement{
              parser::Unwrap<parser::ArrayElement>(ompObject)}) {
        CheckArraySection(*arrayElement, GetLastName(*dataRef), clauseId);
      }
    }
  }
}

void OmpStructureChecker::CheckSharedBindingInOuterContext(
    const parser::OmpObjectList &redObjectList) {
  //  TODO: Verify the assumption here that the immediately enclosing region is
  //  the parallel region to which the worksharing construct having reduction
  //  binds to.
  if (auto *enclosingContext{GetEnclosingDirContext()}) {
    for (auto it : enclosingContext->clauseInfo) {
      llvmOmpClause type = it.first;
      const auto *clause = it.second;
      if (llvm::omp::privateReductionSet.test(type)) {
        if (const auto *objList{GetOmpObjectList(*clause)}) {
          for (const auto &ompObject : objList->v) {
            if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
              if (const auto *symbol{name->symbol}) {
                for (const auto &redOmpObject : redObjectList.v) {
                  if (const auto *rname{
                          parser::Unwrap<parser::Name>(redOmpObject)}) {
                    if (const auto *rsymbol{rname->symbol}) {
                      if (rsymbol->name() == symbol->name()) {
                        context_.Say(GetContext().clauseSource,
                            "%s variable '%s' is %s in outer context must"
                            " be shared in the parallel regions to which any"
                            " of the worksharing regions arising from the "
                            "worksharing construct bind."_err_en_US,
                            parser::ToUpperCaseLetters(
                                getClauseName(llvm::omp::Clause::OMPC_reduction)
                                    .str()),
                            symbol->name(),
                            parser::ToUpperCaseLetters(
                                getClauseName(type).str()));
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Ordered &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_ordered);
  // the parameter of ordered clause is optional
  if (const auto &expr{x.v}) {
    RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_ordered, *expr);
    // 2.8.3 Loop SIMD Construct Restriction
    if (llvm::omp::allDoSimdSet.test(GetContext().directive)) {
      context_.Say(GetContext().clauseSource,
          "No ORDERED clause with a parameter can be specified "
          "on the %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Shared &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_shared);
  CheckVarIsNotPartOfAnotherVar(GetContext().clauseSource, x.v, "SHARED");
  CheckCrayPointee(x.v, "SHARED");
}
void OmpStructureChecker::Enter(const parser::OmpClause::Private &x) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(x.v, symbols);
  CheckAllowedClause(llvm::omp::Clause::OMPC_private);
  CheckVarIsNotPartOfAnotherVar(GetContext().clauseSource, x.v, "PRIVATE");
  CheckIntentInPointer(symbols, llvm::omp::Clause::OMPC_private);
  CheckCrayPointee(x.v, "PRIVATE");
}

void OmpStructureChecker::Enter(const parser::OmpClause::Nowait &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_nowait);
}

bool OmpStructureChecker::IsDataRefTypeParamInquiry(
    const parser::DataRef *dataRef) {
  bool dataRefIsTypeParamInquiry{false};
  if (const auto *structComp{
          parser::Unwrap<parser::StructureComponent>(dataRef)}) {
    if (const auto *compSymbol{structComp->component.symbol}) {
      if (const auto *compSymbolMiscDetails{
              std::get_if<MiscDetails>(&compSymbol->details())}) {
        const auto detailsKind = compSymbolMiscDetails->kind();
        dataRefIsTypeParamInquiry =
            (detailsKind == MiscDetails::Kind::KindParamInquiry ||
                detailsKind == MiscDetails::Kind::LenParamInquiry);
      } else if (compSymbol->has<TypeParamDetails>()) {
        dataRefIsTypeParamInquiry = true;
      }
    }
  }
  return dataRefIsTypeParamInquiry;
}

void OmpStructureChecker::CheckVarIsNotPartOfAnotherVar(
    const parser::CharBlock &source, const parser::OmpObjectList &objList,
    llvm::StringRef clause) {
  for (const auto &ompObject : objList.v) {
    CheckVarIsNotPartOfAnotherVar(source, ompObject, clause);
  }
}

void OmpStructureChecker::CheckVarIsNotPartOfAnotherVar(
    const parser::CharBlock &source, const parser::OmpObject &ompObject,
    llvm::StringRef clause) {
  common::visit(
      common::visitors{
          [&](const parser::Designator &designator) {
            if (const auto *dataRef{
                    std::get_if<parser::DataRef>(&designator.u)}) {
              if (IsDataRefTypeParamInquiry(dataRef)) {
                context_.Say(source,
                    "A type parameter inquiry cannot appear on the %s directive"_err_en_US,
                    ContextDirectiveAsFortran());
              } else if (parser::Unwrap<parser::StructureComponent>(
                             ompObject) ||
                  parser::Unwrap<parser::ArrayElement>(ompObject)) {
                if (llvm::omp::nonPartialVarSet.test(GetContext().directive)) {
                  context_.Say(source,
                      "A variable that is part of another variable (as an array or structure element) cannot appear on the %s directive"_err_en_US,
                      ContextDirectiveAsFortran());
                } else {
                  context_.Say(source,
                      "A variable that is part of another variable (as an array or structure element) cannot appear in a %s clause"_err_en_US,
                      clause.data());
                }
              }
            }
          },
          [&](const parser::Name &name) {},
      },
      ompObject.u);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Firstprivate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_firstprivate);

  CheckVarIsNotPartOfAnotherVar(GetContext().clauseSource, x.v, "FIRSTPRIVATE");
  CheckCrayPointee(x.v, "FIRSTPRIVATE");
  CheckIsLoopIvPartOfClause(llvmOmpClause::OMPC_firstprivate, x.v);

  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  CheckCopyingPolymorphicAllocatable(
      currSymbols, llvm::omp::Clause::OMPC_firstprivate);

  DirectivesClauseTriple dirClauseTriple;
  // Check firstprivate variables in worksharing constructs
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_do,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_sections,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_single,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));
  // Check firstprivate variables in distribute construct
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_distribute,
      std::make_pair(
          llvm::omp::Directive::OMPD_teams, llvm::omp::privateReductionSet));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_distribute,
      std::make_pair(llvm::omp::Directive::OMPD_target_teams,
          llvm::omp::privateReductionSet));
  // Check firstprivate variables in task and taskloop constructs
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_task,
      std::make_pair(llvm::omp::Directive::OMPD_parallel,
          OmpClauseSet{llvm::omp::Clause::OMPC_reduction}));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_taskloop,
      std::make_pair(llvm::omp::Directive::OMPD_parallel,
          OmpClauseSet{llvm::omp::Clause::OMPC_reduction}));

  CheckPrivateSymbolsInOuterCxt(
      currSymbols, dirClauseTriple, llvm::omp::Clause::OMPC_firstprivate);
}

void OmpStructureChecker::CheckIsLoopIvPartOfClause(
    llvmOmpClause clause, const parser::OmpObjectList &ompObjectList) {
  for (const auto &ompObject : ompObjectList.v) {
    if (const parser::Name *name{parser::Unwrap<parser::Name>(ompObject)}) {
      if (name->symbol == GetContext().loopIV) {
        context_.Say(name->source,
            "DO iteration variable %s is not allowed in %s clause."_err_en_US,
            name->ToString(),
            parser::ToUpperCaseLetters(getClauseName(clause).str()));
      }
    }
  }
}

// Restrictions specific to each clause are implemented apart from the
// generalized restrictions.
void OmpStructureChecker::Enter(const parser::OmpClause::Aligned &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_aligned);
  if (OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_aligned, GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    if (auto *align{OmpGetUniqueModifier<parser::OmpAlignment>(modifiers)}) {
      if (const auto &v{GetIntValue(align->v)}; !v || *v <= 0) {
        context_.Say(OmpGetModifierSource(modifiers, align),
            "The alignment value should be a constant positive integer"_err_en_US);
      }
    }
  }
  // 2.8.1 TODO: list-item attribute check
}

void OmpStructureChecker::Enter(const parser::OmpClause::Defaultmap &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_defaultmap);
  unsigned version{context_.langOptions().OpenMPVersion};
  using ImplicitBehavior = parser::OmpDefaultmapClause::ImplicitBehavior;
  auto behavior{std::get<ImplicitBehavior>(x.v.t)};
  if (version <= 45) {
    if (behavior != ImplicitBehavior::Tofrom) {
      context_.Say(GetContext().clauseSource,
          "%s is not allowed in %s, %s"_warn_en_US,
          parser::ToUpperCaseLetters(
              parser::OmpDefaultmapClause::EnumToString(behavior)),
          ThisVersion(version), TryVersion(50));
    }
  }
  if (!OmpVerifyModifiers(x.v, llvm::omp::OMPC_defaultmap,
          GetContext().clauseSource, context_)) {
    // If modifier verification fails, return early.
    return;
  }
  auto &modifiers{OmpGetModifiers(x.v)};
  auto *maybeCategory{
      OmpGetUniqueModifier<parser::OmpVariableCategory>(modifiers)};
  if (maybeCategory) {
    using VariableCategory = parser::OmpVariableCategory;
    VariableCategory::Value category{maybeCategory->v};
    unsigned tryVersion{0};
    if (version <= 45 && category != VariableCategory::Value::Scalar) {
      tryVersion = 50;
    }
    if (version < 52 && category == VariableCategory::Value::All) {
      tryVersion = 52;
    }
    if (tryVersion) {
      context_.Say(GetContext().clauseSource,
          "%s is not allowed in %s, %s"_warn_en_US,
          parser::ToUpperCaseLetters(VariableCategory::EnumToString(category)),
          ThisVersion(version), TryVersion(tryVersion));
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::If &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_if);
  unsigned version{context_.langOptions().OpenMPVersion};
  llvm::omp::Directive dir{GetContext().directive};

  auto isConstituent{[](llvm::omp::Directive dir, llvm::omp::Directive part) {
    using namespace llvm::omp;
    llvm::ArrayRef<Directive> dirLeafs{getLeafConstructsOrSelf(dir)};
    llvm::ArrayRef<Directive> partLeafs{getLeafConstructsOrSelf(part)};
    // Maybe it's sufficient to check if every leaf of `part` is also a leaf
    // of `dir`, but to be safe check if `partLeafs` is a sub-sequence of
    // `dirLeafs`.
    size_t dirSize{dirLeafs.size()}, partSize{partLeafs.size()};
    // Find the first leaf from `part` in `dir`.
    if (auto first = llvm::find(dirLeafs, partLeafs.front());
        first != dirLeafs.end()) {
      // A leaf can only appear once in a compound directive, so if `part`
      // is a subsequence of `dir`, it must start here.
      size_t firstPos{
          static_cast<size_t>(std::distance(dirLeafs.begin(), first))};
      llvm::ArrayRef<Directive> subSeq{
          first, std::min<size_t>(dirSize - firstPos, partSize)};
      return subSeq == partLeafs;
    }
    return false;
  }};

  if (OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_if, GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    if (auto *dnm{OmpGetUniqueModifier<parser::OmpDirectiveNameModifier>(
            modifiers)}) {
      llvm::omp::Directive sub{dnm->v};
      std::string subName{
          parser::ToUpperCaseLetters(getDirectiveName(sub).str())};
      std::string dirName{
          parser::ToUpperCaseLetters(getDirectiveName(dir).str())};

      parser::CharBlock modifierSource{OmpGetModifierSource(modifiers, dnm)};
      auto desc{OmpGetDescriptor<parser::OmpDirectiveNameModifier>()};
      std::string modName{desc.name.str()};

      if (!isConstituent(dir, sub)) {
        context_
            .Say(modifierSource,
                "%s is not a constituent of the %s directive"_err_en_US,
                subName, dirName)
            .Attach(GetContext().directiveSource,
                "Cannot apply to directive"_en_US);
      } else {
        static llvm::omp::Directive valid45[]{
            llvm::omp::OMPD_cancel, //
            llvm::omp::OMPD_parallel, //
            /* OMP 5.0+ also allows OMPD_simd */
            llvm::omp::OMPD_target, //
            llvm::omp::OMPD_target_data, //
            llvm::omp::OMPD_target_enter_data, //
            llvm::omp::OMPD_target_exit_data, //
            llvm::omp::OMPD_target_update, //
            llvm::omp::OMPD_task, //
            llvm::omp::OMPD_taskloop, //
            /* OMP 5.2+ also allows OMPD_teams */
        };
        if (version < 50 && sub == llvm::omp::OMPD_simd) {
          context_.Say(modifierSource,
              "%s is not allowed as '%s' in %s, %s"_warn_en_US, subName,
              modName, ThisVersion(version), TryVersion(50));
        } else if (version < 52 && sub == llvm::omp::OMPD_teams) {
          context_.Say(modifierSource,
              "%s is not allowed as '%s' in %s, %s"_warn_en_US, subName,
              modName, ThisVersion(version), TryVersion(52));
        } else if (!llvm::is_contained(valid45, sub) &&
            sub != llvm::omp::OMPD_simd && sub != llvm::omp::OMPD_teams) {
          context_.Say(modifierSource,
              "%s is not allowed as '%s' in %s"_err_en_US, subName, modName,
              ThisVersion(version));
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Detach &x) {
  unsigned version{context_.langOptions().OpenMPVersion};
  if (version >= 52) {
    SetContextClauseInfo(llvm::omp::Clause::OMPC_detach);
  } else {
    // OpenMP 5.0: 2.10.1 Task construct restrictions
    CheckAllowedClause(llvm::omp::Clause::OMPC_detach);
  }
  // OpenMP 5.2: 12.5.2 Detach clause restrictions
  if (version >= 52) {
    CheckVarIsNotPartOfAnotherVar(GetContext().clauseSource, x.v.v, "DETACH");
  }

  if (const auto *name{parser::Unwrap<parser::Name>(x.v.v)}) {
    if (version >= 52 && IsPointer(*name->symbol)) {
      context_.Say(GetContext().clauseSource,
          "The event-handle: `%s` must not have the POINTER attribute"_err_en_US,
          name->ToString());
    }
    if (!name->symbol->GetType()->IsNumeric(TypeCategory::Integer)) {
      context_.Say(GetContext().clauseSource,
          "The event-handle: `%s` must be of type integer(kind=omp_event_handle_kind)"_err_en_US,
          name->ToString());
    }
  }
}

void OmpStructureChecker::CheckAllowedMapTypes(parser::OmpMapType::Value type,
    llvm::ArrayRef<parser::OmpMapType::Value> allowed) {
  if (llvm::is_contained(allowed, type)) {
    return;
  }

  llvm::SmallVector<std::string> names;
  llvm::transform(
      allowed, std::back_inserter(names), [](parser::OmpMapType::Value val) {
        return parser::ToUpperCaseLetters(
            parser::OmpMapType::EnumToString(val));
      });
  llvm::sort(names);
  context_.Say(GetContext().clauseSource,
      "Only the %s map types are permitted for MAP clauses on the %s directive"_err_en_US,
      llvm::join(names, ", "), ContextDirectiveAsFortran());
}

void OmpStructureChecker::Enter(const parser::OmpClause::Map &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_map);
  if (!OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_map, GetContext().clauseSource, context_)) {
    return;
  }

  auto &modifiers{OmpGetModifiers(x.v)};
  unsigned version{context_.langOptions().OpenMPVersion};
  if (auto commas{std::get<bool>(x.v.t)}; !commas && version >= 52) {
    context_.Say(GetContext().clauseSource,
        "The specification of modifiers without comma separators for the "
        "'MAP' clause has been deprecated in OpenMP 5.2"_port_en_US);
  }
  if (auto *iter{OmpGetUniqueModifier<parser::OmpIterator>(modifiers)}) {
    CheckIteratorModifier(*iter);
  }
  if (auto *type{OmpGetUniqueModifier<parser::OmpMapType>(modifiers)}) {
    using Directive = llvm::omp::Directive;
    using Value = parser::OmpMapType::Value;

    static auto isValidForVersion{
        [](parser::OmpMapType::Value t, unsigned version) {
          switch (t) {
          case parser::OmpMapType::Value::Alloc:
          case parser::OmpMapType::Value::Delete:
          case parser::OmpMapType::Value::Release:
            return version < 60;
          case parser::OmpMapType::Value::Storage:
            return version >= 60;
          default:
            return true;
          }
        }};

    llvm::SmallVector<parser::OmpMapType::Value> mapEnteringTypes{[&]() {
      llvm::SmallVector<parser::OmpMapType::Value> result;
      for (size_t i{0}; i != parser::OmpMapType::Value_enumSize; ++i) {
        auto t{static_cast<parser::OmpMapType::Value>(i)};
        if (isValidForVersion(t, version) && IsMapEnteringType(t)) {
          result.push_back(t);
        }
      }
      return result;
    }()};
    llvm::SmallVector<parser::OmpMapType::Value> mapExitingTypes{[&]() {
      llvm::SmallVector<parser::OmpMapType::Value> result;
      for (size_t i{0}; i != parser::OmpMapType::Value_enumSize; ++i) {
        auto t{static_cast<parser::OmpMapType::Value>(i)};
        if (isValidForVersion(t, version) && IsMapExitingType(t)) {
          result.push_back(t);
        }
      }
      return result;
    }()};

    llvm::omp::Directive dir{GetContext().directive};
    llvm::ArrayRef<llvm::omp::Directive> leafs{
        llvm::omp::getLeafConstructsOrSelf(dir)};

    if (llvm::is_contained(leafs, Directive::OMPD_target) ||
        llvm::is_contained(leafs, Directive::OMPD_target_data)) {
      if (version >= 60) {
        // Map types listed in the decay table. [6.0:276]
        CheckAllowedMapTypes(
            type->v, {Value::Storage, Value::From, Value::To, Value::Tofrom});
      } else {
        CheckAllowedMapTypes(
            type->v, {Value::Alloc, Value::From, Value::To, Value::Tofrom});
      }
    } else if (llvm::is_contained(leafs, Directive::OMPD_target_enter_data)) {
      CheckAllowedMapTypes(type->v, mapEnteringTypes);
    } else if (llvm::is_contained(leafs, Directive::OMPD_target_exit_data)) {
      CheckAllowedMapTypes(type->v, mapExitingTypes);
    }
  }

  auto &&typeMods{
      OmpGetRepeatableModifier<parser::OmpMapTypeModifier>(modifiers)};
  struct Less {
    using Iterator = decltype(typeMods.begin());
    bool operator()(Iterator a, Iterator b) const {
      const parser::OmpMapTypeModifier *pa = *a;
      const parser::OmpMapTypeModifier *pb = *b;
      return pa->v < pb->v;
    }
  };
  if (auto maybeIter{FindDuplicate<Less>(typeMods)}) {
    context_.Say(GetContext().clauseSource,
        "Duplicate map-type-modifier entry '%s' will be ignored"_warn_en_US,
        parser::ToUpperCaseLetters(
            parser::OmpMapTypeModifier::EnumToString((**maybeIter)->v)));
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Schedule &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_schedule);
  const parser::OmpScheduleClause &scheduleClause = x.v;
  if (!OmpVerifyModifiers(scheduleClause, llvm::omp::OMPC_schedule,
          GetContext().clauseSource, context_)) {
    return;
  }

  // 2.7 Loop Construct Restriction
  if (llvm::omp::allDoSet.test(GetContext().directive)) {
    auto &modifiers{OmpGetModifiers(scheduleClause)};
    auto kind{std::get<parser::OmpScheduleClause::Kind>(scheduleClause.t)};
    auto &chunk{
        std::get<std::optional<parser::ScalarIntExpr>>(scheduleClause.t)};
    if (chunk) {
      if (kind == parser::OmpScheduleClause::Kind::Runtime ||
          kind == parser::OmpScheduleClause::Kind::Auto) {
        context_.Say(GetContext().clauseSource,
            "When SCHEDULE clause has %s specified, "
            "it must not have chunk size specified"_err_en_US,
            parser::ToUpperCaseLetters(
                parser::OmpScheduleClause::EnumToString(kind)));
      }
      if (const auto &chunkExpr{std::get<std::optional<parser::ScalarIntExpr>>(
              scheduleClause.t)}) {
        RequiresPositiveParameter(
            llvm::omp::Clause::OMPC_schedule, *chunkExpr, "chunk size");
      }
    }

    auto *ordering{
        OmpGetUniqueModifier<parser::OmpOrderingModifier>(modifiers)};
    if (ordering &&
        ordering->v == parser::OmpOrderingModifier::Value::Nonmonotonic) {
      if (kind != parser::OmpScheduleClause::Kind::Dynamic &&
          kind != parser::OmpScheduleClause::Kind::Guided) {
        context_.Say(GetContext().clauseSource,
            "The NONMONOTONIC modifier can only be specified with "
            "SCHEDULE(DYNAMIC) or SCHEDULE(GUIDED)"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Device &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_device);
  const parser::OmpDeviceClause &deviceClause{x.v};
  const auto &device{std::get<parser::ScalarIntExpr>(deviceClause.t)};
  RequiresPositiveParameter(
      llvm::omp::Clause::OMPC_device, device, "device expression");
  llvm::omp::Directive dir{GetContext().directive};

  if (OmpVerifyModifiers(deviceClause, llvm::omp::OMPC_device,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(deviceClause)};

    if (auto *deviceMod{
            OmpGetUniqueModifier<parser::OmpDeviceModifier>(modifiers)}) {
      using Value = parser::OmpDeviceModifier::Value;
      if (dir != llvm::omp::OMPD_target && deviceMod->v == Value::Ancestor) {
        auto name{OmpGetDescriptor<parser::OmpDeviceModifier>().name};
        context_.Say(OmpGetModifierSource(modifiers, deviceMod),
            "The ANCESTOR %s must not appear on the DEVICE clause on any directive other than the TARGET construct. Found on %s construct."_err_en_US,
            name.str(), parser::ToUpperCaseLetters(getDirectiveName(dir)));
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Depend &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_depend);
  llvm::omp::Directive dir{GetContext().directive};
  unsigned version{context_.langOptions().OpenMPVersion};

  auto *doaDep{std::get_if<parser::OmpDoacross>(&x.v.u)};
  auto *taskDep{std::get_if<parser::OmpDependClause::TaskDep>(&x.v.u)};
  assert(((doaDep == nullptr) != (taskDep == nullptr)) &&
      "Unexpected alternative in update clause");

  if (doaDep) {
    CheckDoacross(*doaDep);
    CheckDependenceType(doaDep->GetDepType());
  } else {
    using Modifier = parser::OmpDependClause::TaskDep::Modifier;
    auto &modifiers{std::get<std::optional<std::list<Modifier>>>(taskDep->t)};
    if (!modifiers) {
      context_.Say(GetContext().clauseSource,
          "A DEPEND clause on a TASK construct must have a valid task dependence type"_err_en_US);
      return;
    }
    CheckTaskDependenceType(taskDep->GetTaskDepType());
  }

  if (dir == llvm::omp::OMPD_depobj) {
    // [5.0:255:11], [5.1:288:3]
    // A depend clause on a depobj construct must not have source, sink [or
    // depobj](5.0) as dependence-type.
    if (version >= 50) {
      bool invalidDep{false};
      if (taskDep) {
        if (version == 50) {
          invalidDep = taskDep->GetTaskDepType() ==
              parser::OmpTaskDependenceType::Value::Depobj;
        }
      } else {
        invalidDep = true;
      }
      if (invalidDep) {
        context_.Say(GetContext().clauseSource,
            "A DEPEND clause on a DEPOBJ construct must not have %s as dependence type"_err_en_US,
            version == 50 ? "SINK, SOURCE or DEPOBJ" : "SINK or SOURCE");
      }
    }
  } else if (dir != llvm::omp::OMPD_ordered) {
    if (doaDep) {
      context_.Say(GetContext().clauseSource,
          "The SINK and SOURCE dependence types can only be used with the ORDERED directive, used here in the %s construct"_err_en_US,
          parser::ToUpperCaseLetters(getDirectiveName(dir)));
    }
  }
  if (taskDep) {
    auto &objList{std::get<parser::OmpObjectList>(taskDep->t)};
    if (dir == llvm::omp::OMPD_depobj) {
      // [5.0:255:13], [5.1:288:6], [5.2:322:26]
      // A depend clause on a depobj construct must only specify one locator.
      if (objList.v.size() != 1) {
        context_.Say(GetContext().clauseSource,
            "A DEPEND clause on a DEPOBJ construct must only specify "
            "one locator"_err_en_US);
      }
    }
    for (const auto &object : objList.v) {
      if (const auto *name{std::get_if<parser::Name>(&object.u)}) {
        context_.Say(GetContext().clauseSource,
            "Common block name ('%s') cannot appear in a DEPEND "
            "clause"_err_en_US,
            name->ToString());
      } else if (auto *designator{std::get_if<parser::Designator>(&object.u)}) {
        if (auto *dataRef{std::get_if<parser::DataRef>(&designator->u)}) {
          CheckDependList(*dataRef);
          if (const auto *arr{
                  std::get_if<common::Indirection<parser::ArrayElement>>(
                      &dataRef->u)}) {
            CheckArraySection(arr->value(), GetLastName(*dataRef),
                llvm::omp::Clause::OMPC_depend);
          }
        }
      }
    }
    if (OmpVerifyModifiers(*taskDep, llvm::omp::OMPC_depend,
            GetContext().clauseSource, context_)) {
      auto &modifiers{OmpGetModifiers(*taskDep)};
      if (OmpGetUniqueModifier<parser::OmpIterator>(modifiers)) {
        if (dir == llvm::omp::OMPD_depobj) {
          context_.Say(GetContext().clauseSource,
              "An iterator-modifier may specify multiple locators, a DEPEND clause on a DEPOBJ construct must only specify one locator"_warn_en_US);
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Doacross &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_doacross);
  CheckDoacross(x.v.v);
}

void OmpStructureChecker::CheckDoacross(const parser::OmpDoacross &doa) {
  if (std::holds_alternative<parser::OmpDoacross::Source>(doa.u)) {
    // Nothing to check here.
    return;
  }

  // Process SINK dependence type. SINK may only appear in an ORDER construct,
  // which references a prior ORDERED(n) clause on a DO or SIMD construct
  // that marks the top of the loop nest.

  auto &sink{std::get<parser::OmpDoacross::Sink>(doa.u)};
  const std::list<parser::OmpIteration> &vec{sink.v.v};

  // Check if the variables in the iteration vector are unique.
  struct Less {
    using Iterator = std::list<parser::OmpIteration>::const_iterator;
    bool operator()(Iterator a, Iterator b) const {
      auto namea{std::get<parser::Name>(a->t)};
      auto nameb{std::get<parser::Name>(b->t)};
      assert(namea.symbol && nameb.symbol && "Unresolved symbols");
      // The non-determinism of the "<" doesn't matter, we only care about
      // equality, i.e.  a == b  <=>  !(a < b) && !(b < a)
      return reinterpret_cast<uintptr_t>(namea.symbol) <
          reinterpret_cast<uintptr_t>(nameb.symbol);
    }
  };
  if (auto maybeIter{FindDuplicate<Less>(vec)}) {
    auto name{std::get<parser::Name>((*maybeIter)->t)};
    context_.Say(name.source,
        "Duplicate variable '%s' in the iteration vector"_err_en_US,
        name.ToString());
  }

  // Check if the variables in the iteration vector are induction variables.
  // Ignore any mismatch between the size of the iteration vector and the
  // number of DO constructs on the stack. This is checked elsewhere.

  auto GetLoopDirective{[](const parser::OpenMPLoopConstruct &x) {
    auto &begin{std::get<parser::OmpBeginLoopDirective>(x.t)};
    return std::get<parser::OmpLoopDirective>(begin.t).v;
  }};
  auto GetLoopClauses{[](const parser::OpenMPLoopConstruct &x)
                          -> const std::list<parser::OmpClause> & {
    auto &begin{std::get<parser::OmpBeginLoopDirective>(x.t)};
    return std::get<parser::OmpClauseList>(begin.t).v;
  }};

  std::set<const Symbol *> inductionVars;
  for (const LoopConstruct &loop : llvm::reverse(loopStack_)) {
    if (auto *doc{std::get_if<const parser::DoConstruct *>(&loop)}) {
      // Do-construct, collect the induction variable.
      if (auto &control{(*doc)->GetLoopControl()}) {
        if (auto *b{std::get_if<parser::LoopControl::Bounds>(&control->u)}) {
          inductionVars.insert(b->name.thing.symbol);
        }
      }
    } else {
      // Omp-loop-construct, check if it's do/simd with an ORDERED clause.
      auto *loopc{std::get_if<const parser::OpenMPLoopConstruct *>(&loop)};
      assert(loopc && "Expecting OpenMPLoopConstruct");
      llvm::omp::Directive loopDir{GetLoopDirective(**loopc)};
      if (loopDir == llvm::omp::OMPD_do || loopDir == llvm::omp::OMPD_simd) {
        auto IsOrdered{[](const parser::OmpClause &c) {
          return c.Id() == llvm::omp::OMPC_ordered;
        }};
        // If it has ORDERED clause, stop the traversal.
        if (llvm::any_of(GetLoopClauses(**loopc), IsOrdered)) {
          break;
        }
      }
    }
  }
  for (const parser::OmpIteration &iter : vec) {
    auto &name{std::get<parser::Name>(iter.t)};
    if (!inductionVars.count(name.symbol)) {
      context_.Say(name.source,
          "The iteration vector element '%s' is not an induction variable within the ORDERED loop nest"_err_en_US,
          name.ToString());
    }
  }
}

void OmpStructureChecker::CheckCopyingPolymorphicAllocatable(
    SymbolSourceMap &symbols, const llvm::omp::Clause clause) {
  if (context_.ShouldWarn(common::UsageWarning::Portability)) {
    for (auto &[symbol, source] : symbols) {
      if (IsPolymorphicAllocatable(*symbol)) {
        context_.Warn(common::UsageWarning::Portability, source,
            "If a polymorphic variable with allocatable attribute '%s' is in %s clause, the behavior is unspecified"_port_en_US,
            symbol->name(),
            parser::ToUpperCaseLetters(getClauseName(clause).str()));
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Copyprivate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_copyprivate);
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(x.v, symbols);
  CheckVariableListItem(symbols);
  CheckIntentInPointer(symbols, llvm::omp::Clause::OMPC_copyprivate);
  CheckCopyingPolymorphicAllocatable(
      symbols, llvm::omp::Clause::OMPC_copyprivate);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Lastprivate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_lastprivate);

  const auto &objectList{std::get<parser::OmpObjectList>(x.v.t)};
  CheckVarIsNotPartOfAnotherVar(
      GetContext().clauseSource, objectList, "LASTPRIVATE");
  CheckCrayPointee(objectList, "LASTPRIVATE");

  DirectivesClauseTriple dirClauseTriple;
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(objectList, currSymbols);
  CheckDefinableObjects(currSymbols, llvm::omp::Clause::OMPC_lastprivate);
  CheckCopyingPolymorphicAllocatable(
      currSymbols, llvm::omp::Clause::OMPC_lastprivate);

  // Check lastprivate variables in worksharing constructs
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_do,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_sections,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));

  CheckPrivateSymbolsInOuterCxt(
      currSymbols, dirClauseTriple, llvm::omp::Clause::OMPC_lastprivate);

  if (OmpVerifyModifiers(x.v, llvm::omp::OMPC_lastprivate,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    using LastprivateModifier = parser::OmpLastprivateModifier;
    if (auto *modifier{OmpGetUniqueModifier<LastprivateModifier>(modifiers)}) {
      CheckLastprivateModifier(*modifier);
    }
  }
}

// Add any restrictions related to Modifiers/Directives with
// Lastprivate clause here:
void OmpStructureChecker::CheckLastprivateModifier(
    const parser::OmpLastprivateModifier &modifier) {
  using LastprivateModifier = parser::OmpLastprivateModifier;
  const DirectiveContext &dirCtx{GetContext()};
  if (modifier.v == LastprivateModifier::Value::Conditional &&
      dirCtx.directive == llvm::omp::Directive::OMPD_taskloop) {
    // [5.2:268:17]
    // The conditional lastprivate-modifier must not be specified.
    context_.Say(GetContext().clauseSource,
        "'CONDITIONAL' modifier on lastprivate clause with TASKLOOP "
        "directive is not allowed"_err_en_US);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Copyin &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_copyin);

  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  CheckCopyingPolymorphicAllocatable(
      currSymbols, llvm::omp::Clause::OMPC_copyin);
}

void OmpStructureChecker::CheckStructureComponent(
    const parser::OmpObjectList &objects, llvm::omp::Clause clauseId) {
  auto CheckComponent{[&](const parser::Designator &designator) {
    if (auto *dataRef{std::get_if<parser::DataRef>(&designator.u)}) {
      if (!IsDataRefTypeParamInquiry(dataRef)) {
        if (auto *comp{parser::Unwrap<parser::StructureComponent>(*dataRef)}) {
          context_.Say(comp->component.source,
              "A variable that is part of another variable cannot appear on the %s clause"_err_en_US,
              parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
        }
      }
    }
  }};

  for (const auto &object : objects.v) {
    common::visit(
        common::visitors{
            CheckComponent,
            [&](const parser::Name &name) {},
        },
        object.u);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Update &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_update);
  llvm::omp::Directive dir{GetContext().directive};
  unsigned version{context_.langOptions().OpenMPVersion};

  const parser::OmpDependenceType *depType{nullptr};
  const parser::OmpTaskDependenceType *taskType{nullptr};
  if (auto &maybeUpdate{x.v}) {
    depType = std::get_if<parser::OmpDependenceType>(&maybeUpdate->u);
    taskType = std::get_if<parser::OmpTaskDependenceType>(&maybeUpdate->u);
  }

  if (!depType && !taskType) {
    assert(dir == llvm::omp::Directive::OMPD_atomic &&
        "Unexpected alternative in update clause");
    return;
  }

  if (depType) {
    CheckDependenceType(depType->v);
  } else if (taskType) {
    CheckTaskDependenceType(taskType->v);
  }

  // [5.1:288:4-5]
  // An update clause on a depobj construct must not have source, sink or depobj
  // as dependence-type.
  // [5.2:322:3]
  // task-dependence-type must not be depobj.
  if (dir == llvm::omp::OMPD_depobj) {
    if (version >= 51) {
      bool invalidDep{false};
      if (taskType) {
        invalidDep =
            taskType->v == parser::OmpTaskDependenceType::Value::Depobj;
      } else {
        invalidDep = true;
      }
      if (invalidDep) {
        context_.Say(GetContext().clauseSource,
            "An UPDATE clause on a DEPOBJ construct must not have SINK, SOURCE or DEPOBJ as dependence type"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::UseDevicePtr &x) {
  CheckStructureComponent(x.v, llvm::omp::Clause::OMPC_use_device_ptr);
  CheckAllowedClause(llvm::omp::Clause::OMPC_use_device_ptr);
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  semantics::UnorderedSymbolSet listVars;
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_use_device_ptr)) {
    const auto &useDevicePtrClause{
        std::get<parser::OmpClause::UseDevicePtr>(clause->u)};
    const auto &useDevicePtrList{useDevicePtrClause.v};
    std::list<parser::Name> useDevicePtrNameList;
    for (const auto &ompObject : useDevicePtrList.v) {
      if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
        if (name->symbol) {
          if (!(IsBuiltinCPtr(*(name->symbol)))) {
            context_.Warn(common::UsageWarning::OpenMPUsage, clause->source,
                "Use of non-C_PTR type '%s' in USE_DEVICE_PTR is deprecated, use USE_DEVICE_ADDR instead"_warn_en_US,
                name->ToString());
          } else {
            useDevicePtrNameList.push_back(*name);
          }
        }
      }
    }
    CheckMultipleOccurrence(
        listVars, useDevicePtrNameList, clause->source, "USE_DEVICE_PTR");
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::UseDeviceAddr &x) {
  CheckStructureComponent(x.v, llvm::omp::Clause::OMPC_use_device_addr);
  CheckAllowedClause(llvm::omp::Clause::OMPC_use_device_addr);
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  semantics::UnorderedSymbolSet listVars;
  for (auto [_, clause] :
      FindClauses(llvm::omp::Clause::OMPC_use_device_addr)) {
    const auto &useDeviceAddrClause{
        std::get<parser::OmpClause::UseDeviceAddr>(clause->u)};
    const auto &useDeviceAddrList{useDeviceAddrClause.v};
    std::list<parser::Name> useDeviceAddrNameList;
    for (const auto &ompObject : useDeviceAddrList.v) {
      if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
        if (name->symbol) {
          useDeviceAddrNameList.push_back(*name);
        }
      }
    }
    CheckMultipleOccurrence(
        listVars, useDeviceAddrNameList, clause->source, "USE_DEVICE_ADDR");
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::IsDevicePtr &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_is_device_ptr);
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  semantics::UnorderedSymbolSet listVars;
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_is_device_ptr)) {
    const auto &isDevicePtrClause{
        std::get<parser::OmpClause::IsDevicePtr>(clause->u)};
    const auto &isDevicePtrList{isDevicePtrClause.v};
    SymbolSourceMap currSymbols;
    GetSymbolsInObjectList(isDevicePtrList, currSymbols);
    for (auto &[symbol, source] : currSymbols) {
      if (!(IsBuiltinCPtr(*symbol))) {
        context_.Say(clause->source,
            "Variable '%s' in IS_DEVICE_PTR clause must be of type C_PTR"_err_en_US,
            source.ToString());
      } else if (!(IsDummy(*symbol))) {
        context_.Warn(common::UsageWarning::OpenMPUsage, clause->source,
            "Variable '%s' in IS_DEVICE_PTR clause must be a dummy argument. "
            "This semantic check is deprecated from OpenMP 5.2 and later."_warn_en_US,
            source.ToString());
      } else if (IsAllocatableOrPointer(*symbol) || IsValue(*symbol)) {
        context_.Warn(common::UsageWarning::OpenMPUsage, clause->source,
            "Variable '%s' in IS_DEVICE_PTR clause must be a dummy argument "
            "that does not have the ALLOCATABLE, POINTER or VALUE attribute. "
            "This semantic check is deprecated from OpenMP 5.2 and later."_warn_en_US,
            source.ToString());
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::HasDeviceAddr &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_has_device_addr);
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  semantics::UnorderedSymbolSet listVars;
  for (auto [_, clause] :
      FindClauses(llvm::omp::Clause::OMPC_has_device_addr)) {
    const auto &hasDeviceAddrClause{
        std::get<parser::OmpClause::HasDeviceAddr>(clause->u)};
    const auto &hasDeviceAddrList{hasDeviceAddrClause.v};
    std::list<parser::Name> hasDeviceAddrNameList;
    for (const auto &ompObject : hasDeviceAddrList.v) {
      if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
        if (name->symbol) {
          hasDeviceAddrNameList.push_back(*name);
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Enter &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_enter);
  if (!OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_enter, GetContext().clauseSource, context_)) {
    return;
  }
  const parser::OmpObjectList &objList{std::get<parser::OmpObjectList>(x.v.t)};
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objList, symbols);
  for (const auto &[symbol, source] : symbols) {
    if (!IsExtendedListItem(*symbol)) {
      context_.SayWithDecl(*symbol, source,
          "'%s' must be a variable or a procedure"_err_en_US, symbol->name());
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::From &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_from);
  if (!OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_from, GetContext().clauseSource, context_)) {
    return;
  }

  auto &modifiers{OmpGetModifiers(x.v)};
  unsigned version{context_.langOptions().OpenMPVersion};

  if (auto *iter{OmpGetUniqueModifier<parser::OmpIterator>(modifiers)}) {
    CheckIteratorModifier(*iter);
  }

  const auto &objList{std::get<parser::OmpObjectList>(x.v.t)};
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objList, symbols);
  CheckVariableListItem(symbols);

  // Ref: [4.5:109:19]
  // If a list item is an array section it must specify contiguous storage.
  if (version <= 45) {
    for (const parser::OmpObject &object : objList.v) {
      CheckIfContiguous(object);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::To &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_to);
  if (!OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_to, GetContext().clauseSource, context_)) {
    return;
  }

  auto &modifiers{OmpGetModifiers(x.v)};
  unsigned version{context_.langOptions().OpenMPVersion};

  // The "to" clause is only allowed on "declare target" (pre-5.1), and
  // "target update". In the former case it can take an extended list item,
  // in the latter a variable (a locator).

  // The "declare target" construct (and the "to" clause on it) are already
  // handled (in the declare-target checkers), so just look at "to" in "target
  // update".
  if (GetContext().directive == llvm::omp::OMPD_declare_target) {
    return;
  }

  assert(GetContext().directive == llvm::omp::OMPD_target_update);
  if (auto *iter{OmpGetUniqueModifier<parser::OmpIterator>(modifiers)}) {
    CheckIteratorModifier(*iter);
  }

  const auto &objList{std::get<parser::OmpObjectList>(x.v.t)};
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objList, symbols);
  CheckVariableListItem(symbols);

  // Ref: [4.5:109:19]
  // If a list item is an array section it must specify contiguous storage.
  if (version <= 45) {
    for (const parser::OmpObject &object : objList.v) {
      CheckIfContiguous(object);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::OmpxBare &x) {
  // Don't call CheckAllowedClause, because it allows "ompx_bare" on
  // a non-combined "target" directive (for reasons of splitting combined
  // directives). In source code it's only allowed on "target teams".
  if (GetContext().directive != llvm::omp::Directive::OMPD_target_teams) {
    context_.Say(GetContext().clauseSource,
        "%s clause is only allowed on combined TARGET TEAMS"_err_en_US,
        parser::ToUpperCaseLetters(getClauseName(llvm::omp::OMPC_ompx_bare)));
  }
}

llvm::StringRef OmpStructureChecker::getClauseName(llvm::omp::Clause clause) {
  return llvm::omp::getOpenMPClauseName(clause);
}

llvm::StringRef OmpStructureChecker::getDirectiveName(
    llvm::omp::Directive directive) {
  unsigned version{context_.langOptions().OpenMPVersion};
  return llvm::omp::getOpenMPDirectiveName(directive, version);
}

void OmpStructureChecker::CheckDependList(const parser::DataRef &d) {
  common::visit(
      common::visitors{
          [&](const common::Indirection<parser::ArrayElement> &elem) {
            // Check if the base element is valid on Depend Clause
            CheckDependList(elem.value().base);
          },
          [&](const common::Indirection<parser::StructureComponent> &comp) {
            CheckDependList(comp.value().base);
          },
          [&](const common::Indirection<parser::CoindexedNamedObject> &) {
            context_.Say(GetContext().clauseSource,
                "Coarrays are not supported in DEPEND clause"_err_en_US);
          },
          [&](const parser::Name &) {},
      },
      d.u);
}

// Called from both Reduction and Depend clause.
void OmpStructureChecker::CheckArraySection(
    const parser::ArrayElement &arrayElement, const parser::Name &name,
    const llvm::omp::Clause clause) {
  // Sometimes substring operations are incorrectly parsed as array accesses.
  // Detect this by looking for array accesses on character variables which are
  // not arrays.
  bool isSubstring{false};
  // Cannot analyze a base of an assumed-size array on its own. If we know
  // this is an array (assumed-size or not) we can ignore it, since we're
  // looking for strings.
  if (!IsAssumedSizeArray(*name.symbol)) {
    evaluate::ExpressionAnalyzer ea{context_};
    if (MaybeExpr expr = ea.Analyze(arrayElement.base)) {
      if (expr->Rank() == 0) {
        // Not an array: rank 0
        if (std::optional<evaluate::DynamicType> type = expr->GetType()) {
          if (type->category() == evaluate::TypeCategory::Character) {
            // Substrings are explicitly denied by the standard [6.0:163:9-11].
            // This is supported as an extension. This restriction was added in
            // OpenMP 5.2.
            isSubstring = true;
            context_.Say(GetContext().clauseSource,
                "The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2."_port_en_US);
          } else {
            llvm_unreachable(
                "Array indexing on a variable that isn't an array");
          }
        }
      }
    }
  }
  if (!arrayElement.subscripts.empty()) {
    for (const auto &subscript : arrayElement.subscripts) {
      if (const auto *triplet{
              std::get_if<parser::SubscriptTriplet>(&subscript.u)}) {
        if (std::get<0>(triplet->t) && std::get<1>(triplet->t)) {
          std::optional<int64_t> strideVal{std::nullopt};
          if (const auto &strideExpr = std::get<2>(triplet->t)) {
            // OpenMP 6.0 Section 5.2.5: Array Sections
            // Restrictions: if a stride expression is specified it must be
            // positive. A stride of 0 doesn't make sense.
            strideVal = GetIntValue(strideExpr);
            if (strideVal && *strideVal < 1) {
              context_.Say(GetContext().clauseSource,
                  "'%s' in %s clause must have a positive stride"_err_en_US,
                  name.ToString(),
                  parser::ToUpperCaseLetters(getClauseName(clause).str()));
            }
            if (isSubstring) {
              context_.Say(GetContext().clauseSource,
                  "Cannot specify a step for a substring"_err_en_US);
            }
          }
          const auto &lower{std::get<0>(triplet->t)};
          const auto &upper{std::get<1>(triplet->t)};
          if (lower && upper) {
            const auto lval{GetIntValue(lower)};
            const auto uval{GetIntValue(upper)};
            if (lval && uval) {
              int64_t sectionLen = *uval - *lval;
              if (strideVal) {
                sectionLen = sectionLen / *strideVal;
              }

              if (sectionLen < 1) {
                context_.Say(GetContext().clauseSource,
                    "'%s' in %s clause"
                    " is a zero size array section"_err_en_US,
                    name.ToString(),
                    parser::ToUpperCaseLetters(getClauseName(clause).str()));
                break;
              }
            }
          }
        }
      } else if (std::get_if<parser::IntExpr>(&subscript.u)) {
        // base(n) is valid as an array index but not as a substring operation
        if (isSubstring) {
          context_.Say(GetContext().clauseSource,
              "Substrings must be in the form parent-string(lb:ub)"_err_en_US);
        }
      }
    }
  }
}

void OmpStructureChecker::CheckIntentInPointer(
    SymbolSourceMap &symbols, llvm::omp::Clause clauseId) {
  for (auto &[symbol, source] : symbols) {
    if (IsPointer(*symbol) && IsIntentIn(*symbol)) {
      context_.Say(source,
          "Pointer '%s' with the INTENT(IN) attribute may not appear in a %s clause"_err_en_US,
          symbol->name(),
          parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
    }
  }
}

void OmpStructureChecker::CheckProcedurePointer(
    SymbolSourceMap &symbols, llvm::omp::Clause clause) {
  for (const auto &[symbol, source] : symbols) {
    if (IsProcedurePointer(*symbol)) {
      context_.Say(source,
          "Procedure pointer '%s' may not appear in a %s clause"_err_en_US,
          symbol->name(),
          parser::ToUpperCaseLetters(getClauseName(clause).str()));
    }
  }
}

void OmpStructureChecker::CheckCrayPointee(
    const parser::OmpObjectList &objectList, llvm::StringRef clause,
    bool suggestToUseCrayPointer) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objectList, symbols);
  for (auto it{symbols.begin()}; it != symbols.end(); ++it) {
    const auto *symbol{it->first};
    const auto source{it->second};
    if (symbol->test(Symbol::Flag::CrayPointee)) {
      std::string suggestionMsg = "";
      if (suggestToUseCrayPointer)
        suggestionMsg = ", use Cray Pointer '" +
            semantics::GetCrayPointer(*symbol).name().ToString() + "' instead";
      context_.Say(source,
          "Cray Pointee '%s' may not appear in %s clause%s"_err_en_US,
          symbol->name(), clause.str(), suggestionMsg);
    }
  }
}

void OmpStructureChecker::GetSymbolsInObjectList(
    const parser::OmpObjectList &objectList, SymbolSourceMap &symbols) {
  for (const auto &ompObject : objectList.v) {
    if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
      if (const auto *symbol{name->symbol}) {
        if (const auto *commonBlockDetails{
                symbol->detailsIf<CommonBlockDetails>()}) {
          for (const auto &object : commonBlockDetails->objects()) {
            symbols.emplace(&object->GetUltimate(), name->source);
          }
        } else {
          symbols.emplace(&symbol->GetUltimate(), name->source);
        }
      }
    }
  }
}

void OmpStructureChecker::CheckDefinableObjects(
    SymbolSourceMap &symbols, const llvm::omp::Clause clause) {
  for (auto &[symbol, source] : symbols) {
    if (auto msg{WhyNotDefinable(source, context_.FindScope(source),
            DefinabilityFlags{}, *symbol)}) {
      context_
          .Say(source,
              "Variable '%s' on the %s clause is not definable"_err_en_US,
              symbol->name(),
              parser::ToUpperCaseLetters(getClauseName(clause).str()))
          .Attach(std::move(msg->set_severity(parser::Severity::Because)));
    }
  }
}

void OmpStructureChecker::CheckPrivateSymbolsInOuterCxt(
    SymbolSourceMap &currSymbols, DirectivesClauseTriple &dirClauseTriple,
    const llvm::omp::Clause currClause) {
  SymbolSourceMap enclosingSymbols;
  auto range{dirClauseTriple.equal_range(GetContext().directive)};
  for (auto dirIter{range.first}; dirIter != range.second; ++dirIter) {
    auto enclosingDir{dirIter->second.first};
    auto enclosingClauseSet{dirIter->second.second};
    if (auto *enclosingContext{GetEnclosingContextWithDir(enclosingDir)}) {
      for (auto it{enclosingContext->clauseInfo.begin()};
          it != enclosingContext->clauseInfo.end(); ++it) {
        if (enclosingClauseSet.test(it->first)) {
          if (const auto *ompObjectList{GetOmpObjectList(*it->second)}) {
            GetSymbolsInObjectList(*ompObjectList, enclosingSymbols);
          }
        }
      }

      // Check if the symbols in current context are private in outer context
      for (auto &[symbol, source] : currSymbols) {
        if (enclosingSymbols.find(symbol) != enclosingSymbols.end()) {
          context_.Say(source,
              "%s variable '%s' is PRIVATE in outer context"_err_en_US,
              parser::ToUpperCaseLetters(getClauseName(currClause).str()),
              symbol->name());
        }
      }
    }
  }
}

bool OmpStructureChecker::CheckTargetBlockOnlyTeams(
    const parser::Block &block) {
  bool nestedTeams{false};

  if (!block.empty()) {
    auto it{block.begin()};
    if (const auto *ompConstruct{
            parser::Unwrap<parser::OpenMPConstruct>(*it)}) {
      if (const auto *ompBlockConstruct{
              std::get_if<parser::OpenMPBlockConstruct>(&ompConstruct->u)}) {
        llvm::omp::Directive dirId{ompBlockConstruct->BeginDir().DirId()};
        if (dirId == llvm::omp::Directive::OMPD_teams) {
          nestedTeams = true;
        }
      }
    }

    if (nestedTeams && ++it == block.end()) {
      return true;
    }
  }

  return false;
}

void OmpStructureChecker::CheckWorkshareBlockStmts(
    const parser::Block &block, parser::CharBlock source) {
  OmpWorkshareBlockChecker ompWorkshareBlockChecker{context_, source};

  for (auto it{block.begin()}; it != block.end(); ++it) {
    if (parser::Unwrap<parser::AssignmentStmt>(*it) ||
        parser::Unwrap<parser::ForallStmt>(*it) ||
        parser::Unwrap<parser::ForallConstruct>(*it) ||
        parser::Unwrap<parser::WhereStmt>(*it) ||
        parser::Unwrap<parser::WhereConstruct>(*it)) {
      parser::Walk(*it, ompWorkshareBlockChecker);
    } else if (const auto *ompConstruct{
                   parser::Unwrap<parser::OpenMPConstruct>(*it)}) {
      if (const auto *ompAtomicConstruct{
              std::get_if<parser::OpenMPAtomicConstruct>(&ompConstruct->u)}) {
        // Check if assignment statements in the enclosing OpenMP Atomic
        // construct are allowed in the Workshare construct
        parser::Walk(*ompAtomicConstruct, ompWorkshareBlockChecker);
      } else if (const auto *ompCriticalConstruct{
                     std::get_if<parser::OpenMPCriticalConstruct>(
                         &ompConstruct->u)}) {
        // All the restrictions on the Workshare construct apply to the
        // statements in the enclosing critical constructs
        const auto &criticalBlock{
            std::get<parser::Block>(ompCriticalConstruct->t)};
        CheckWorkshareBlockStmts(criticalBlock, source);
      } else {
        // Check if OpenMP constructs enclosed in the Workshare construct are
        // 'Parallel' constructs
        auto currentDir{llvm::omp::Directive::OMPD_unknown};
        if (const auto *ompBlockConstruct{
                std::get_if<parser::OpenMPBlockConstruct>(&ompConstruct->u)}) {
          currentDir = ompBlockConstruct->BeginDir().DirId();
        } else if (const auto *ompLoopConstruct{
                       std::get_if<parser::OpenMPLoopConstruct>(
                           &ompConstruct->u)}) {
          const auto &beginLoopDir{
              std::get<parser::OmpBeginLoopDirective>(ompLoopConstruct->t)};
          const auto &beginDir{
              std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
          currentDir = beginDir.v;
        } else if (const auto *ompSectionsConstruct{
                       std::get_if<parser::OpenMPSectionsConstruct>(
                           &ompConstruct->u)}) {
          const auto &beginSectionsDir{
              std::get<parser::OmpBeginSectionsDirective>(
                  ompSectionsConstruct->t)};
          const auto &beginDir{
              std::get<parser::OmpSectionsDirective>(beginSectionsDir.t)};
          currentDir = beginDir.v;
        }

        if (!llvm::omp::topParallelSet.test(currentDir)) {
          context_.Say(source,
              "OpenMP constructs enclosed in WORKSHARE construct may consist "
              "of ATOMIC, CRITICAL or PARALLEL constructs only"_err_en_US);
        }
      }
    } else {
      context_.Say(source,
          "The structured block in a WORKSHARE construct may consist of only "
          "SCALAR or ARRAY assignments, FORALL or WHERE statements, "
          "FORALL, WHERE, ATOMIC, CRITICAL or PARALLEL constructs"_err_en_US);
    }
  }
}

void OmpStructureChecker::CheckWorkdistributeBlockStmts(
    const parser::Block &block, parser::CharBlock source) {
  unsigned version{context_.langOptions().OpenMPVersion};
  unsigned since{60};
  if (version < since)
    context_.Say(source,
        "WORKDISTRIBUTE construct is not allowed in %s, %s"_err_en_US,
        ThisVersion(version), TryVersion(since));

  OmpWorkdistributeBlockChecker ompWorkdistributeBlockChecker{context_, source};

  for (auto it{block.begin()}; it != block.end(); ++it) {
    if (parser::Unwrap<parser::AssignmentStmt>(*it)) {
      parser::Walk(*it, ompWorkdistributeBlockChecker);
    } else {
      context_.Say(source,
          "The structured block in a WORKDISTRIBUTE construct may consist of only SCALAR or ARRAY assignments"_err_en_US);
    }
  }
}

void OmpStructureChecker::CheckIfContiguous(const parser::OmpObject &object) {
  if (auto contig{IsContiguous(context_, object)}; contig && !*contig) {
    const parser::Name *name{GetObjectName(object)};
    assert(name && "Expecting name component");
    context_.Say(name->source,
        "Reference to '%s' must be a contiguous object"_err_en_US,
        name->ToString());
  }
}

namespace {
struct NameHelper {
  template <typename T>
  static const parser::Name *Visit(const common::Indirection<T> &x) {
    return Visit(x.value());
  }
  static const parser::Name *Visit(const parser::Substring &x) {
    return Visit(std::get<parser::DataRef>(x.t));
  }
  static const parser::Name *Visit(const parser::ArrayElement &x) {
    return Visit(x.base);
  }
  static const parser::Name *Visit(const parser::Designator &x) {
    return common::visit([](auto &&s) { return Visit(s); }, x.u);
  }
  static const parser::Name *Visit(const parser::DataRef &x) {
    return common::visit([](auto &&s) { return Visit(s); }, x.u);
  }
  static const parser::Name *Visit(const parser::OmpObject &x) {
    return common::visit([](auto &&s) { return Visit(s); }, x.u);
  }
  template <typename T> static const parser::Name *Visit(T &&) {
    return nullptr;
  }
  static const parser::Name *Visit(const parser::Name &x) { return &x; }
};
} // namespace

const parser::Name *OmpStructureChecker::GetObjectName(
    const parser::OmpObject &object) {
  return NameHelper::Visit(object);
}

void OmpStructureChecker::Enter(
    const parser::OmpClause::AtomicDefaultMemOrder &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_atomic_default_mem_order);
}

void OmpStructureChecker::Enter(const parser::OmpClause::DynamicAllocators &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_dynamic_allocators);
}

void OmpStructureChecker::Enter(const parser::OmpClause::ReverseOffload &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_reverse_offload);
}

void OmpStructureChecker::Enter(const parser::OmpClause::UnifiedAddress &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_unified_address);
}

void OmpStructureChecker::Enter(
    const parser::OmpClause::UnifiedSharedMemory &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_unified_shared_memory);
}

void OmpStructureChecker::Enter(const parser::OmpClause::SelfMaps &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_self_maps);
}

void OmpStructureChecker::Enter(const parser::OpenMPInteropConstruct &x) {
  bool isDependClauseOccured{false};
  int targetCount{0}, targetSyncCount{0};
  const auto &dir{std::get<parser::OmpDirectiveName>(x.v.t)};
  std::set<const Symbol *> objectSymbolList;
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_interop);
  const auto &clauseList{std::get<std::optional<parser::OmpClauseList>>(x.v.t)};
  for (const auto &clause : clauseList->v) {
    common::visit(
        common::visitors{
            [&](const parser::OmpClause::Init &initClause) {
              if (OmpVerifyModifiers(initClause.v, llvm::omp::OMPC_init,
                      GetContext().directiveSource, context_)) {

                auto &modifiers{OmpGetModifiers(initClause.v)};
                auto &&interopTypeModifier{
                    OmpGetRepeatableModifier<parser::OmpInteropType>(
                        modifiers)};
                for (const auto &it : interopTypeModifier) {
                  if (it->v == parser::OmpInteropType::Value::TargetSync) {
                    ++targetSyncCount;
                  } else {
                    ++targetCount;
                  }
                }
              }
              const auto &interopVar{parser::Unwrap<parser::OmpObject>(
                  std::get<parser::OmpObject>(initClause.v.t))};
              const auto *name{parser::Unwrap<parser::Name>(interopVar)};
              const auto *objectSymbol{name->symbol};
              if (llvm::is_contained(objectSymbolList, objectSymbol)) {
                context_.Say(GetContext().directiveSource,
                    "Each interop-var may be specified for at most one action-clause of each INTEROP construct."_err_en_US);
              } else {
                objectSymbolList.insert(objectSymbol);
              }
            },
            [&](const parser::OmpClause::Depend &dependClause) {
              isDependClauseOccured = true;
            },
            [&](const parser::OmpClause::Destroy &destroyClause) {
              const auto &interopVar{
                  parser::Unwrap<parser::OmpObject>(destroyClause.v)};
              const auto *name{parser::Unwrap<parser::Name>(interopVar)};
              const auto *objectSymbol{name->symbol};
              if (llvm::is_contained(objectSymbolList, objectSymbol)) {
                context_.Say(GetContext().directiveSource,
                    "Each interop-var may be specified for at most one action-clause of each INTEROP construct."_err_en_US);
              } else {
                objectSymbolList.insert(objectSymbol);
              }
            },
            [&](const parser::OmpClause::Use &useClause) {
              const auto &interopVar{
                  parser::Unwrap<parser::OmpObject>(useClause.v)};
              const auto *name{parser::Unwrap<parser::Name>(interopVar)};
              const auto *objectSymbol{name->symbol};
              if (llvm::is_contained(objectSymbolList, objectSymbol)) {
                context_.Say(GetContext().directiveSource,
                    "Each interop-var may be specified for at most one action-clause of each INTEROP construct."_err_en_US);
              } else {
                objectSymbolList.insert(objectSymbol);
              }
            },
            [&](const auto &) {},
        },
        clause.u);
  }
  if (targetCount > 1 || targetSyncCount > 1) {
    context_.Say(GetContext().directiveSource,
        "Each interop-type may be specified at most once."_err_en_US);
  }
  if (isDependClauseOccured && !targetSyncCount) {
    context_.Say(GetContext().directiveSource,
        "A DEPEND clause can only appear on the directive if the interop-type includes TARGETSYNC"_err_en_US);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPInteropConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::CheckAllowedRequiresClause(llvmOmpClause clause) {
  CheckAllowedClause(clause);

  if (clause != llvm::omp::Clause::OMPC_atomic_default_mem_order) {
    // Check that it does not appear after a device construct
    if (deviceConstructFound_) {
      context_.Say(GetContext().clauseSource,
          "REQUIRES directive with '%s' clause found lexically after device "
          "construct"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clause).str()));
    }
  }
}

} // namespace Fortran::semantics
