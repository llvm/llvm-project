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
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/expression.h"
#include <list>
#include <map>

namespace Fortran::semantics {

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
    bool withinConstruct{false};
    std::int64_t associatedLoopLevel{0};
  };

  DirContext &GetContext() {
    CHECK(!dirContext_.empty());
    return dirContext_.back();
  }
  void PushContext(const parser::CharBlock &source, T dir) {
    dirContext_.emplace_back(source, dir, context_.FindScope(source));
  }
  void PopContext() { dirContext_.pop_back(); }
  void SetContextDirectiveSource(parser::CharBlock &dir) {
    GetContext().directiveSource = dir;
  }
  Scope &currScope() { return GetContext().scope; }
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
    auto it{GetContext().objectWithDSA.find(&symbol)};
    return it != GetContext().objectWithDSA.end();
  }
  void SetContextAssociatedLoopLevel(std::int64_t level) {
    GetContext().associatedLoopLevel = level;
  }
  Symbol &MakeAssocSymbol(const SourceName &name, Symbol &prev, Scope &scope) {
    const auto pair{scope.try_emplace(name, Attrs{}, HostAssocDetails{prev})};
    return *pair.first->second;
  }
  Symbol &MakeAssocSymbol(const SourceName &name, Symbol &prev) {
    return MakeAssocSymbol(name, prev, currScope());
  }
  static const parser::Name *GetDesignatorNameIfDataRef(
      const parser::Designator &designator) {
    const auto *dataRef{std::get_if<parser::DataRef>(&designator.u)};
    return dataRef ? std::get_if<parser::Name>(&dataRef->u) : nullptr;
  }
  void AddDataSharingAttributeObject(SymbolRef object) {
    dataSharingAttributeObjects_.insert(object);
  }
  void ClearDataSharingAttributeObjects() {
    dataSharingAttributeObjects_.clear();
  }
  bool HasDataSharingAttributeObject(const Symbol &);
  const parser::Name &GetLoopIndex(const parser::DoConstruct &);
  const parser::DoConstruct *GetDoConstructIf(
      const parser::ExecutionPartConstruct &);
  Symbol *DeclarePrivateAccessEntity(
      const parser::Name &, Symbol::Flag, Scope &);
  Symbol *DeclarePrivateAccessEntity(Symbol &, Symbol::Flag, Scope &);
  Symbol *DeclareOrMarkOtherAccessEntity(const parser::Name &, Symbol::Flag);

  SymbolSet dataSharingAttributeObjects_; // on one directive
  SemanticsContext &context_;
  std::vector<DirContext> dirContext_; // used as a stack
};

class AccAttributeVisitor : DirectiveAttributeVisitor<llvm::acc::Directive> {
public:
  explicit AccAttributeVisitor(SemanticsContext &context)
      : DirectiveAttributeVisitor(context) {}

  template <typename A> void Walk(const A &x) { parser::Walk(x, *this); }
  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}

  bool Pre(const parser::SpecificationPart &x) {
    Walk(std::get<std::list<parser::OpenACCDeclarativeConstruct>>(x.t));
    return false;
  }

  bool Pre(const parser::OpenACCBlockConstruct &);
  void Post(const parser::OpenACCBlockConstruct &) { PopContext(); }
  bool Pre(const parser::OpenACCCombinedConstruct &);
  void Post(const parser::OpenACCCombinedConstruct &) { PopContext(); }

  void Post(const parser::AccBeginBlockDirective &) {
    GetContext().withinConstruct = true;
  }

  bool Pre(const parser::OpenACCLoopConstruct &);
  void Post(const parser::OpenACCLoopConstruct &) { PopContext(); }
  void Post(const parser::AccLoopDirective &) {
    GetContext().withinConstruct = true;
  }

  bool Pre(const parser::OpenACCStandaloneConstruct &);
  void Post(const parser::OpenACCStandaloneConstruct &) { PopContext(); }
  void Post(const parser::AccStandaloneDirective &) {
    GetContext().withinConstruct = true;
  }

  void Post(const parser::AccDefaultClause &);

  bool Pre(const parser::AccClause::Copy &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccCopyIn);
    ResolveAccObjectList(x.v, Symbol::Flag::AccCopyOut);
    return false;
  }

  bool Pre(const parser::AccClause::Create &x) {
    const auto &objectList{std::get<parser::AccObjectList>(x.v.t)};
    ResolveAccObjectList(objectList, Symbol::Flag::AccCreate);
    return false;
  }

  bool Pre(const parser::AccClause::Copyin &x) {
    const auto &objectList{std::get<parser::AccObjectList>(x.v.t)};
    ResolveAccObjectList(objectList, Symbol::Flag::AccCopyIn);
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
  bool Pre(const parser::AccClause::FirstPrivate &x) {
    ResolveAccObjectList(x.v, Symbol::Flag::AccFirstPrivate);
    return false;
  }

  void Post(const parser::Name &);

private:
  std::int64_t GetAssociatedLoopLevelFromClauses(const parser::AccClauseList &);

  static constexpr Symbol::Flags dataSharingAttributeFlags{
      Symbol::Flag::AccShared, Symbol::Flag::AccPrivate,
      Symbol::Flag::AccPresent, Symbol::Flag::AccFirstPrivate,
      Symbol::Flag::AccReduction};

  static constexpr Symbol::Flags dataMappingAttributeFlags{
      Symbol::Flag::AccCreate, Symbol::Flag::AccCopyIn,
      Symbol::Flag::AccCopyOut, Symbol::Flag::AccDelete};

  static constexpr Symbol::Flags accFlagsRequireNewSymbol{
      Symbol::Flag::AccPrivate, Symbol::Flag::AccFirstPrivate,
      Symbol::Flag::AccReduction};

  static constexpr Symbol::Flags accFlagsRequireMark{};

  void PrivatizeAssociatedLoopIndex(const parser::OpenACCLoopConstruct &);
  void ResolveAccObjectList(const parser::AccObjectList &, Symbol::Flag);
  void ResolveAccObject(const parser::AccObject &, Symbol::Flag);
  Symbol *ResolveAcc(const parser::Name &, Symbol::Flag, Scope &);
  Symbol *ResolveAcc(Symbol &, Symbol::Flag, Scope &);
  Symbol *ResolveAccCommonBlockName(const parser::Name *);
  Symbol *DeclareOrMarkOtherAccessEntity(const parser::Name &, Symbol::Flag);
  Symbol *DeclareOrMarkOtherAccessEntity(Symbol &, Symbol::Flag);
  void CheckMultipleAppearances(
      const parser::Name &, const Symbol &, Symbol::Flag);
};

// Data-sharing and Data-mapping attributes for data-refs in OpenMP construct
class OmpAttributeVisitor : DirectiveAttributeVisitor<llvm::omp::Directive> {
public:
  explicit OmpAttributeVisitor(SemanticsContext &context)
      : DirectiveAttributeVisitor(context) {}

  template <typename A> void Walk(const A &x) { parser::Walk(x, *this); }
  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}

  bool Pre(const parser::SpecificationPart &x) {
    Walk(std::get<std::list<parser::OpenMPDeclarativeConstruct>>(x.t));
    return false;
  }

  bool Pre(const parser::OpenMPBlockConstruct &);
  void Post(const parser::OpenMPBlockConstruct &) { PopContext(); }
  void Post(const parser::OmpBeginBlockDirective &) {
    GetContext().withinConstruct = true;
  }

  bool Pre(const parser::OpenMPLoopConstruct &);
  void Post(const parser::OpenMPLoopConstruct &) { PopContext(); }
  void Post(const parser::OmpBeginLoopDirective &) {
    GetContext().withinConstruct = true;
  }
  bool Pre(const parser::DoConstruct &);

  bool Pre(const parser::OpenMPSectionsConstruct &);
  void Post(const parser::OpenMPSectionsConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPThreadprivate &);
  void Post(const parser::OpenMPThreadprivate &) { PopContext(); }

  // 2.15.3 Data-Sharing Attribute Clauses
  void Post(const parser::OmpDefaultClause &);
  bool Pre(const parser::OmpClause::Shared &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpShared);
    return false;
  }
  bool Pre(const parser::OmpClause::Private &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpPrivate);
    return false;
  }
  bool Pre(const parser::OmpClause::Firstprivate &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpFirstPrivate);
    return false;
  }
  bool Pre(const parser::OmpClause::Lastprivate &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpLastPrivate);
    return false;
  }

  void Post(const parser::Name &);

private:
  std::int64_t GetAssociatedLoopLevelFromClauses(const parser::OmpClauseList &);

  static constexpr Symbol::Flags dataSharingAttributeFlags{
      Symbol::Flag::OmpShared, Symbol::Flag::OmpPrivate,
      Symbol::Flag::OmpFirstPrivate, Symbol::Flag::OmpLastPrivate,
      Symbol::Flag::OmpReduction, Symbol::Flag::OmpLinear};

  static constexpr Symbol::Flags ompFlagsRequireNewSymbol{
      Symbol::Flag::OmpPrivate, Symbol::Flag::OmpLinear,
      Symbol::Flag::OmpFirstPrivate, Symbol::Flag::OmpLastPrivate,
      Symbol::Flag::OmpReduction};

  static constexpr Symbol::Flags ompFlagsRequireMark{
      Symbol::Flag::OmpThreadprivate};

  // Predetermined DSA rules
  void PrivatizeAssociatedLoopIndex(const parser::OpenMPLoopConstruct &);
  void ResolveSeqLoopIndexInParallelOrTaskConstruct(const parser::Name &);

  void ResolveOmpObjectList(const parser::OmpObjectList &, Symbol::Flag);
  void ResolveOmpObject(const parser::OmpObject &, Symbol::Flag);
  Symbol *ResolveOmp(const parser::Name &, Symbol::Flag, Scope &);
  Symbol *ResolveOmp(Symbol &, Symbol::Flag, Scope &);
  Symbol *ResolveOmpCommonBlockName(const parser::Name *);
  Symbol *DeclareOrMarkOtherAccessEntity(const parser::Name &, Symbol::Flag);
  Symbol *DeclareOrMarkOtherAccessEntity(Symbol &, Symbol::Flag);
  void CheckMultipleAppearances(
      const parser::Name &, const Symbol &, Symbol::Flag);
};

template <typename T>
bool DirectiveAttributeVisitor<T>::HasDataSharingAttributeObject(
    const Symbol &object) {
  auto it{dataSharingAttributeObjects_.find(object)};
  return it != dataSharingAttributeObjects_.end();
}

template <typename T>
const parser::Name &DirectiveAttributeVisitor<T>::GetLoopIndex(
    const parser::DoConstruct &x) {
  using Bounds = parser::LoopControl::Bounds;
  return std::get<Bounds>(x.GetLoopControl()->u).name.thing;
}

template <typename T>
const parser::DoConstruct *DirectiveAttributeVisitor<T>::GetDoConstructIf(
    const parser::ExecutionPartConstruct &x) {
  return parser::Unwrap<parser::DoConstruct>(x);
}

template <typename T>
Symbol *DirectiveAttributeVisitor<T>::DeclarePrivateAccessEntity(
    const parser::Name &name, Symbol::Flag flag, Scope &scope) {
  if (!name.symbol) {
    return nullptr; // not resolved by Name Resolution step, do nothing
  }
  name.symbol = DeclarePrivateAccessEntity(*name.symbol, flag, scope);
  return name.symbol;
}

template <typename T>
Symbol *DirectiveAttributeVisitor<T>::DeclarePrivateAccessEntity(
    Symbol &object, Symbol::Flag flag, Scope &scope) {
  if (object.owner() != currScope()) {
    auto &symbol{MakeAssocSymbol(object.name(), object, scope)};
    symbol.set(flag);
    return &symbol;
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
  return true;
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
  PrivatizeAssociatedLoopIndex(x);
  return true;
}

bool AccAttributeVisitor::Pre(const parser::OpenACCStandaloneConstruct &x) {
  const auto &standaloneDir{std::get<parser::AccStandaloneDirective>(x.t)};
  switch (standaloneDir.v) {
  case llvm::acc::Directive::ACCD_cache:
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

bool AccAttributeVisitor::Pre(const parser::OpenACCCombinedConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginCombinedDirective>(x.t)};
  const auto &combinedDir{
      std::get<parser::AccCombinedDirective>(beginBlockDir.t)};
  switch (combinedDir.v) {
  case llvm::acc::Directive::ACCD_kernels_loop:
  case llvm::acc::Directive::ACCD_parallel_loop:
  case llvm::acc::Directive::ACCD_serial_loop:
    PushContext(combinedDir.source, combinedDir.v);
    break;
  default:
    break;
  }
  ClearDataSharingAttributeObjects();
  return true;
}

std::int64_t AccAttributeVisitor::GetAssociatedLoopLevelFromClauses(
    const parser::AccClauseList &x) {
  std::int64_t collapseLevel{0};
  for (const auto &clause : x.v) {
    if (const auto *collapseClause{
            std::get_if<parser::AccClause::Collapse>(&clause.u)}) {
      if (const auto v{EvaluateInt64(context_, collapseClause->v)}) {
        collapseLevel = *v;
      }
    }
  }

  if (collapseLevel) {
    return collapseLevel;
  }
  return 1; // default is outermost loop
}

void AccAttributeVisitor::PrivatizeAssociatedLoopIndex(
    const parser::OpenACCLoopConstruct &x) {
  std::int64_t level{GetContext().associatedLoopLevel};
  if (level <= 0) { // collpase value was negative or 0
    return;
  }
  Symbol::Flag ivDSA{Symbol::Flag::AccPrivate};

  const auto &outer{std::get<std::optional<parser::DoConstruct>>(x.t)};
  for (const parser::DoConstruct *loop{&*outer}; loop && level > 0; --level) {
    // go through all the nested do-loops and resolve index variables
    const parser::Name &iv{GetLoopIndex(*loop)};
    if (auto *symbol{ResolveAcc(iv, ivDSA, currScope())}) {
      symbol->set(Symbol::Flag::AccPreDetermined);
      iv.symbol = symbol; // adjust the symbol within region
      AddToContextObjectWithDSA(*symbol, ivDSA);
    }

    const auto &block{std::get<parser::Block>(loop->t)};
    const auto it{block.begin()};
    loop = it != block.end() ? GetDoConstructIf(*it) : nullptr;
  }
  CHECK(level == 0);
}

void AccAttributeVisitor::Post(const parser::AccDefaultClause &x) {
  if (!dirContext_.empty()) {
    switch (x.v) {
    case parser::AccDefaultClause::Arg::Present:
      SetContextDefaultDSA(Symbol::Flag::AccPresent);
      break;
    case parser::AccDefaultClause::Arg::None:
      SetContextDefaultDSA(Symbol::Flag::AccNone);
      break;
    }
  }
}

// For OpenACC constructs, check all the data-refs within the constructs
// and adjust the symbol for each Name if necessary
void AccAttributeVisitor::Post(const parser::Name &name) {
  auto *symbol{name.symbol};
  if (symbol && !dirContext_.empty() && GetContext().withinConstruct) {
    if (!symbol->owner().IsDerivedType() && !symbol->has<ProcEntityDetails>() &&
        !IsObjectWithDSA(*symbol)) {
      if (Symbol * found{currScope().FindSymbol(name.source)}) {
        if (symbol != found) {
          name.symbol = found; // adjust the symbol within region
        } else if (GetContext().defaultDSA == Symbol::Flag::AccNone) {
          // 2.5.14.
          context_.Say(name.source,
              "The DEFAULT(NONE) clause requires that '%s' must be listed in "
              "a data-mapping clause"_err_en_US,
              symbol->name());
        }
      }
    }
  } // within OpenACC construct
}

Symbol *AccAttributeVisitor::ResolveAccCommonBlockName(
    const parser::Name *name) {
  if (!name) {
    return nullptr;
  } else if (auto *prev{
                 GetContext().scope.parent().FindCommonBlock(name->source)}) {
    name->symbol = prev;
    return prev;
  } else {
    return nullptr;
  }
}

void AccAttributeVisitor::ResolveAccObjectList(
    const parser::AccObjectList &accObjectList, Symbol::Flag accFlag) {
  for (const auto &accObject : accObjectList.v) {
    ResolveAccObject(accObject, accFlag);
  }
}

void AccAttributeVisitor::ResolveAccObject(
    const parser::AccObject &accObject, Symbol::Flag accFlag) {
  std::visit(
      common::visitors{
          [&](const parser::Designator &designator) {
            if (const auto *name{GetDesignatorNameIfDataRef(designator)}) {
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
                  "COMMON block must be declared in the same scoping unit "
                  "in which the OpenACC directive or clause appears"_err_en_US);
            }
          },
      },
      accObject.u);
}

Symbol *AccAttributeVisitor::ResolveAcc(
    const parser::Name &name, Symbol::Flag accFlag, Scope &scope) {
  if (accFlagsRequireNewSymbol.test(accFlag)) {
    return DeclarePrivateAccessEntity(name, accFlag, scope);
  } else {
    return DeclareOrMarkOtherAccessEntity(name, accFlag);
  }
}

Symbol *AccAttributeVisitor::ResolveAcc(
    Symbol &symbol, Symbol::Flag accFlag, Scope &scope) {
  if (accFlagsRequireNewSymbol.test(accFlag)) {
    return DeclarePrivateAccessEntity(symbol, accFlag, scope);
  } else {
    return DeclareOrMarkOtherAccessEntity(symbol, accFlag);
  }
}

Symbol *AccAttributeVisitor::DeclareOrMarkOtherAccessEntity(
    const parser::Name &name, Symbol::Flag accFlag) {
  Symbol *prev{currScope().FindSymbol(name.source)};
  if (!name.symbol || !prev) {
    return nullptr;
  } else if (prev != name.symbol) {
    name.symbol = prev;
  }
  return DeclareOrMarkOtherAccessEntity(*prev, accFlag);
}

Symbol *AccAttributeVisitor::DeclareOrMarkOtherAccessEntity(
    Symbol &object, Symbol::Flag accFlag) {
  if (accFlagsRequireMark.test(accFlag)) {
    object.set(accFlag);
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
  if (accFlagsRequireNewSymbol.test(accFlag)) {
    if (const auto *details{symbol.detailsIf<HostAssocDetails>()}) {
      target = &details->symbol();
    }
  }
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

bool OmpAttributeVisitor::Pre(const parser::OpenMPBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::OmpBeginBlockDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
  switch (beginDir.v) {
  case llvm::omp::Directive::OMPD_master:
  case llvm::omp::Directive::OMPD_ordered:
  case llvm::omp::Directive::OMPD_parallel:
  case llvm::omp::Directive::OMPD_single:
  case llvm::omp::Directive::OMPD_target:
  case llvm::omp::Directive::OMPD_target_data:
  case llvm::omp::Directive::OMPD_task:
  case llvm::omp::Directive::OMPD_teams:
  case llvm::omp::Directive::OMPD_workshare:
  case llvm::omp::Directive::OMPD_parallel_workshare:
  case llvm::omp::Directive::OMPD_target_teams:
  case llvm::omp::Directive::OMPD_target_parallel:
    PushContext(beginDir.source, beginDir.v);
    break;
  default:
    // TODO others
    break;
  }
  ClearDataSharingAttributeObjects();
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
  const auto &clauseList{std::get<parser::OmpClauseList>(beginLoopDir.t)};
  switch (beginDir.v) {
  case llvm::omp::Directive::OMPD_distribute:
  case llvm::omp::Directive::OMPD_distribute_parallel_do:
  case llvm::omp::Directive::OMPD_distribute_parallel_do_simd:
  case llvm::omp::Directive::OMPD_distribute_simd:
  case llvm::omp::Directive::OMPD_do:
  case llvm::omp::Directive::OMPD_do_simd:
  case llvm::omp::Directive::OMPD_parallel_do:
  case llvm::omp::Directive::OMPD_parallel_do_simd:
  case llvm::omp::Directive::OMPD_simd:
  case llvm::omp::Directive::OMPD_target_parallel_do:
  case llvm::omp::Directive::OMPD_target_parallel_do_simd:
  case llvm::omp::Directive::OMPD_target_teams_distribute:
  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do:
  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd:
  case llvm::omp::Directive::OMPD_target_teams_distribute_simd:
  case llvm::omp::Directive::OMPD_target_simd:
  case llvm::omp::Directive::OMPD_taskloop:
  case llvm::omp::Directive::OMPD_taskloop_simd:
  case llvm::omp::Directive::OMPD_teams_distribute:
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_do:
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_do_simd:
  case llvm::omp::Directive::OMPD_teams_distribute_simd:
    PushContext(beginDir.source, beginDir.v);
    break;
  default:
    break;
  }
  ClearDataSharingAttributeObjects();
  SetContextAssociatedLoopLevel(GetAssociatedLoopLevelFromClauses(clauseList));
  PrivatizeAssociatedLoopIndex(x);
  return true;
}

void OmpAttributeVisitor::ResolveSeqLoopIndexInParallelOrTaskConstruct(
    const parser::Name &iv) {
  auto targetIt{dirContext_.rbegin()};
  for (;; ++targetIt) {
    if (targetIt == dirContext_.rend()) {
      return;
    }
    if (llvm::omp::parallelSet.test(targetIt->directive) ||
        llvm::omp::taskGeneratingSet.test(targetIt->directive)) {
      break;
    }
  }
  if (auto *symbol{ResolveOmp(iv, Symbol::Flag::OmpPrivate, targetIt->scope)}) {
    targetIt++;
    symbol->set(Symbol::Flag::OmpPreDetermined);
    iv.symbol = symbol; // adjust the symbol within region
    for (auto it{dirContext_.rbegin()}; it != targetIt; ++it) {
      AddToContextObjectWithDSA(*symbol, Symbol::Flag::OmpPrivate, *it);
    }
  }
}

// 2.15.1.1 Data-sharing Attribute Rules - Predetermined
//   - A loop iteration variable for a sequential loop in a parallel
//     or task generating construct is private in the innermost such
//     construct that encloses the loop
bool OmpAttributeVisitor::Pre(const parser::DoConstruct &x) {
  if (!dirContext_.empty() && GetContext().withinConstruct) {
    if (const auto &iv{GetLoopIndex(x)}; iv.symbol) {
      if (!iv.symbol->test(Symbol::Flag::OmpPreDetermined)) {
        ResolveSeqLoopIndexInParallelOrTaskConstruct(iv);
      } else {
        // TODO: conflict checks with explicitly determined DSA
      }
    }
  }
  return true;
}

std::int64_t OmpAttributeVisitor::GetAssociatedLoopLevelFromClauses(
    const parser::OmpClauseList &x) {
  std::int64_t orderedLevel{0};
  std::int64_t collapseLevel{0};
  for (const auto &clause : x.v) {
    if (const auto *orderedClause{
            std::get_if<parser::OmpClause::Ordered>(&clause.u)}) {
      if (const auto v{EvaluateInt64(context_, orderedClause->v)}) {
        orderedLevel = *v;
      }
    }
    if (const auto *collapseClause{
            std::get_if<parser::OmpClause::Collapse>(&clause.u)}) {
      if (const auto v{EvaluateInt64(context_, collapseClause->v)}) {
        collapseLevel = *v;
      }
    }
  }

  if (orderedLevel && (!collapseLevel || orderedLevel >= collapseLevel)) {
    return orderedLevel;
  } else if (!orderedLevel && collapseLevel) {
    return collapseLevel;
  } // orderedLevel < collapseLevel is an error handled in structural checks
  return 1; // default is outermost loop
}

// 2.15.1.1 Data-sharing Attribute Rules - Predetermined
//   - The loop iteration variable(s) in the associated do-loop(s) of a do,
//     parallel do, taskloop, or distribute construct is (are) private.
//   - The loop iteration variable in the associated do-loop of a simd construct
//     with just one associated do-loop is linear with a linear-step that is the
//     increment of the associated do-loop.
//   - The loop iteration variables in the associated do-loops of a simd
//     construct with multiple associated do-loops are lastprivate.
//
// TODO: revisit after semantics checks are completed for do-loop association of
//       collapse and ordered
void OmpAttributeVisitor::PrivatizeAssociatedLoopIndex(
    const parser::OpenMPLoopConstruct &x) {
  std::int64_t level{GetContext().associatedLoopLevel};
  if (level <= 0) {
    return;
  }
  Symbol::Flag ivDSA;
  if (!llvm::omp::simdSet.test(GetContext().directive)) {
    ivDSA = Symbol::Flag::OmpPrivate;
  } else if (level == 1) {
    ivDSA = Symbol::Flag::OmpLinear;
  } else {
    ivDSA = Symbol::Flag::OmpLastPrivate;
  }

  const auto &outer{std::get<std::optional<parser::DoConstruct>>(x.t)};
  for (const parser::DoConstruct *loop{&*outer}; loop && level > 0; --level) {
    // go through all the nested do-loops and resolve index variables
    const parser::Name &iv{GetLoopIndex(*loop)};
    if (auto *symbol{ResolveOmp(iv, ivDSA, currScope())}) {
      symbol->set(Symbol::Flag::OmpPreDetermined);
      iv.symbol = symbol; // adjust the symbol within region
      AddToContextObjectWithDSA(*symbol, ivDSA);
    }

    const auto &block{std::get<parser::Block>(loop->t)};
    const auto it{block.begin()};
    loop = it != block.end() ? GetDoConstructIf(*it) : nullptr;
  }
  CHECK(level == 0);
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPSectionsConstruct &x) {
  const auto &beginSectionsDir{
      std::get<parser::OmpBeginSectionsDirective>(x.t)};
  const auto &beginDir{
      std::get<parser::OmpSectionsDirective>(beginSectionsDir.t)};
  switch (beginDir.v) {
  case llvm::omp::Directive::OMPD_parallel_sections:
  case llvm::omp::Directive::OMPD_sections:
    PushContext(beginDir.source, beginDir.v);
    break;
  default:
    break;
  }
  ClearDataSharingAttributeObjects();
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPThreadprivate &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_threadprivate);
  const auto &list{std::get<parser::OmpObjectList>(x.t)};
  ResolveOmpObjectList(list, Symbol::Flag::OmpThreadprivate);
  return false;
}

void OmpAttributeVisitor::Post(const parser::OmpDefaultClause &x) {
  if (!dirContext_.empty()) {
    switch (x.v) {
    case parser::OmpDefaultClause::Type::Private:
      SetContextDefaultDSA(Symbol::Flag::OmpPrivate);
      break;
    case parser::OmpDefaultClause::Type::Firstprivate:
      SetContextDefaultDSA(Symbol::Flag::OmpFirstPrivate);
      break;
    case parser::OmpDefaultClause::Type::Shared:
      SetContextDefaultDSA(Symbol::Flag::OmpShared);
      break;
    case parser::OmpDefaultClause::Type::None:
      SetContextDefaultDSA(Symbol::Flag::OmpNone);
      break;
    }
  }
}

// For OpenMP constructs, check all the data-refs within the constructs
// and adjust the symbol for each Name if necessary
void OmpAttributeVisitor::Post(const parser::Name &name) {
  auto *symbol{name.symbol};
  if (symbol && !dirContext_.empty() && GetContext().withinConstruct) {
    if (!symbol->owner().IsDerivedType() && !symbol->has<ProcEntityDetails>() &&
        !IsObjectWithDSA(*symbol)) {
      // TODO: create a separate function to go through the rules for
      //       predetermined, explicitly determined, and implicitly
      //       determined data-sharing attributes (2.15.1.1).
      if (Symbol * found{currScope().FindSymbol(name.source)}) {
        if (symbol != found) {
          name.symbol = found; // adjust the symbol within region
        } else if (GetContext().defaultDSA == Symbol::Flag::OmpNone) {
          context_.Say(name.source,
              "The DEFAULT(NONE) clause requires that '%s' must be listed in "
              "a data-sharing attribute clause"_err_en_US,
              symbol->name());
        }
      }
    }
  } // within OpenMP construct
}

Symbol *OmpAttributeVisitor::ResolveOmpCommonBlockName(
    const parser::Name *name) {
  if (auto *prev{name
              ? GetContext().scope.parent().FindCommonBlock(name->source)
              : nullptr}) {
    name->symbol = prev;
    return prev;
  } else {
    return nullptr;
  }
}

void OmpAttributeVisitor::ResolveOmpObjectList(
    const parser::OmpObjectList &ompObjectList, Symbol::Flag ompFlag) {
  for (const auto &ompObject : ompObjectList.v) {
    ResolveOmpObject(ompObject, ompFlag);
  }
}

void OmpAttributeVisitor::ResolveOmpObject(
    const parser::OmpObject &ompObject, Symbol::Flag ompFlag) {
  std::visit(
      common::visitors{
          [&](const parser::Designator &designator) {
            if (const auto *name{GetDesignatorNameIfDataRef(designator)}) {
              if (auto *symbol{ResolveOmp(*name, ompFlag, currScope())}) {
                AddToContextObjectWithDSA(*symbol, ompFlag);
                if (dataSharingAttributeFlags.test(ompFlag)) {
                  CheckMultipleAppearances(*name, *symbol, ompFlag);
                }
              }
            } else {
              // Array sections to be changed to substrings as needed
              if (AnalyzeExpr(context_, designator)) {
                if (std::holds_alternative<parser::Substring>(designator.u)) {
                  context_.Say(designator.source,
                      "Substrings are not allowed on OpenMP "
                      "directives or clauses"_err_en_US);
                }
              }
              // other checks, more TBD
            }
          },
          [&](const parser::Name &name) { // common block
            if (auto *symbol{ResolveOmpCommonBlockName(&name)}) {
              CheckMultipleAppearances(
                  name, *symbol, Symbol::Flag::OmpCommonBlock);
              // 2.15.3 When a named common block appears in a list, it has the
              // same meaning as if every explicit member of the common block
              // appeared in the list
              for (auto &object : symbol->get<CommonBlockDetails>().objects()) {
                if (auto *resolvedObject{
                        ResolveOmp(*object, ompFlag, currScope())}) {
                  AddToContextObjectWithDSA(*resolvedObject, ompFlag);
                }
              }
            } else {
              context_.Say(name.source, // 2.15.3
                  "COMMON block must be declared in the same scoping unit "
                  "in which the OpenMP directive or clause appears"_err_en_US);
            }
          },
      },
      ompObject.u);
}

Symbol *OmpAttributeVisitor::ResolveOmp(
    const parser::Name &name, Symbol::Flag ompFlag, Scope &scope) {
  if (ompFlagsRequireNewSymbol.test(ompFlag)) {
    return DeclarePrivateAccessEntity(name, ompFlag, scope);
  } else {
    return DeclareOrMarkOtherAccessEntity(name, ompFlag);
  }
}

Symbol *OmpAttributeVisitor::ResolveOmp(
    Symbol &symbol, Symbol::Flag ompFlag, Scope &scope) {
  if (ompFlagsRequireNewSymbol.test(ompFlag)) {
    return DeclarePrivateAccessEntity(symbol, ompFlag, scope);
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
  if (HasDataSharingAttributeObject(*target) &&
      !WithMultipleAppearancesOmpException(symbol, ompFlag)) {
    context_.Say(name.source,
        "'%s' appears in more than one data-sharing clause "
        "on the same OpenMP directive"_err_en_US,
        name.ToString());
  } else {
    AddDataSharingAttributeObject(*target);
  }
}

void ResolveAccParts(
    SemanticsContext &context, const parser::ProgramUnit &node) {
  if (context.IsEnabled(common::LanguageFeature::OpenACC)) {
    AccAttributeVisitor{context}.Walk(node);
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

} // namespace Fortran::semantics
