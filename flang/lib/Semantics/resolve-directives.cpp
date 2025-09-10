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
#include "llvm/Frontend/OpenMP/OMP.h.inc"
#include "llvm/Support/Debug.h"
#include <list>
#include <map>

template <typename T>
static Fortran::semantics::Scope *GetScope(
    Fortran::semantics::SemanticsContext &context, const T &x) {
  std::optional<Fortran::parser::CharBlock> source{GetLastSource(x)};
  return source ? &context.FindScope(*source) : nullptr;
}

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
  std::optional<DirContext> GetContextIf() {
    return dirContext_.empty()
        ? std::nullopt
        : std::make_optional<DirContext>(dirContext_.back());
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
              x.name.thing, Symbol::Flag::AccPrivate, currScope())}) {
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

  void Post(const parser::Name &);

private:
  std::int64_t GetAssociatedLoopLevelFromClauses(const parser::AccClauseList &);

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

  void CheckAssociatedLoop(const parser::DoConstruct &);
  void ResolveAccObjectList(const parser::AccObjectList &, Symbol::Flag);
  void ResolveAccObject(const parser::AccObject &, Symbol::Flag);
  Symbol *ResolveAcc(const parser::Name &, Symbol::Flag, Scope &);
  Symbol *ResolveAcc(Symbol &, Symbol::Flag, Scope &);
  Symbol *ResolveName(const parser::Name &, bool parentScope = false);
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
  Scope *topScope_;
};

// Data-sharing and Data-mapping attributes for data-refs in OpenMP construct
class OmpAttributeVisitor : DirectiveAttributeVisitor<llvm::omp::Directive> {
public:
  explicit OmpAttributeVisitor(SemanticsContext &context)
      : DirectiveAttributeVisitor(context) {}

  template <typename A> void Walk(const A &x) { parser::Walk(x, *this); }
  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}

  template <typename A> bool Pre(const parser::Statement<A> &statement) {
    currentStatementSource_ = statement.source;
    // Keep track of the labels in all the labelled statements
    if (statement.label) {
      auto label{statement.label.value()};
      // Get the context to check if the labelled statement is in an
      // enclosing OpenMP construct
      std::optional<DirContext> thisContext{GetContextIf()};
      targetLabels_.emplace(
          label, std::make_pair(currentStatementSource_, thisContext));
      // Check if a statement that causes a jump to the 'label'
      // has already been encountered
      auto range{sourceLabels_.equal_range(label)};
      for (auto it{range.first}; it != range.second; ++it) {
        // Check if both the statement with 'label' and the statement that
        // causes a jump to the 'label' are in the same scope
        CheckLabelContext(it->second.first, currentStatementSource_,
            it->second.second, thisContext);
      }
    }
    return true;
  }

  bool Pre(const parser::InternalSubprogram &) {
    // Clear the labels being tracked in the previous scope
    ClearLabels();
    return true;
  }

  bool Pre(const parser::ModuleSubprogram &) {
    // Clear the labels being tracked in the previous scope
    ClearLabels();
    return true;
  }

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

  bool Pre(const parser::OmpMetadirectiveDirective &x) {
    PushContext(x.source, llvm::omp::Directive::OMPD_metadirective);
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
  void Post(const parser::OmpBeginLoopDirective &) {
    GetContext().withinConstruct = true;
  }
  bool Pre(const parser::DoConstruct &);

  bool Pre(const parser::OpenMPSectionsConstruct &);
  void Post(const parser::OpenMPSectionsConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPSectionConstruct &);
  void Post(const parser::OpenMPSectionConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPCriticalConstruct &critical);
  void Post(const parser::OpenMPCriticalConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPDeclareSimdConstruct &x) {
    PushContext(x.source, llvm::omp::Directive::OMPD_declare_simd);
    const auto &name{std::get<std::optional<parser::Name>>(x.t)};
    if (name) {
      ResolveOmpName(*name, Symbol::Flag::OmpDeclareSimd);
    }
    return true;
  }
  void Post(const parser::OpenMPDeclareSimdConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPDepobjConstruct &x) {
    PushContext(x.source, llvm::omp::Directive::OMPD_depobj);
    for (auto &arg : x.v.Arguments().v) {
      if (auto *locator{std::get_if<parser::OmpLocator>(&arg.u)}) {
        if (auto *object{std::get_if<parser::OmpObject>(&locator->u)}) {
          ResolveOmpObject(*object, Symbol::Flag::OmpDependObject);
        }
      }
    }
    return true;
  }
  void Post(const parser::OpenMPDepobjConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPFlushConstruct &x) {
    PushContext(x.source, llvm::omp::Directive::OMPD_flush);
    for (auto &arg : x.v.Arguments().v) {
      if (auto *locator{std::get_if<parser::OmpLocator>(&arg.u)}) {
        if (auto *object{std::get_if<parser::OmpObject>(&locator->u)}) {
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
    }
    return true;
  }
  void Post(const parser::OpenMPFlushConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPRequiresConstruct &x) {
    using Flags = WithOmpDeclarative::RequiresFlags;
    using Requires = WithOmpDeclarative::RequiresFlag;
    PushContext(x.source, llvm::omp::Directive::OMPD_requires);

    // Gather information from the clauses.
    Flags flags;
    std::optional<common::OmpMemoryOrderType> memOrder;
    for (const auto &clause : std::get<parser::OmpClauseList>(x.t).v) {
      flags |= common::visit(
          common::visitors{
              [&memOrder](
                  const parser::OmpClause::AtomicDefaultMemOrder &atomic) {
                memOrder = atomic.v.v;
                return Flags{};
              },
              [](const parser::OmpClause::ReverseOffload &) {
                return Flags{Requires::ReverseOffload};
              },
              [](const parser::OmpClause::UnifiedAddress &) {
                return Flags{Requires::UnifiedAddress};
              },
              [](const parser::OmpClause::UnifiedSharedMemory &) {
                return Flags{Requires::UnifiedSharedMemory};
              },
              [](const parser::OmpClause::DynamicAllocators &) {
                return Flags{Requires::DynamicAllocators};
              },
              [](const auto &) { return Flags{}; }},
          clause.u);
    }
    // Merge clauses into parents' symbols details.
    AddOmpRequiresToScope(currScope(), flags, memOrder);
    return true;
  }
  void Post(const parser::OpenMPRequiresConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPDeclareTargetConstruct &);
  void Post(const parser::OpenMPDeclareTargetConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPDeclareMapperConstruct &);
  void Post(const parser::OpenMPDeclareMapperConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPDeclareReductionConstruct &);
  void Post(const parser::OpenMPDeclareReductionConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPThreadprivate &);
  void Post(const parser::OpenMPThreadprivate &) { PopContext(); }

  bool Pre(const parser::OpenMPDeclarativeAllocate &);
  void Post(const parser::OpenMPDeclarativeAllocate &) { PopContext(); }

  bool Pre(const parser::OpenMPAssumeConstruct &);
  void Post(const parser::OpenMPAssumeConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPAtomicConstruct &);
  void Post(const parser::OpenMPAtomicConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPDispatchConstruct &);
  void Post(const parser::OpenMPDispatchConstruct &) { PopContext(); }

  bool Pre(const parser::OpenMPExecutableAllocate &);
  void Post(const parser::OpenMPExecutableAllocate &);

  bool Pre(const parser::OpenMPAllocatorsConstruct &);
  void Post(const parser::OpenMPAllocatorsConstruct &);

  bool Pre(const parser::OmpDeclareVariantDirective &x) {
    PushContext(x.source, llvm::omp::Directive::OMPD_declare_variant);
    return true;
  }
  void Post(const parser::OmpDeclareVariantDirective &) { PopContext(); };

  void Post(const parser::OmpObjectList &x) {
    // The objects from OMP clauses should have already been resolved,
    // except common blocks (the ResolveNamesVisitor does not visit
    // parser::Name, those are dealt with as members of other structures).
    // Iterate over elements of x, and resolve any common blocks that
    // are still unresolved.
    for (const parser::OmpObject &obj : x.v) {
      auto *name{std::get_if<parser::Name>(&obj.u)};
      if (name && !name->symbol) {
        Resolve(*name, currScope().MakeCommonBlock(name->source));
      }
    }
  }

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
    const auto &objectList{std::get<parser::OmpObjectList>(x.t)};
    ResolveOmpObjectList(objectList, Symbol::Flag::OmpAllocate);
    return false;
  }
  bool Pre(const parser::OmpClause::Firstprivate &x) {
    ResolveOmpObjectList(x.v, Symbol::Flag::OmpFirstPrivate);
    return false;
  }
  bool Pre(const parser::OmpClause::Lastprivate &x) {
    const auto &objList{std::get<parser::OmpObjectList>(x.v.t)};
    ResolveOmpObjectList(objList, Symbol::Flag::OmpLastPrivate);
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
    auto &objects{std::get<parser::OmpObjectList>(x.t)};
    ResolveOmpObjectList(objects, Symbol::Flag::OmpLinear);
    return false;
  }

  bool Pre(const parser::OmpClause::Uniform &x) {
    ResolveOmpNameList(x.v, Symbol::Flag::OmpUniform);
    return false;
  }

  bool Pre(const parser::OmpInReductionClause &x) {
    auto &objects{std::get<parser::OmpObjectList>(x.t)};
    ResolveOmpObjectList(objects, Symbol::Flag::OmpInReduction);
    return false;
  }

  bool Pre(const parser::OmpClause::Reduction &x) {
    const auto &objList{std::get<parser::OmpObjectList>(x.v.t)};
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
            if (!procRef->v.thing.component.symbol) {
              if (!ResolveName(&procRef->v.thing.component)) {
                createDummyProcSymbol(&procRef->v.thing.component);
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
    const auto &alignedNameList{std::get<parser::OmpObjectList>(x.t)};
    ResolveOmpObjectList(alignedNameList, Symbol::Flag::OmpAligned);
    return false;
  }

  bool Pre(const parser::OmpClause::Nontemporal &x) {
    const auto &nontemporalNameList{x.v};
    ResolveOmpNameList(nontemporalNameList, Symbol::Flag::OmpNontemporal);
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

  // Keep track of labels in the statements that causes jumps to target labels
  void Post(const parser::GotoStmt &gotoStmt) { CheckSourceLabel(gotoStmt.v); }
  void Post(const parser::ComputedGotoStmt &computedGotoStmt) {
    for (auto &label : std::get<std::list<parser::Label>>(computedGotoStmt.t)) {
      CheckSourceLabel(label);
    }
  }
  void Post(const parser::ArithmeticIfStmt &arithmeticIfStmt) {
    CheckSourceLabel(std::get<1>(arithmeticIfStmt.t));
    CheckSourceLabel(std::get<2>(arithmeticIfStmt.t));
    CheckSourceLabel(std::get<3>(arithmeticIfStmt.t));
  }
  void Post(const parser::AssignedGotoStmt &assignedGotoStmt) {
    for (auto &label : std::get<std::list<parser::Label>>(assignedGotoStmt.t)) {
      CheckSourceLabel(label);
    }
  }
  void Post(const parser::AltReturnSpec &altReturnSpec) {
    CheckSourceLabel(altReturnSpec.v);
  }
  void Post(const parser::ErrLabel &errLabel) { CheckSourceLabel(errLabel.v); }
  void Post(const parser::EndLabel &endLabel) { CheckSourceLabel(endLabel.v); }
  void Post(const parser::EorLabel &eorLabel) { CheckSourceLabel(eorLabel.v); }

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

    const auto &ompObjList{std::get<parser::OmpObjectList>(x.t)};
    for (const auto &ompObj : ompObjList.v) {
      common::visit(
          common::visitors{
              [&](const parser::Designator &designator) {
                if (const auto *name{
                        semantics::getDesignatorNameIfDataRef(designator)}) {
                  if (name->symbol) {
                    name->symbol->set(
                        ompFlag.value_or(Symbol::Flag::OmpMapStorage));
                    AddToContextObjectWithDSA(*name->symbol,
                        ompFlag.value_or(Symbol::Flag::OmpMapStorage));
                    if (semantics::IsAssumedSizeArray(*name->symbol)) {
                      context_.Say(designator.source,
                          "Assumed-size whole arrays may not appear on the %s "
                          "clause"_err_en_US,
                          "MAP");
                    }
                  }
                }
              },
              [&](const auto &name) {},
          },
          ompObj.u);

      ResolveOmpObject(ompObj, ompFlag.value_or(Symbol::Flag::OmpMapStorage));
    }
  }

  const parser::OmpClause *associatedClause{nullptr};
  void SetAssociatedClause(const parser::OmpClause *c) { associatedClause = c; }
  const parser::OmpClause *GetAssociatedClause() { return associatedClause; }

private:
  /// Given a vector of loop levels and a vector of corresponding clauses find
  /// the largest loop level and set the associated loop level to the found
  /// maximum. This is used for error handling to ensure that the number of
  /// affected loops is not larger that the number of available loops.
  std::int64_t SetAssociatedMaxClause(llvm::SmallVector<std::int64_t> &,
      llvm::SmallVector<const parser::OmpClause *> &);
  std::int64_t GetNumAffectedLoopsFromLoopConstruct(
      const parser::OpenMPLoopConstruct &);
  void CollectNumAffectedLoopsFromLoopConstruct(
      const parser::OpenMPLoopConstruct &, llvm::SmallVector<std::int64_t> &,
      llvm::SmallVector<const parser::OmpClause *> &);
  void CollectNumAffectedLoopsFromInnerLoopContruct(
      const parser::OpenMPLoopConstruct &, llvm::SmallVector<std::int64_t> &,
      llvm::SmallVector<const parser::OmpClause *> &);
  void CollectNumAffectedLoopsFromClauses(const parser::OmpClauseList &,
      llvm::SmallVector<std::int64_t> &,
      llvm::SmallVector<const parser::OmpClause *> &);

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

  std::vector<const parser::Name *> allocateNames_; // on one directive
  UnorderedSymbolSet privateDataSharingAttributeObjects_; // on one directive
  UnorderedSymbolSet stmtFunctionExprSymbols_;
  std::multimap<const parser::Label,
      std::pair<parser::CharBlock, std::optional<DirContext>>>
      sourceLabels_;
  std::map<const parser::Label,
      std::pair<parser::CharBlock, std::optional<DirContext>>>
      targetLabels_;
  parser::CharBlock currentStatementSource_;

  void AddAllocateName(const parser::Name *&object) {
    allocateNames_.push_back(object);
  }
  void ClearAllocateNames() { allocateNames_.clear(); }

  void AddPrivateDataSharingAttributeObjects(SymbolRef object) {
    privateDataSharingAttributeObjects_.insert(object);
  }
  void ClearPrivateDataSharingAttributeObjects() {
    privateDataSharingAttributeObjects_.clear();
  }

  // Predetermined DSA rules
  void PrivatizeAssociatedLoopIndexAndCheckLoopLevel(
      const parser::OpenMPLoopConstruct &);
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
  Symbol *ResolveName(const parser::Name *);
  Symbol *ResolveOmpObjectScope(const parser::Name *);
  Symbol *DeclareOrMarkOtherAccessEntity(const parser::Name &, Symbol::Flag);
  Symbol *DeclareOrMarkOtherAccessEntity(Symbol &, Symbol::Flag);
  void CheckMultipleAppearances(
      const parser::Name &, const Symbol &, Symbol::Flag);

  void CheckDataCopyingClause(
      const parser::Name &, const Symbol &, Symbol::Flag);
  void CheckAssocLoopLevel(std::int64_t level, const parser::OmpClause *clause);
  void CheckObjectIsPrivatizable(
      const parser::Name &, const Symbol &, Symbol::Flag);
  void CheckSourceLabel(const parser::Label &);
  void CheckLabelContext(const parser::CharBlock, const parser::CharBlock,
      std::optional<DirContext>, std::optional<DirContext>);
  void ClearLabels() {
    sourceLabels_.clear();
    targetLabels_.clear();
  };
  void CheckAllNamesInAllocateStmt(const parser::CharBlock &source,
      const parser::OmpObjectList &ompObjectList,
      const parser::AllocateStmt &allocate);
  void CheckNameInAllocateStmt(const parser::CharBlock &source,
      const parser::Name &ompObject, const parser::AllocateStmt &allocate);

  std::int64_t ordCollapseLevel{0};

  void AddOmpRequiresToScope(Scope &, WithOmpDeclarative::RequiresFlags,
      std::optional<common::OmpMemoryOrderType>);
  void IssueNonConformanceWarning(llvm::omp::Directive D,
      parser::CharBlock source, unsigned EmitFromVersion);

  void CreateImplicitSymbols(const Symbol *symbol);

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

template <typename T>
bool DirectiveAttributeVisitor<T>::HasDataSharingAttributeObject(
    const Symbol &object) {
  auto it{dataSharingAttributeObjects_.find(object)};
  return it != dataSharingAttributeObjects_.end();
}

template <typename T>
const parser::Name *DirectiveAttributeVisitor<T>::GetLoopIndex(
    const parser::DoConstruct &x) {
  using Bounds = parser::LoopControl::Bounds;
  if (x.GetLoopControl()) {
    if (const Bounds * b{std::get_if<Bounds>(&x.GetLoopControl()->u)}) {
      return &b->name.thing;
    } else {
      return nullptr;
    }
  } else {
    context_
        .Say(std::get<parser::Statement<parser::NonLabelDoStmt>>(x.t).source,
            "Loop control is not present in the DO LOOP"_err_en_US)
        .Attach(GetContext().directiveSource,
            "associated with the enclosing LOOP construct"_en_US);
    return nullptr;
  }
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
  CheckAssociatedLoop(*outer);
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

Symbol *AccAttributeVisitor::ResolveName(
    const parser::Name &name, bool parentScope) {
  Symbol *prev{currScope().FindSymbol(name.source)};
  // Check in parent scope if asked for.
  if (!prev && parentScope) {
    prev = currScope().parent().FindSymbol(name.source);
  }
  if (prev != name.symbol) {
    name.symbol = prev;
  }
  return prev;
}

Symbol *AccAttributeVisitor::ResolveFctName(const parser::Name &name) {
  Symbol *prev{currScope().FindSymbol(name.source)};
  if (!prev || (prev && prev->IsFuncResult())) {
    prev = currScope().parent().FindSymbol(name.source);
    if (!prev) {
      prev = &context_.globalScope().MakeSymbol(
          name.source, Attrs{}, ProcEntityDetails{});
    }
  }
  if (prev != name.symbol) {
    name.symbol = prev;
  }
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
  if (symbol.has<SubprogramDetails>()) {
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
                     std::get_if<Fortran::parser::AccClause::Gang>(
                         &clause.u)}) {
        for (auto &device : currentDevices) {
          device->set_isGang();
        }
        if (gangClause->v) {
          const Fortran::parser::AccGangArgList &x = *gangClause->v;
          int numArgs{0};
          for (const Fortran::parser::AccGangArg &gangArg : x.v) {
            CHECK(numArgs <= 1 && "expecting 0 or 1 gang dim args");
            if (const auto *dim{std::get_if<Fortran::parser::AccGangArg::Dim>(
                    &gangArg.u)}) {
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
                     std::get_if<Fortran::parser::AccClause::Bind>(
                         &clause.u)}) {
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
    symbol.get<SubprogramDetails>().add_openACCRoutineInfo(info);
  }
}

bool AccAttributeVisitor::Pre(const parser::OpenACCRoutineConstruct &x) {
  const auto &verbatim{std::get<parser::Verbatim>(x.t)};
  if (topScope_) {
    PushContext(
        verbatim.source, llvm::acc::Directive::ACCD_routine, *topScope_);
  } else {
    PushContext(verbatim.source, llvm::acc::Directive::ACCD_routine);
  }
  const auto &optName{std::get<std::optional<parser::Name>>(x.t)};
  if (optName) {
    if (Symbol *sym = ResolveFctName(*optName)) {
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
    PushContext(combinedDir.source, combinedDir.v);
    break;
  default:
    break;
  }
  const auto &clauseList{std::get<parser::AccClauseList>(beginBlockDir.t)};
  SetContextAssociatedLoopLevel(GetAssociatedLoopLevelFromClauses(clauseList));
  const auto &outer{std::get<std::optional<parser::DoConstruct>>(x.t)};
  CheckAssociatedLoop(*outer);
  ClearDataSharingAttributeObjects();
  return true;
}

static bool IsLastNameArray(const parser::Designator &designator) {
  const auto &name{GetLastName(designator)};
  const evaluate::DataRef dataRef{*(name.symbol)};
  return common::visit(
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
    const parser::DoConstruct &outerDoConstruct) {
  std::int64_t level{GetContext().associatedLoopLevel};
  if (level <= 0) { // collapse value was negative or 0
    return;
  }

  const auto getNextDoConstruct =
      [this](const parser::Block &block,
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
        break;
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
  using Bounds = parser::LoopControl::Bounds;
  for (const parser::DoConstruct *loop{&outerDoConstruct}; loop && level > 0;) {
    // Go through all nested loops to ensure index variable exists.
    if (const parser::Name * ivName{GetLoopIndex(*loop)}) {
      if (auto *symbol{ResolveAcc(*ivName, flag, currScope())}) {
        if (auto &control{loop->GetLoopControl()}) {
          if (const Bounds * b{std::get_if<Bounds>(&control->u)}) {
            if (auto lowerExpr{semantics::AnalyzeExpr(context_, b->lower)}) {
              semantics::UnorderedSymbolSet lowerSyms =
                  evaluate::CollectSymbols(*lowerExpr);
              checkExprHasSymbols(ivs, lowerSyms);
            }
            if (auto upperExpr{semantics::AnalyzeExpr(context_, b->upper)}) {
              semantics::UnorderedSymbolSet upperSyms =
                  evaluate::CollectSymbols(*upperExpr);
              checkExprHasSymbols(ivs, upperSyms);
            }
          }
        }
        ivs.push_back(symbol);
      }
    }

    const auto &block{std::get<parser::Block>(loop->t)};
    --level;
    loop = getNextDoConstruct(block, level);
  }
  CHECK(level == 0);
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

// For OpenACC constructs, check all the data-refs within the constructs
// and adjust the symbol for each Name if necessary
void AccAttributeVisitor::Post(const parser::Name &name) {
  auto *symbol{name.symbol};
  if (symbol && WithinConstruct()) {
    symbol = &symbol->GetUltimate();
    if (!symbol->owner().IsDerivedType() && !symbol->has<ProcEntityDetails>() &&
        !symbol->has<SubprogramDetails>() && !IsObjectWithVisibleDSA(*symbol)) {
      if (Symbol * found{currScope().FindSymbol(name.source)}) {
        if (symbol != found) {
          name.symbol = found; // adjust the symbol within region
        } else if (GetContext().defaultDSA == Symbol::Flag::AccNone) {
          // 2.5.14.
          context_.Say(name.source,
              "The DEFAULT(NONE) clause requires that '%s' must be listed in a data-mapping clause"_err_en_US,
              symbol->name());
        }
      }
    }
  } // within OpenACC construct
}

Symbol *AccAttributeVisitor::ResolveAccCommonBlockName(
    const parser::Name *name) {
  if (auto *prev{name
              ? GetContext().scope.parent().FindCommonBlock(name->source)
              : nullptr}) {
    name->symbol = prev;
    return prev;
  }
  // Check if the Common Block is declared in the current scope
  if (auto *commonBlockSymbol{
          name ? GetContext().scope.FindCommonBlock(name->source) : nullptr}) {
    name->symbol = commonBlockSymbol;
    return commonBlockSymbol;
  }
  return nullptr;
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
                    semantics::getDesignatorNameIfDataRef(designator)}) {
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
  return DeclareOrMarkOtherAccessEntity(name, accFlag);
}

Symbol *AccAttributeVisitor::ResolveAcc(
    Symbol &symbol, Symbol::Flag accFlag, Scope &scope) {
  return DeclareOrMarkOtherAccessEntity(symbol, accFlag);
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
  switch (dirId) {
  case llvm::omp::Directive::OMPD_masked:
  case llvm::omp::Directive::OMPD_parallel_masked:
  case llvm::omp::Directive::OMPD_master:
  case llvm::omp::Directive::OMPD_parallel_master:
  case llvm::omp::Directive::OMPD_ordered:
  case llvm::omp::Directive::OMPD_parallel:
  case llvm::omp::Directive::OMPD_scope:
  case llvm::omp::Directive::OMPD_single:
  case llvm::omp::Directive::OMPD_target:
  case llvm::omp::Directive::OMPD_target_data:
  case llvm::omp::Directive::OMPD_task:
  case llvm::omp::Directive::OMPD_taskgroup:
  case llvm::omp::Directive::OMPD_teams:
  case llvm::omp::Directive::OMPD_workdistribute:
  case llvm::omp::Directive::OMPD_workshare:
  case llvm::omp::Directive::OMPD_parallel_workshare:
  case llvm::omp::Directive::OMPD_target_teams:
  case llvm::omp::Directive::OMPD_target_teams_workdistribute:
  case llvm::omp::Directive::OMPD_target_parallel:
  case llvm::omp::Directive::OMPD_teams_workdistribute:
    PushContext(dirSpec.source, dirId);
    break;
  default:
    // TODO others
    break;
  }
  if (dirId == llvm::omp::Directive::OMPD_master ||
      dirId == llvm::omp::Directive::OMPD_parallel_master)
    IssueNonConformanceWarning(dirId, dirSpec.source, 52);
  ClearDataSharingAttributeObjects();
  ClearPrivateDataSharingAttributeObjects();
  ClearAllocateNames();
  return true;
}

void OmpAttributeVisitor::Post(const parser::OmpBlockConstruct &x) {
  const parser::OmpDirectiveSpecification &dirSpec{x.BeginDir()};
  llvm::omp::Directive dirId{dirSpec.DirId()};
  switch (dirId) {
  case llvm::omp::Directive::OMPD_masked:
  case llvm::omp::Directive::OMPD_master:
  case llvm::omp::Directive::OMPD_parallel_masked:
  case llvm::omp::Directive::OMPD_parallel_master:
  case llvm::omp::Directive::OMPD_parallel:
  case llvm::omp::Directive::OMPD_scope:
  case llvm::omp::Directive::OMPD_single:
  case llvm::omp::Directive::OMPD_target:
  case llvm::omp::Directive::OMPD_task:
  case llvm::omp::Directive::OMPD_teams:
  case llvm::omp::Directive::OMPD_workdistribute:
  case llvm::omp::Directive::OMPD_parallel_workshare:
  case llvm::omp::Directive::OMPD_target_teams:
  case llvm::omp::Directive::OMPD_target_parallel:
  case llvm::omp::Directive::OMPD_target_teams_workdistribute:
  case llvm::omp::Directive::OMPD_teams_workdistribute: {
    bool hasPrivate;
    for (const auto *allocName : allocateNames_) {
      hasPrivate = false;
      for (auto privateObj : privateDataSharingAttributeObjects_) {
        const Symbol &symbolPrivate{*privateObj};
        if (allocName->source == symbolPrivate.name()) {
          hasPrivate = true;
          break;
        }
      }
      if (!hasPrivate) {
        context_.Say(allocName->source,
            "The ALLOCATE clause requires that '%s' must be listed in a "
            "private "
            "data-sharing attribute clause on the same directive"_err_en_US,
            allocName->ToString());
      }
    }
    break;
  }
  default:
    break;
  }
  PopContext();
}

bool OmpAttributeVisitor::Pre(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  const auto &standaloneDir{std::get<parser::OmpDirectiveName>(x.v.t)};
  switch (standaloneDir.v) {
  case llvm::omp::Directive::OMPD_barrier:
  case llvm::omp::Directive::OMPD_ordered:
  case llvm::omp::Directive::OMPD_scan:
  case llvm::omp::Directive::OMPD_target_enter_data:
  case llvm::omp::Directive::OMPD_target_exit_data:
  case llvm::omp::Directive::OMPD_target_update:
  case llvm::omp::Directive::OMPD_taskwait:
  case llvm::omp::Directive::OMPD_taskyield:
    PushContext(standaloneDir.source, standaloneDir.v);
    break;
  default:
    break;
  }
  ClearDataSharingAttributeObjects();
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
  switch (beginDir.v) {
  case llvm::omp::Directive::OMPD_distribute:
  case llvm::omp::Directive::OMPD_distribute_parallel_do:
  case llvm::omp::Directive::OMPD_distribute_parallel_do_simd:
  case llvm::omp::Directive::OMPD_distribute_simd:
  case llvm::omp::Directive::OMPD_do:
  case llvm::omp::Directive::OMPD_do_simd:
  case llvm::omp::Directive::OMPD_loop:
  case llvm::omp::Directive::OMPD_masked_taskloop_simd:
  case llvm::omp::Directive::OMPD_masked_taskloop:
  case llvm::omp::Directive::OMPD_master_taskloop_simd:
  case llvm::omp::Directive::OMPD_master_taskloop:
  case llvm::omp::Directive::OMPD_parallel_do:
  case llvm::omp::Directive::OMPD_parallel_do_simd:
  case llvm::omp::Directive::OMPD_parallel_masked_taskloop_simd:
  case llvm::omp::Directive::OMPD_parallel_masked_taskloop:
  case llvm::omp::Directive::OMPD_parallel_master_taskloop_simd:
  case llvm::omp::Directive::OMPD_parallel_master_taskloop:
  case llvm::omp::Directive::OMPD_simd:
  case llvm::omp::Directive::OMPD_target_loop:
  case llvm::omp::Directive::OMPD_target_parallel_do:
  case llvm::omp::Directive::OMPD_target_parallel_do_simd:
  case llvm::omp::Directive::OMPD_target_parallel_loop:
  case llvm::omp::Directive::OMPD_target_teams_distribute:
  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do:
  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd:
  case llvm::omp::Directive::OMPD_target_teams_distribute_simd:
  case llvm::omp::Directive::OMPD_target_teams_loop:
  case llvm::omp::Directive::OMPD_target_simd:
  case llvm::omp::Directive::OMPD_taskloop:
  case llvm::omp::Directive::OMPD_taskloop_simd:
  case llvm::omp::Directive::OMPD_teams_distribute:
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_do:
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_do_simd:
  case llvm::omp::Directive::OMPD_teams_distribute_simd:
  case llvm::omp::Directive::OMPD_teams_loop:
  case llvm::omp::Directive::OMPD_tile:
  case llvm::omp::Directive::OMPD_unroll:
    PushContext(beginDir.source, beginDir.v);
    break;
  default:
    break;
  }
  if (beginDir.v == llvm::omp::OMPD_master_taskloop ||
      beginDir.v == llvm::omp::OMPD_master_taskloop_simd ||
      beginDir.v == llvm::omp::OMPD_parallel_master_taskloop ||
      beginDir.v == llvm::omp::OMPD_parallel_master_taskloop_simd ||
      beginDir.v == llvm::omp::Directive::OMPD_target_loop)
    IssueNonConformanceWarning(beginDir.v, beginDir.source, 52);
  ClearDataSharingAttributeObjects();
  SetContextAssociatedLoopLevel(GetNumAffectedLoopsFromLoopConstruct(x));

  if (beginDir.v == llvm::omp::Directive::OMPD_do) {
    auto &optLoopCons = std::get<std::optional<parser::NestedConstruct>>(x.t);
    if (optLoopCons.has_value()) {
      if (const auto &doConstruct{
              std::get_if<parser::DoConstruct>(&*optLoopCons)}) {
        if (doConstruct->IsDoWhile()) {
          return true;
        }
      }
    }
  }
  PrivatizeAssociatedLoopIndexAndCheckLoopLevel(x);
  ordCollapseLevel = GetNumAffectedLoopsFromLoopConstruct(x) + 1;
  return true;
}

void OmpAttributeVisitor::ResolveSeqLoopIndexInParallelOrTaskConstruct(
    const parser::Name &iv) {
  // Find the parallel or task generating construct enclosing the
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
  }
  // If this symbol already has a data-sharing attribute then there is nothing
  // to do here.
  if (const Symbol * symbol{iv.symbol}) {
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
    ordCollapseLevel--;
    for (auto iv : ivs) {
      if (!iv->symbol->test(Symbol::Flag::OmpPreDetermined)) {
        ResolveSeqLoopIndexInParallelOrTaskConstruct(*iv);
      } else {
        // TODO: conflict checks with explicitly determined DSA
      }
      if (ordCollapseLevel) {
        if (const auto *details{iv->symbol->detailsIf<HostAssocDetails>()}) {
          const Symbol *tpSymbol = &details->symbol();
          if (tpSymbol->test(Symbol::Flag::OmpThreadprivate)) {
            context_.Say(iv->source,
                "Loop iteration variable %s is not allowed in THREADPRIVATE."_err_en_US,
                iv->ToString());
          }
        }
      }
    }
  }
  return true;
}

static bool isSizesClause(const parser::OmpClause *clause) {
  return std::holds_alternative<parser::OmpClause::Sizes>(clause->u);
}

std::int64_t OmpAttributeVisitor::SetAssociatedMaxClause(
    llvm::SmallVector<std::int64_t> &levels,
    llvm::SmallVector<const parser::OmpClause *> &clauses) {

  // Find the tile level to ensure that the COLLAPSE clause value
  // does not exeed the number of tiled loops.
  std::int64_t tileLevel = 0;
  for (auto [level, clause] : llvm::zip_equal(levels, clauses))
    if (isSizesClause(clause))
      tileLevel = level;

  std::int64_t maxLevel = 1;
  const parser::OmpClause *maxClause = nullptr;
  for (auto [level, clause] : llvm::zip_equal(levels, clauses)) {
    if (tileLevel > 0 && tileLevel < level) {
      context_.Say(clause->source,
          "The value of the parameter in the COLLAPSE clause must"
          " not be larger than the number of the number of tiled loops"
          " because collapse currently is limited to independent loop"
          " iterations."_err_en_US);
      return 1;
    }

    if (level > maxLevel) {
      maxLevel = level;
      maxClause = clause;
    }
  }
  if (maxClause)
    SetAssociatedClause(maxClause);
  return maxLevel;
}

std::int64_t OmpAttributeVisitor::GetNumAffectedLoopsFromLoopConstruct(
    const parser::OpenMPLoopConstruct &x) {
  llvm::SmallVector<std::int64_t> levels;
  llvm::SmallVector<const parser::OmpClause *> clauses;

  CollectNumAffectedLoopsFromLoopConstruct(x, levels, clauses);
  return SetAssociatedMaxClause(levels, clauses);
}

void OmpAttributeVisitor::CollectNumAffectedLoopsFromLoopConstruct(
    const parser::OpenMPLoopConstruct &x,
    llvm::SmallVector<std::int64_t> &levels,
    llvm::SmallVector<const parser::OmpClause *> &clauses) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &clauseList{std::get<parser::OmpClauseList>(beginLoopDir.t)};

  CollectNumAffectedLoopsFromClauses(clauseList, levels, clauses);
  CollectNumAffectedLoopsFromInnerLoopContruct(x, levels, clauses);
}

void OmpAttributeVisitor::CollectNumAffectedLoopsFromInnerLoopContruct(
    const parser::OpenMPLoopConstruct &x,
    llvm::SmallVector<std::int64_t> &levels,
    llvm::SmallVector<const parser::OmpClause *> &clauses) {

  const auto &nestedOptional =
      std::get<std::optional<parser::NestedConstruct>>(x.t);
  assert(nestedOptional.has_value() &&
      "Expected a DoConstruct or OpenMPLoopConstruct");
  const auto *innerConstruct =
      std::get_if<common::Indirection<parser::OpenMPLoopConstruct>>(
          &(nestedOptional.value()));

  if (innerConstruct) {
    CollectNumAffectedLoopsFromLoopConstruct(
        innerConstruct->value(), levels, clauses);
  }
}

void OmpAttributeVisitor::CollectNumAffectedLoopsFromClauses(
    const parser::OmpClauseList &x, llvm::SmallVector<std::int64_t> &levels,
    llvm::SmallVector<const parser::OmpClause *> &clauses) {
  for (const auto &clause : x.v) {
    if (const auto oclause{
            std::get_if<parser::OmpClause::Ordered>(&clause.u)}) {
      std::int64_t level = 0;
      if (const auto v{EvaluateInt64(context_, oclause->v)}) {
        level = *v;
      }
      levels.push_back(level);
      clauses.push_back(&clause);
    }

    if (const auto cclause{
            std::get_if<parser::OmpClause::Collapse>(&clause.u)}) {
      std::int64_t level = 0;
      if (const auto v{EvaluateInt64(context_, cclause->v)}) {
        level = *v;
      }
      levels.push_back(level);
      clauses.push_back(&clause);
    }

    if (const auto tclause{std::get_if<parser::OmpClause::Sizes>(&clause.u)}) {
      levels.push_back(tclause->v.size());
      clauses.push_back(&clause);
    }
  }
}

// 2.15.1.1 Data-sharing Attribute Rules - Predetermined
//   - The loop iteration variable(s) in the associated do-loop(s) of a do,
//     parallel do, taskloop, or distribute construct is (are) private.
//   - The loop iteration variable in the associated do-loop of a simd construct
//     with just one associated do-loop is linear with a linear-step that is the
//     increment of the associated do-loop.
//   - The loop iteration variables in the associated do-loops of a simd
//     construct with multiple associated do-loops are lastprivate.
void OmpAttributeVisitor::PrivatizeAssociatedLoopIndexAndCheckLoopLevel(
    const parser::OpenMPLoopConstruct &x) {
  unsigned version{context_.langOptions().OpenMPVersion};
  std::int64_t level{GetContext().associatedLoopLevel};
  if (level <= 0) {
    return;
  }
  Symbol::Flag ivDSA;
  if (!llvm::omp::allSimdSet.test(GetContext().directive)) {
    ivDSA = Symbol::Flag::OmpPrivate;
  } else if (level == 1) {
    ivDSA = Symbol::Flag::OmpLinear;
  } else {
    ivDSA = Symbol::Flag::OmpLastPrivate;
  }

  bool isLoopConstruct{
      GetContext().directive == llvm::omp::Directive::OMPD_loop};
  const parser::OmpClause *clause{GetAssociatedClause()};
  bool hasCollapseClause{
      clause ? (clause->Id() == llvm::omp::OMPC_collapse) : false};
  const parser::OpenMPLoopConstruct *innerMostLoop = &x;
  const parser::NestedConstruct *innerMostNest = nullptr;
  while (auto &optLoopCons{
      std::get<std::optional<parser::NestedConstruct>>(innerMostLoop->t)}) {
    innerMostNest = &(optLoopCons.value());
    if (const auto *innerLoop{
            std::get_if<common::Indirection<parser::OpenMPLoopConstruct>>(
                innerMostNest)}) {
      innerMostLoop = &(innerLoop->value());
    } else
      break;
  }

  if (innerMostNest) {
    if (const auto &outer{std::get_if<parser::DoConstruct>(innerMostNest)}) {
      for (const parser::DoConstruct *loop{&*outer}; loop && level > 0;
          --level) {
        if (loop->IsDoConcurrent()) {
          // DO CONCURRENT is explicitly allowed for the LOOP construct so long
          // as there isn't a COLLAPSE clause
          if (isLoopConstruct) {
            if (hasCollapseClause) {
              // hasCollapseClause implies clause != nullptr
              context_.Say(clause->source,
                  "DO CONCURRENT loops cannot be used with the COLLAPSE clause."_err_en_US);
            }
          } else {
            auto &stmt =
                std::get<parser::Statement<parser::NonLabelDoStmt>>(loop->t);
            context_.Say(stmt.source,
                "DO CONCURRENT loops cannot form part of a loop nest."_err_en_US);
          }
        }
        // go through all the nested do-loops and resolve index variables
        const parser::Name *iv{GetLoopIndex(*loop)};
        if (iv) {
          if (auto *symbol{ResolveOmp(*iv, ivDSA, currScope())}) {
            SetSymbolDSA(*symbol, {Symbol::Flag::OmpPreDetermined, ivDSA});
            iv->symbol = symbol; // adjust the symbol within region
            AddToContextObjectWithDSA(*symbol, ivDSA);
          }

          const auto &block{std::get<parser::Block>(loop->t)};
          const auto it{block.begin()};
          loop = it != block.end() ? GetDoConstructIf(*it) : nullptr;
        }
      }
      CheckAssocLoopLevel(level, GetAssociatedClause());
    } else if (const auto &loop{std::get_if<
                   common::Indirection<parser::OpenMPLoopConstruct>>(
                   innerMostNest)}) {
      auto &beginDirective =
          std::get<parser::OmpBeginLoopDirective>(loop->value().t);
      auto &beginLoopDirective =
          std::get<parser::OmpLoopDirective>(beginDirective.t);
      if (beginLoopDirective.v != llvm::omp::Directive::OMPD_unroll &&
          beginLoopDirective.v != llvm::omp::Directive::OMPD_tile) {
        context_.Say(GetContext().directiveSource,
            "Only UNROLL or TILE constructs are allowed between an OpenMP Loop Construct and a DO construct"_err_en_US,
            parser::ToUpperCaseLetters(llvm::omp::getOpenMPDirectiveName(
                GetContext().directive, version)
                    .str()));
      } else {
        PrivatizeAssociatedLoopIndexAndCheckLoopLevel(loop->value());
      }
    } else {
      context_.Say(GetContext().directiveSource,
          "A DO loop must follow the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(
              llvm::omp::getOpenMPDirectiveName(GetContext().directive, version)
                  .str()));
    }
  }
}

void OmpAttributeVisitor::CheckAssocLoopLevel(
    std::int64_t level, const parser::OmpClause *clause) {
  if (clause && level != 0) {
    context_.Say(clause->source,
        "The value of the parameter in the COLLAPSE or ORDERED clause must"
        " not be larger than the number of nested loops"
        " following the construct."_err_en_US);
  }
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPGroupprivate &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_groupprivate);
  for (const parser::OmpArgument &arg : x.v.Arguments().v) {
    if (auto *locator{std::get_if<parser::OmpLocator>(&arg.u)}) {
      if (auto *object{std::get_if<parser::OmpObject>(&locator->u)}) {
        ResolveOmpObject(*object, Symbol::Flag::OmpGroupPrivate);
      }
    }
  }
  return true;
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
  PushContext(beginSpec.DirName().source, beginSpec.DirName().v);
  GetContext().withinConstruct = true;
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPDeclareTargetConstruct &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_declare_target);
  const auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
  if (const auto *objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
    ResolveOmpObjectList(*objectList, Symbol::Flag::OmpDeclareTarget);
  } else if (const auto *clauseList{
                 parser::Unwrap<parser::OmpClauseList>(spec.u)}) {
    for (const auto &clause : clauseList->v) {
      if (const auto *toClause{std::get_if<parser::OmpClause::To>(&clause.u)}) {
        auto &objList{std::get<parser::OmpObjectList>(toClause->v.t)};
        ResolveOmpObjectList(objList, Symbol::Flag::OmpDeclareTarget);
      } else if (const auto *linkClause{
                     std::get_if<parser::OmpClause::Link>(&clause.u)}) {
        ResolveOmpObjectList(linkClause->v, Symbol::Flag::OmpDeclareTarget);
      } else if (const auto *enterClause{
                     std::get_if<parser::OmpClause::Enter>(&clause.u)}) {
        ResolveOmpObjectList(std::get<parser::OmpObjectList>(enterClause->v.t),
            Symbol::Flag::OmpDeclareTarget);
      }
    }
  }
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPDeclareMapperConstruct &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_declare_mapper);
  return true;
}

bool OmpAttributeVisitor::Pre(
    const parser::OpenMPDeclareReductionConstruct &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_declare_reduction);
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPThreadprivate &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_threadprivate);
  const auto &list{std::get<parser::OmpObjectList>(x.t)};
  ResolveOmpObjectList(list, Symbol::Flag::OmpThreadprivate);
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPDeclarativeAllocate &x) {
  PushContext(x.source, llvm::omp::Directive::OMPD_allocate);
  const auto &list{std::get<parser::OmpObjectList>(x.t)};
  ResolveOmpObjectList(list, Symbol::Flag::OmpDeclarativeAllocateDirective);
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

bool OmpAttributeVisitor::Pre(const parser::OpenMPExecutableAllocate &x) {
  IssueNonConformanceWarning(llvm::omp::Directive::OMPD_allocate, x.source, 52);

  PushContext(x.source, llvm::omp::Directive::OMPD_allocate);
  const auto &list{std::get<std::optional<parser::OmpObjectList>>(x.t)};
  if (list) {
    ResolveOmpObjectList(*list, Symbol::Flag::OmpExecutableAllocateDirective);
  }
  return true;
}

bool OmpAttributeVisitor::Pre(const parser::OpenMPAllocatorsConstruct &x) {
  const parser::OmpDirectiveSpecification &dirSpec{x.BeginDir()};
  PushContext(x.source, dirSpec.DirId());

  for (const auto &clause : dirSpec.Clauses().v) {
    if (const auto *allocClause{
            std::get_if<parser::OmpClause::Allocate>(&clause.u)}) {
      ResolveOmpObjectList(std::get<parser::OmpObjectList>(allocClause->v.t),
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

void OmpAttributeVisitor::Post(const parser::OpenMPExecutableAllocate &x) {
  bool hasAllocator = false;
  // TODO: Investigate whether searching the clause list can be done with
  // parser::Unwrap instead of the following loop
  const auto &clauseList{std::get<parser::OmpClauseList>(x.t)};
  for (const auto &clause : clauseList.v) {
    if (std::get_if<parser::OmpClause::Allocator>(&clause.u)) {
      hasAllocator = true;
    }
  }

  if (IsNestedInDirective(llvm::omp::Directive::OMPD_target) && !hasAllocator) {
    // TODO: expand this check to exclude the case when a requires
    //       directive with the dynamic_allocators clause is present
    //       in the same compilation unit (OMP5.0 2.11.3).
    context_.Say(x.source,
        "ALLOCATE directives that appear in a TARGET region "
        "must specify an allocator clause"_err_en_US);
  }

  const auto &allocateStmt =
      std::get<parser::Statement<parser::AllocateStmt>>(x.t).statement;
  if (const auto &list{std::get<std::optional<parser::OmpObjectList>>(x.t)}) {
    CheckAllNamesInAllocateStmt(
        std::get<parser::Verbatim>(x.t).source, *list, allocateStmt);
  }
  if (const auto &subDirs{
          std::get<std::optional<std::list<parser::OpenMPDeclarativeAllocate>>>(
              x.t)}) {
    for (const auto &dalloc : *subDirs) {
      CheckAllNamesInAllocateStmt(std::get<parser::Verbatim>(dalloc.t).source,
          std::get<parser::OmpObjectList>(dalloc.t), allocateStmt);
    }
  }
  PopContext();
}

void OmpAttributeVisitor::Post(const parser::OpenMPAllocatorsConstruct &x) {
  const parser::OmpDirectiveSpecification &dirSpec{x.BeginDir()};
  auto &block{std::get<parser::Block>(x.t)};

  omp::SourcedActionStmt action{omp::GetActionStmt(block)};
  const parser::AllocateStmt *allocate{[&]() {
    if (action) {
      if (auto *alloc{std::get_if<common::Indirection<parser::AllocateStmt>>(
              &action.stmt->u)}) {
        return &alloc->value();
      }
    }
    return static_cast<const parser::AllocateStmt *>(nullptr);
  }()};

  if (allocate) {
    for (const auto &clause : dirSpec.Clauses().v) {
      if (auto *alloc{std::get_if<parser::OmpClause::Allocate>(&clause.u)}) {
        CheckAllNamesInAllocateStmt(
            x.source, std::get<parser::OmpObjectList>(alloc->v.t), *allocate);

        using OmpAllocatorSimpleModifier = parser::OmpAllocatorSimpleModifier;
        using OmpAllocatorComplexModifier = parser::OmpAllocatorComplexModifier;

        auto &modifiers{OmpGetModifiers(alloc->v)};
        bool hasAllocator{
            OmpGetUniqueModifier<OmpAllocatorSimpleModifier>(modifiers) ||
            OmpGetUniqueModifier<OmpAllocatorComplexModifier>(modifiers)};

        // TODO: As with allocate directive, exclude the case when a requires
        //       directive with the dynamic_allocators clause is present in
        //       the same compilation unit (OMP5.0 2.11.3).
        if (IsNestedInDirective(llvm::omp::Directive::OMPD_target) &&
            !hasAllocator) {
          context_.Say(x.source,
              "ALLOCATORS directives that appear in a TARGET region "
              "must specify an allocator"_err_en_US);
        }
      }
    }
  }
  PopContext();
}

static bool IsPrivatizable(const Symbol *sym) {
  auto *misc{sym->detailsIf<MiscDetails>()};
  return IsVariableName(*sym) && !IsProcedure(*sym) && !IsNamedConstant(*sym) &&
      ( // OpenMP 5.2, 5.1.1: Assumed-size arrays are shared
          !semantics::IsAssumedSizeArray(*sym) ||
          // If CrayPointer is among the DSA list then the
          // CrayPointee is Privatizable
          sym->test(Symbol::Flag::CrayPointee)) &&
      !sym->owner().IsDerivedType() &&
      sym->owner().kind() != Scope::Kind::ImpliedDos &&
      sym->owner().kind() != Scope::Kind::Forall &&
      !sym->detailsIf<semantics::AssocEntityDetails>() &&
      !sym->detailsIf<semantics::NamelistDetails>() &&
      (!misc ||
          (misc->kind() != MiscDetails::Kind::ComplexPartRe &&
              misc->kind() != MiscDetails::Kind::ComplexPartIm &&
              misc->kind() != MiscDetails::Kind::KindParamInquiry &&
              misc->kind() != MiscDetails::Kind::LenParamInquiry &&
              misc->kind() != MiscDetails::Kind::ConstructName));
}

static bool IsSymbolStaticStorageDuration(const Symbol &symbol) {
  LLVM_DEBUG(llvm::dbgs() << "IsSymbolStaticStorageDuration(" << symbol.name()
                          << "):\n");
  auto ultSym = symbol.GetUltimate();
  // Module-scope variable
  return (ultSym.owner().kind() == Scope::Kind::Module) ||
      // Data statement variable
      (ultSym.flags().test(Symbol::Flag::InDataStmt)) ||
      // Save attribute variable
      (ultSym.attrs().test(Attr::SAVE)) ||
      // Referenced in a common block
      (ultSym.flags().test(Symbol::Flag::InCommonBlock));
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

void OmpAttributeVisitor::CreateImplicitSymbols(const Symbol *symbol) {
  if (!IsPrivatizable(symbol)) {
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "CreateImplicitSymbols: " << *symbol << '\n');

  // Implicitly determined DSAs
  // OMP 5.2 5.1.1 - Variables Referenced in a Construct
  Symbol *lastDeclSymbol = nullptr;
  Symbol::Flags prevDSA;
  for (int dirDepth{0}; dirDepth < (int)dirContext_.size(); ++dirDepth) {
    DirContext &dirContext = dirContext_[dirDepth];
    Symbol::Flags dsa;

    Scope &scope{context_.FindScope(dirContext.directiveSource)};
    auto it{scope.find(symbol->name())};
    if (it != scope.end()) {
      // There is already a symbol in the current scope, use its DSA.
      dsa = GetSymbolDSA(*it->second);
    } else {
      for (auto symMap : dirContext.objectWithDSA) {
        if (symMap.first->name() == symbol->name()) {
          // `symbol` already has a data-sharing attribute in the current
          // context, use it.
          dsa.set(symMap.second);
          break;
        }
      }
    }

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
    bool isStaticStorageDuration = IsSymbolStaticStorageDuration(*symbol);

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

// For OpenMP constructs, check all the data-refs within the constructs
// and adjust the symbol for each Name if necessary
void OmpAttributeVisitor::Post(const parser::Name &name) {
  auto *symbol{name.symbol};

  if (symbol && WithinConstruct()) {
    if (IsPrivatizable(symbol) && !IsObjectWithDSA(*symbol)) {
      // TODO: create a separate function to go through the rules for
      //       predetermined, explicitly determined, and implicitly
      //       determined data-sharing attributes (2.15.1.1).
      if (Symbol * found{currScope().FindSymbol(name.source)}) {
        if (symbol != found) {
          name.symbol = found; // adjust the symbol within region
        } else if (GetContext().defaultDSA == Symbol::Flag::OmpNone &&
            !symbol->GetUltimate().test(Symbol::Flag::OmpThreadprivate) &&
            // Exclude indices of sequential loops that are privatised in
            // the scope of the parallel region, and not in this scope.
            // TODO: check whether this should be caught in IsObjectWithDSA
            !symbol->test(Symbol::Flag::OmpPrivate)) {
          if (symbol->GetUltimate().test(Symbol::Flag::CrayPointee)) {
            std::string crayPtrName{
                semantics::GetCrayPointer(*symbol).name().ToString()};
            if (!IsObjectWithDSA(*currScope().FindSymbol(crayPtrName)))
              context_.Say(name.source,
                  "The DEFAULT(NONE) clause requires that the Cray Pointer '%s' must be listed in a data-sharing attribute clause"_err_en_US,
                  crayPtrName);
          } else {
            context_.Say(name.source,
                "The DEFAULT(NONE) clause requires that '%s' must be listed in a data-sharing attribute clause"_err_en_US,
                symbol->name());
          }
        }
      }
    }

    if (Symbol * found{currScope().FindSymbol(name.source)}) {
      if (found->GetUltimate().test(semantics::Symbol::Flag::OmpThreadprivate))
        return;
    }

    CreateImplicitSymbols(symbol);
  } // within OpenMP construct
}

Symbol *OmpAttributeVisitor::ResolveName(const parser::Name *name) {
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

// Use this function over ResolveOmpName when an omp object's scope needs
// resolving, it's symbol flag isn't important and a simple check for resolution
// failure is desired. Using ResolveOmpName means needing to work with the
// context to check for failure, whereas here a pointer comparison is all that's
// needed.
Symbol *OmpAttributeVisitor::ResolveOmpObjectScope(const parser::Name *name) {

  // TODO: Investigate whether the following block can be replaced by, or
  // included in, the ResolveOmpName function
  if (auto *prev{name ? GetContext().scope.parent().FindSymbol(name->source)
                      : nullptr}) {
    name->symbol = prev;
    return nullptr;
  }

  // TODO: Investigate whether the following block can be replaced by, or
  // included in, the ResolveOmpName function
  if (auto *ompSymbol{
          name ? GetContext().scope.FindSymbol(name->source) : nullptr}) {
    name->symbol = ompSymbol;
    return ompSymbol;
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

  const auto *name{semantics::getDesignatorNameIfDataRef(designator)};
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
                parser::ToUpperCaseLetters(
                    llvm::omp::getOpenMPDirectiveName(directive, version)));
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

      if (ompFlag == Symbol::Flag::OmpAllocate) {
        AddAllocateName(name);
      }
    }
    if (ompFlag == Symbol::Flag::OmpDeclarativeAllocateDirective &&
        IsAllocatable(*symbol) &&
        !IsNestedInDirective(llvm::omp::Directive::OMPD_allocate)) {
      context_.Say(designator.source,
          "List items specified in the ALLOCATE directive must not have the ALLOCATABLE attribute unless the directive is associated with an ALLOCATE statement"_err_en_US);
    }
    if ((ompFlag == Symbol::Flag::OmpDeclarativeAllocateDirective ||
            ompFlag == Symbol::Flag::OmpExecutableAllocateDirective) &&
        ResolveOmpObjectScope(name) == nullptr) {
      context_.Say(designator.source, // 2.15.3
          "List items must be declared in the same scoping unit in which the %s directive appears"_err_en_US,
          parser::ToUpperCaseLetters(
              llvm::omp::getOpenMPDirectiveName(directive, version)));
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
      }
    }
  } else {
    context_.Say(name.source, // 2.15.3
        "COMMON block must be declared in the same scoping unit in which the OpenMP directive or clause appears"_err_en_US);
  }
}

void OmpAttributeVisitor::ResolveOmpObject(
    const parser::OmpObject &ompObject, Symbol::Flag ompFlag) {
  common::visit(common::visitors{
                    [&](const parser::Designator &designator) {
                      ResolveOmpDesignator(designator, ompFlag);
                    },
                    [&](const parser::Name &name) { // common block
                      ResolveOmpCommonBlock(name, ompFlag);
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
    if (privateDataSharingAttributeFlags.test(ompFlag)) {
      AddPrivateDataSharingAttributeObjects(*target);
    }
  }
}

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

void ResolveOmpTopLevelParts(
    SemanticsContext &context, const parser::Program &program) {
  if (!context.IsEnabled(common::LanguageFeature::OpenMP)) {
    return;
  }

  // Gather REQUIRES clauses from all non-module top-level program unit symbols,
  // combine them together ensuring compatibility and apply them to all these
  // program units. Modules are skipped because their REQUIRES clauses should be
  // propagated via USE statements instead.
  WithOmpDeclarative::RequiresFlags combinedFlags;
  std::optional<common::OmpMemoryOrderType> combinedMemOrder;

  // Function to go through non-module top level program units and extract
  // REQUIRES information to be processed by a function-like argument.
  auto processProgramUnits{[&](auto processFn) {
    for (const parser::ProgramUnit &unit : program.v) {
      if (!std::holds_alternative<common::Indirection<parser::Module>>(
              unit.u) &&
          !std::holds_alternative<common::Indirection<parser::Submodule>>(
              unit.u) &&
          !std::holds_alternative<
              common::Indirection<parser::CompilerDirective>>(unit.u)) {
        Symbol *symbol{common::visit(
            [&context](auto &x) {
              Scope *scope = GetScope(context, x.value());
              return scope ? scope->symbol() : nullptr;
            },
            unit.u)};
        // FIXME There is no symbol defined for MainProgram units in certain
        // circumstances, so REQUIRES information has no place to be stored in
        // these cases.
        if (!symbol) {
          continue;
        }
        common::visit(
            [&](auto &details) {
              if constexpr (std::is_convertible_v<decltype(&details),
                                WithOmpDeclarative *>) {
                processFn(*symbol, details);
              }
            },
            symbol->details());
      }
    }
  }};

  // Combine global REQUIRES information from all program units except modules
  // and submodules.
  processProgramUnits([&](Symbol &symbol, WithOmpDeclarative &details) {
    if (const WithOmpDeclarative::RequiresFlags *
        flags{details.ompRequires()}) {
      combinedFlags |= *flags;
    }
    if (const common::OmpMemoryOrderType *
        memOrder{details.ompAtomicDefaultMemOrder()}) {
      if (combinedMemOrder && *combinedMemOrder != *memOrder) {
        context.Say(symbol.scope()->sourceRange(),
            "Conflicting '%s' REQUIRES clauses found in compilation "
            "unit"_err_en_US,
            parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(
                llvm::omp::Clause::OMPC_atomic_default_mem_order)
                                           .str()));
      }
      combinedMemOrder = *memOrder;
    }
  });

  // Update all program units except modules and submodules with the combined
  // global REQUIRES information.
  processProgramUnits([&](Symbol &, WithOmpDeclarative &details) {
    if (combinedFlags.any()) {
      details.set_ompRequires(combinedFlags);
    }
    if (combinedMemOrder) {
      details.set_ompAtomicDefaultMemOrder(*combinedMemOrder);
    }
  });
}

static bool IsSymbolThreadprivate(const Symbol &symbol) {
  if (const auto *details{symbol.detailsIf<HostAssocDetails>()}) {
    return details->symbol().test(Symbol::Flag::OmpThreadprivate);
  }
  return symbol.test(Symbol::Flag::OmpThreadprivate);
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

void OmpAttributeVisitor::CheckSourceLabel(const parser::Label &label) {
  // Get the context to check if the statement causing a jump to the 'label' is
  // in an enclosing OpenMP construct
  std::optional<DirContext> thisContext{GetContextIf()};
  sourceLabels_.emplace(
      label, std::make_pair(currentStatementSource_, thisContext));
  // Check if the statement with 'label' to which a jump is being introduced
  // has already been encountered
  auto it{targetLabels_.find(label)};
  if (it != targetLabels_.end()) {
    // Check if both the statement with 'label' and the statement that causes a
    // jump to the 'label' are in the same scope
    CheckLabelContext(currentStatementSource_, it->second.first, thisContext,
        it->second.second);
  }
}

// Check for invalid branch into or out of OpenMP structured blocks
void OmpAttributeVisitor::CheckLabelContext(const parser::CharBlock source,
    const parser::CharBlock target, std::optional<DirContext> sourceContext,
    std::optional<DirContext> targetContext) {
  auto dirContextsSame = [](DirContext &lhs, DirContext &rhs) -> bool {
    // Sometimes nested constructs share a scope but are different contexts.
    // The directiveSource comparison is for OmpSection. Sections do not have
    // their own scopes and two different sections both have the same directive.
    // Their source however is different. This string comparison is unfortunate
    // but should only happen for GOTOs inside of SECTION.
    return (lhs.scope == rhs.scope) && (lhs.directive == rhs.directive) &&
        (lhs.directiveSource == rhs.directiveSource);
  };
  unsigned version{context_.langOptions().OpenMPVersion};
  if (targetContext &&
      (!sourceContext ||
          (!dirContextsSame(*targetContext, *sourceContext) &&
              !DoesScopeContain(
                  &targetContext->scope, sourceContext->scope)))) {
    context_
        .Say(source, "invalid branch into an OpenMP structured block"_err_en_US)
        .Attach(target, "In the enclosing %s directive branched into"_en_US,
            parser::ToUpperCaseLetters(llvm::omp::getOpenMPDirectiveName(
                targetContext->directive, version)
                    .str()));
  }
  if (sourceContext &&
      (!targetContext ||
          (!dirContextsSame(*sourceContext, *targetContext) &&
              !DoesScopeContain(
                  &sourceContext->scope, targetContext->scope)))) {
    context_
        .Say(source,
            "invalid branch leaving an OpenMP structured block"_err_en_US)
        .Attach(target, "Outside the enclosing %s directive"_en_US,
            parser::ToUpperCaseLetters(llvm::omp::getOpenMPDirectiveName(
                sourceContext->directive, version)
                    .str()));
  }
}

// Goes through the names in an OmpObjectList and checks if each name appears
// in the given allocate statement
void OmpAttributeVisitor::CheckAllNamesInAllocateStmt(
    const parser::CharBlock &source, const parser::OmpObjectList &ompObjectList,
    const parser::AllocateStmt &allocate) {
  for (const auto &obj : ompObjectList.v) {
    if (const auto *d{std::get_if<parser::Designator>(&obj.u)}) {
      if (const auto *ref{std::get_if<parser::DataRef>(&d->u)}) {
        if (const auto *n{std::get_if<parser::Name>(&ref->u)}) {
          CheckNameInAllocateStmt(source, *n, allocate);
        }
      }
    }
  }
}

void OmpAttributeVisitor::CheckNameInAllocateStmt(
    const parser::CharBlock &source, const parser::Name &name,
    const parser::AllocateStmt &allocate) {
  for (const auto &allocation :
      std::get<std::list<parser::Allocation>>(allocate.t)) {
    const auto &allocObj = std::get<parser::AllocateObject>(allocation.t);
    if (const auto *n{std::get_if<parser::Name>(&allocObj.u)}) {
      if (n->source == name.source) {
        return;
      }
    }
  }
  unsigned version{context_.langOptions().OpenMPVersion};
  context_.Say(source,
      "Object '%s' in %s directive not "
      "found in corresponding ALLOCATE statement"_err_en_US,
      name.ToString(),
      parser::ToUpperCaseLetters(
          llvm::omp::getOpenMPDirectiveName(GetContext().directive, version)
              .str()));
}

void OmpAttributeVisitor::AddOmpRequiresToScope(Scope &scope,
    WithOmpDeclarative::RequiresFlags flags,
    std::optional<common::OmpMemoryOrderType> memOrder) {
  Scope *scopeIter = &scope;
  do {
    if (Symbol * symbol{scopeIter->symbol()}) {
      common::visit(
          [&](auto &details) {
            // Store clauses information into the symbol for the parent and
            // enclosing modules, programs, functions and subroutines.
            if constexpr (std::is_convertible_v<decltype(&details),
                              WithOmpDeclarative *>) {
              if (flags.any()) {
                if (const WithOmpDeclarative::RequiresFlags *
                    otherFlags{details.ompRequires()}) {
                  flags |= *otherFlags;
                }
                details.set_ompRequires(flags);
              }
              if (memOrder) {
                if (details.has_ompAtomicDefaultMemOrder() &&
                    *details.ompAtomicDefaultMemOrder() != *memOrder) {
                  context_.Say(scopeIter->sourceRange(),
                      "Conflicting '%s' REQUIRES clauses found in compilation "
                      "unit"_err_en_US,
                      parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(
                          llvm::omp::Clause::OMPC_atomic_default_mem_order)
                                                     .str()));
                }
                details.set_ompAtomicDefaultMemOrder(*memOrder);
              }
            }
          },
          symbol->details());
    }
    scopeIter = &scopeIter->parent();
  } while (!scopeIter->IsGlobal());
}

void OmpAttributeVisitor::IssueNonConformanceWarning(llvm::omp::Directive D,
    parser::CharBlock source, unsigned EmitFromVersion) {
  std::string warnStr;
  llvm::raw_string_ostream warnStrOS(warnStr);
  unsigned version{context_.langOptions().OpenMPVersion};
  // We only want to emit the warning when the version being used has the
  // directive deprecated
  if (version < EmitFromVersion) {
    return;
  }
  warnStrOS << "OpenMP directive "
            << parser::ToUpperCaseLetters(
                   llvm::omp::getOpenMPDirectiveName(D, version).str())
            << " has been deprecated";

  auto setAlternativeStr = [&warnStrOS](llvm::StringRef alt) {
    warnStrOS << ", please use " << alt << " instead.";
  };
  switch (D) {
  case llvm::omp::OMPD_master:
    setAlternativeStr("MASKED");
    break;
  case llvm::omp::OMPD_master_taskloop:
    setAlternativeStr("MASKED TASKLOOP");
    break;
  case llvm::omp::OMPD_master_taskloop_simd:
    setAlternativeStr("MASKED TASKLOOP SIMD");
    break;
  case llvm::omp::OMPD_parallel_master:
    setAlternativeStr("PARALLEL MASKED");
    break;
  case llvm::omp::OMPD_parallel_master_taskloop:
    setAlternativeStr("PARALLEL MASKED TASKLOOP");
    break;
  case llvm::omp::OMPD_parallel_master_taskloop_simd:
    setAlternativeStr("PARALLEL_MASKED TASKLOOP SIMD");
    break;
  case llvm::omp::OMPD_allocate:
    setAlternativeStr("ALLOCATORS");
    break;
  case llvm::omp::OMPD_target_loop:
  default:;
  }
  context_.Warn(common::UsageWarning::OpenMPUsage, source, "%s"_warn_en_US,
      warnStrOS.str());
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
