//===-- include/flang/Semantics/tools.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_TOOLS_H_
#define FORTRAN_SEMANTICS_TOOLS_H_

// Simple predicates and look-up functions that are best defined
// canonically for use in semantic checking.

#include "flang/Common/visit.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/shape.h"
#include "flang/Evaluate/type.h"
#include "flang/Evaluate/variable.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"
#include "flang/Support/Fortran.h"
#include <functional>

namespace Fortran::semantics {

class DeclTypeSpec;
class DerivedTypeSpec;
class Scope;
class Symbol;

// Note: Here ProgramUnit includes internal subprograms while TopLevelUnit
// does not. "program-unit" in the Fortran standard matches TopLevelUnit.
const Scope &GetTopLevelUnitContaining(const Scope &);
const Scope &GetTopLevelUnitContaining(const Symbol &);
const Scope &GetProgramUnitContaining(const Scope &);
const Scope &GetProgramUnitContaining(const Symbol &);
const Scope &GetProgramUnitOrBlockConstructContaining(const Scope &);
const Scope &GetProgramUnitOrBlockConstructContaining(const Symbol &);

const Scope *FindModuleContaining(const Scope &);
const Scope *FindModuleOrSubmoduleContaining(const Scope &);
const Scope *FindModuleFileContaining(const Scope &);
const Scope *FindPureProcedureContaining(const Scope &);
const Scope *FindOpenACCConstructContaining(const Scope *);

const Symbol *FindInterface(const Symbol &);
const Symbol *FindSubprogram(const Symbol &);
const Symbol *FindOverriddenBinding(
    const Symbol &, bool &isInaccessibleDeferred);
const Symbol *FindGlobal(const Symbol &);

const DeclTypeSpec *FindParentTypeSpec(const DerivedTypeSpec &);
const DeclTypeSpec *FindParentTypeSpec(const DeclTypeSpec &);
const DeclTypeSpec *FindParentTypeSpec(const Scope &);
const DeclTypeSpec *FindParentTypeSpec(const Symbol &);

const EquivalenceSet *FindEquivalenceSet(const Symbol &);

enum class Tristate { No, Yes, Maybe };
inline Tristate ToTristate(bool x) { return x ? Tristate::Yes : Tristate::No; }

// Is this a user-defined assignment? If both sides are the same derived type
// (and the ranks are okay) the answer is Maybe.
Tristate IsDefinedAssignment(
    const std::optional<evaluate::DynamicType> &lhsType, int lhsRank,
    const std::optional<evaluate::DynamicType> &rhsType, int rhsRank);
// Test for intrinsic unary and binary operators based on types and ranks
bool IsIntrinsicRelational(common::RelationalOperator,
    const evaluate::DynamicType &, int, const evaluate::DynamicType &, int);
bool IsIntrinsicNumeric(const evaluate::DynamicType &);
bool IsIntrinsicNumeric(
    const evaluate::DynamicType &, int, const evaluate::DynamicType &, int);
bool IsIntrinsicLogical(const evaluate::DynamicType &);
bool IsIntrinsicLogical(
    const evaluate::DynamicType &, int, const evaluate::DynamicType &, int);
bool IsIntrinsicConcat(
    const evaluate::DynamicType &, int, const evaluate::DynamicType &, int);

bool IsGenericDefinedOp(const Symbol &);
bool IsDefinedOperator(SourceName);
std::string MakeOpName(SourceName);
bool IsCommonBlockContaining(const Symbol &, const Symbol &);

// Returns true if maybeAncestor exists and is a proper ancestor of a
// descendent scope (or symbol owner).  Will be false, unlike Scope::Contains(),
// if maybeAncestor *is* the descendent.
bool DoesScopeContain(const Scope *maybeAncestor, const Scope &maybeDescendent);
bool DoesScopeContain(const Scope *, const Symbol &);

bool IsUseAssociated(const Symbol &, const Scope &);
bool IsHostAssociated(const Symbol &, const Scope &);
bool IsHostAssociatedIntoSubprogram(const Symbol &, const Scope &);
inline bool IsStmtFunction(const Symbol &symbol) {
  const auto *subprogram{symbol.detailsIf<SubprogramDetails>()};
  return subprogram && subprogram->stmtFunction();
}
bool IsInStmtFunction(const Symbol &);
bool IsStmtFunctionDummy(const Symbol &);
bool IsStmtFunctionResult(const Symbol &);
bool IsPointerDummy(const Symbol &);
bool IsBindCProcedure(const Symbol &);
bool IsBindCProcedure(const Scope &);
// Returns a pointer to the function's symbol when true, else null
const Symbol *IsFunctionResultWithSameNameAsFunction(const Symbol &);
bool IsOrContainsEventOrLockComponent(const Symbol &);
bool CanBeTypeBoundProc(const Symbol &);
// Does a non-PARAMETER symbol have explicit initialization with =value or
// =>target in its declaration (but not in a DATA statement)? (Being
// ALLOCATABLE or having a derived type with default component initialization
// doesn't count; it must be a variable initialization that implies the SAVE
// attribute, or a derived type component default value.)
bool HasDeclarationInitializer(const Symbol &);
// Is the symbol explicitly or implicitly initialized in any way?
bool IsInitialized(const Symbol &, bool ignoreDATAstatements = false,
    bool ignoreAllocatable = false, bool ignorePointer = true);
// Is the symbol a component subject to deallocation or finalization?
bool IsDestructible(const Symbol &, const Symbol *derivedType = nullptr);
bool HasIntrinsicTypeName(const Symbol &);
bool IsSeparateModuleProcedureInterface(const Symbol *);
bool HasAlternateReturns(const Symbol &);
bool IsAutomaticallyDestroyed(const Symbol &);

// Return an ultimate component of type that matches predicate, or nullptr.
const Symbol *FindUltimateComponent(const DerivedTypeSpec &type,
    const std::function<bool(const Symbol &)> &predicate);
const Symbol *FindUltimateComponent(
    const Symbol &symbol, const std::function<bool(const Symbol &)> &predicate);

// Returns an immediate component of type that matches predicate, or nullptr.
// An immediate component of a type is one declared for that type or is an
// immediate component of the type that it extends.
const Symbol *FindImmediateComponent(
    const DerivedTypeSpec &, const std::function<bool(const Symbol &)> &);

inline bool IsPointer(const Symbol &symbol) {
  return symbol.attrs().test(Attr::POINTER);
}
inline bool IsAllocatable(const Symbol &symbol) {
  return symbol.attrs().test(Attr::ALLOCATABLE);
}
inline bool IsValue(const Symbol &symbol) {
  return symbol.attrs().test(Attr::VALUE);
}
// IsAllocatableOrObjectPointer() may be the better choice
inline bool IsAllocatableOrPointer(const Symbol &symbol) {
  return IsPointer(symbol) || IsAllocatable(symbol);
}
inline bool IsNamedConstant(const Symbol &symbol) {
  return symbol.attrs().test(Attr::PARAMETER);
}
inline bool IsOptional(const Symbol &symbol) {
  return symbol.attrs().test(Attr::OPTIONAL);
}
inline bool IsIntentIn(const Symbol &symbol) {
  return symbol.attrs().test(Attr::INTENT_IN);
}
inline bool IsIntentInOut(const Symbol &symbol) {
  return symbol.attrs().test(Attr::INTENT_INOUT);
}
inline bool IsIntentOut(const Symbol &symbol) {
  return symbol.attrs().test(Attr::INTENT_OUT);
}
inline bool IsProtected(const Symbol &symbol) {
  return symbol.attrs().test(Attr::PROTECTED);
}
inline bool IsImpliedDoIndex(const Symbol &symbol) {
  return symbol.owner().kind() == Scope::Kind::ImpliedDos;
}
SymbolVector FinalsForDerivedTypeInstantiation(const DerivedTypeSpec &);
// Returns a non-null pointer to a FINAL procedure, if any.
const Symbol *IsFinalizable(const Symbol &,
    std::set<const DerivedTypeSpec *> * = nullptr,
    bool withImpureFinalizer = false);
const Symbol *IsFinalizable(const DerivedTypeSpec &,
    std::set<const DerivedTypeSpec *> * = nullptr,
    bool withImpureFinalizer = false, std::optional<int> rank = std::nullopt);
const Symbol *HasImpureFinal(
    const Symbol &, std::optional<int> rank = std::nullopt);
// Is this type finalizable or does it contain any polymorphic allocatable
// ultimate components?
bool MayRequireFinalization(const DerivedTypeSpec &);
// Does this type have an allocatable direct component?
bool HasAllocatableDirectComponent(const DerivedTypeSpec &);
// Does this type have any defined assignment at any level (or any polymorphic
// allocatable)?
bool MayHaveDefinedAssignment(const DerivedTypeSpec &);

bool IsInBlankCommon(const Symbol &);
bool IsAssumedLengthCharacter(const Symbol &);
bool IsExternal(const Symbol &);
bool IsModuleProcedure(const Symbol &);
bool HasCoarray(const parser::Expr &);
bool IsAssumedType(const Symbol &);
bool IsPolymorphic(const Symbol &);
bool IsUnlimitedPolymorphic(const Symbol &);
bool IsPolymorphicAllocatable(const Symbol &);

bool IsDeviceAllocatable(const Symbol &symbol);

inline bool IsCUDADeviceContext(const Scope *scope) {
  if (scope) {
    if (const Symbol * symbol{scope->symbol()}) {
      if (const auto *subp{symbol->detailsIf<SubprogramDetails>()}) {
        if (auto attrs{subp->cudaSubprogramAttrs()}) {
          return *attrs != common::CUDASubprogramAttrs::Host;
        }
      }
    }
  }
  return false;
}

inline bool HasCUDAAttr(const Symbol &sym) {
  if (const auto *details{sym.GetUltimate().detailsIf<ObjectEntityDetails>()}) {
    if (details->cudaDataAttr()) {
      return true;
    }
  }
  return false;
}

bool HasCUDAComponent(const Symbol &sym);

inline bool IsCUDADevice(const Symbol &sym) {
  if (const auto *details{sym.GetUltimate().detailsIf<ObjectEntityDetails>()}) {
    return details->cudaDataAttr() &&
        *details->cudaDataAttr() == common::CUDADataAttr::Device;
  }
  return false;
}

inline bool IsCUDAShared(const Symbol &sym) {
  if (const auto *details{sym.GetUltimate().detailsIf<ObjectEntityDetails>()}) {
    return details->cudaDataAttr() &&
        *details->cudaDataAttr() == common::CUDADataAttr::Shared;
  }
  return false;
}

inline bool NeedCUDAAlloc(const Symbol &sym) {
  if (IsDummy(sym)) {
    return false;
  }
  if (const auto *details{sym.GetUltimate().detailsIf<ObjectEntityDetails>()}) {
    if (details->cudaDataAttr() &&
        (*details->cudaDataAttr() == common::CUDADataAttr::Device ||
            *details->cudaDataAttr() == common::CUDADataAttr::Managed ||
            *details->cudaDataAttr() == common::CUDADataAttr::Unified ||
            *details->cudaDataAttr() == common::CUDADataAttr::Shared ||
            *details->cudaDataAttr() == common::CUDADataAttr::Pinned)) {
      return true;
    }
  }
  return false;
}

bool CanCUDASymbolBeGlobal(const Symbol &sym);

const Scope *FindCUDADeviceContext(const Scope *);
std::optional<common::CUDADataAttr> GetCUDADataAttr(const Symbol *);

bool IsAccessible(const Symbol &, const Scope &);

// Return an error if a symbol is not accessible from a scope
std::optional<parser::MessageFormattedText> CheckAccessibleSymbol(
    const Scope &, const Symbol &, bool inStructureConstructor = false);

// Analysis of image control statements
bool IsImageControlStmt(const parser::ExecutableConstruct &);
// Get the location of the image control statement in this ExecutableConstruct
parser::CharBlock GetImageControlStmtLocation(
    const parser::ExecutableConstruct &);
// Image control statements that reference coarrays need an extra message
// to clarify why they're image control statements.  This function returns
// std::nullopt for ExecutableConstructs that do not require an extra message.
std::optional<parser::MessageFixedText> GetImageControlStmtCoarrayMsg(
    const parser::ExecutableConstruct &);

// Returns the complete list of derived type parameter symbols in
// the order in which their declarations appear in the derived type
// definitions (parents first).
SymbolVector OrderParameterDeclarations(const Symbol &);
// Returns the complete list of derived type parameter names in the
// order defined by 7.5.3.2.
SymbolVector OrderParameterNames(const Symbol &);

// Return an existing or new derived type instance
const DeclTypeSpec &FindOrInstantiateDerivedType(Scope &, DerivedTypeSpec &&,
    DeclTypeSpec::Category = DeclTypeSpec::TypeDerived);

// When a subprogram defined in a submodule defines a separate module
// procedure whose interface is defined in an ancestor (sub)module,
// returns a pointer to that interface, else null.
const Symbol *FindSeparateModuleSubprogramInterface(const Symbol *);

// Determines whether an object might be visible outside a
// pure function (C1594); returns a non-null Symbol pointer for
// diagnostic purposes if so.
const Symbol *FindExternallyVisibleObject(
    const Symbol &, const Scope &, bool isPointerDefinition);

template <typename A>
const Symbol *FindExternallyVisibleObject(const A &, const Scope &) {
  return nullptr; // default base case
}

template <typename T>
const Symbol *FindExternallyVisibleObject(
    const evaluate::Designator<T> &designator, const Scope &scope) {
  if (const Symbol * symbol{designator.GetBaseObject().symbol()}) {
    return FindExternallyVisibleObject(*symbol, scope, false);
  } else if (std::holds_alternative<evaluate::CoarrayRef>(designator.u)) {
    // Coindexed values are visible even if their image-local objects are not.
    return designator.GetBaseObject().symbol();
  } else {
    return nullptr;
  }
}

template <typename T>
const Symbol *FindExternallyVisibleObject(
    const evaluate::Expr<T> &expr, const Scope &scope) {
  return common::visit(
      [&](const auto &x) { return FindExternallyVisibleObject(x, scope); },
      expr.u);
}

// Applies GetUltimate(), then if the symbol is a generic procedure shadowing a
// specific procedure of the same name, return it instead.
const Symbol &BypassGeneric(const Symbol &);

// Given a cray pointee symbol, returns the related cray pointer symbol.
const Symbol &GetCrayPointer(const Symbol &crayPointee);

using SomeExpr = evaluate::Expr<evaluate::SomeType>;

bool ExprHasTypeCategory(
    const SomeExpr &expr, const common::TypeCategory &type);
bool ExprTypeKindIsDefault(
    const SomeExpr &expr, const SemanticsContext &context);

class GetExprHelper {
public:
  explicit GetExprHelper(SemanticsContext *context) : context_{context} {}
  GetExprHelper() : crashIfNoExpr_{true} {}

  // Specializations for parse tree nodes that have a typedExpr member.
  const SomeExpr *Get(const parser::Expr &);
  const SomeExpr *Get(const parser::Variable &);
  const SomeExpr *Get(const parser::DataStmtConstant &);
  const SomeExpr *Get(const parser::AllocateObject &);
  const SomeExpr *Get(const parser::PointerObject &);

  template <typename T> const SomeExpr *Get(const common::Indirection<T> &x) {
    return Get(x.value());
  }
  template <typename T> const SomeExpr *Get(const std::optional<T> &x) {
    return x ? Get(*x) : nullptr;
  }
  template <typename T> const SomeExpr *Get(const T &x) {
    static_assert(
        !parser::HasTypedExpr<T>::value, "explicit Get overload must be added");
    if constexpr (ConstraintTrait<T>) {
      return Get(x.thing);
    } else if constexpr (WrapperTrait<T>) {
      return Get(x.v);
    } else {
      return nullptr;
    }
  }

private:
  SemanticsContext *context_{nullptr};
  const bool crashIfNoExpr_{false};
};

// If a SemanticsContext is passed, even if null, it is possible for a null
// pointer to be returned in the event of an expression that had fatal errors.
// Use these first two forms in semantics checks for best error recovery.
// If a SemanticsContext is not passed, a missing expression will
// cause a crash.
template <typename T>
const SomeExpr *GetExpr(SemanticsContext *context, const T &x) {
  return GetExprHelper{context}.Get(x);
}
template <typename T>
const SomeExpr *GetExpr(SemanticsContext &context, const T &x) {
  return GetExprHelper{&context}.Get(x);
}
template <typename T> const SomeExpr *GetExpr(const T &x) {
  return GetExprHelper{}.Get(x);
}

const evaluate::Assignment *GetAssignment(const parser::AssignmentStmt &);
const evaluate::Assignment *GetAssignment(
    const parser::PointerAssignmentStmt &);

template <typename T> std::optional<std::int64_t> GetIntValue(const T &x) {
  if (const auto *expr{GetExpr(nullptr, x)}) {
    return evaluate::ToInt64(*expr);
  } else {
    return std::nullopt;
  }
}

template <typename T> bool IsZero(const T &expr) {
  auto value{GetIntValue(expr)};
  return value && *value == 0;
}

// 15.2.2
enum class ProcedureDefinitionClass {
  None,
  Intrinsic,
  External,
  Internal,
  Module,
  Dummy,
  Pointer,
  StatementFunction
};

ProcedureDefinitionClass ClassifyProcedure(const Symbol &);

// Returns a list of storage associations due to EQUIVALENCE in a
// scope; each storage association is a list of symbol references
// in ascending order of scope offset.  Note that the scope may have
// more EquivalenceSets than this function's result has storage
// associations; these are closures over equivalences.
std::list<std::list<SymbolRef>> GetStorageAssociations(const Scope &);

// Derived type component iterator that provides a C++ LegacyForwardIterator
// iterator over the Ordered, Direct, Ultimate or Potential components of a
// DerivedTypeSpec. These iterators can be used with STL algorithms
// accepting LegacyForwardIterator.
// The kind of component is a template argument of the iterator factory
// ComponentIterator.
//
// - Ordered components are the components from the component order defined
// in 7.5.4.7, except that the parent component IS added between the parent
// component order and the components in order of declaration.
// This "deviation" is important for structure-constructor analysis.
// For this kind of iterator, the component tree is recursively visited in the
// following order:
//  - first, the Ordered components of the parent type (if relevant)
//  - then, the parent component (if relevant, different from 7.5.4.7!)
//  - then, the components in declaration order (without visiting subcomponents)
//
// - Ultimate, Direct and Potential components are as defined in 7.5.1.
//   - Ultimate components of a derived type are the closure of its components
//     of intrinsic type, its ALLOCATABLE or POINTER components, and the
//     ultimate components of its non-ALLOCATABLE non-POINTER derived type
//     components.  (No ultimate component has a derived type unless it is
//     ALLOCATABLE or POINTER.)
//   - Direct components of a derived type are all of its components, and all
//     of the direct components of its non-ALLOCATABLE non-POINTER derived type
//     components.  (Direct components are always present.)
//   - Potential subobject components of a derived type are the closure of
//     its non-POINTER components and the potential subobject components of
//     its non-POINTER derived type components.  (The lifetime of each
//     potential subobject component is that of the entire instance.)
//   - PotentialAndPointer subobject components of a derived type are the
//     closure of its components (including POINTERs) and the
//     PotentialAndPointer subobject components of its non-POINTER derived type
//     components.
//
// type t1                     ultimate components:  x, a, p
//  real x                     direct components:    x, a, p
//  real, allocatable :: a     potential components: x, a
//  real, pointer :: p         potential & pointers: x, a, p
// end type
// type t2                     ultimate components:  y, c%x, c%a, c%p, b
//  real y                     direct components:    y, c, c%x, c%a, c%p, b
//  type(t1) :: c              potential components: y, c, c%x, c%a, b, b%x, b%a
//  type(t1), allocatable :: b potential & pointers: potentials + c%p + b%p
// end type
//
// Parent and procedure components are considered against these definitions.
// For this kind of iterator, the component tree is recursively visited in the
// following order:
//  - the parent component first (if relevant)
//  - then, the components of the parent type (if relevant)
//      + visiting the component and then, if it is derived type data component,
//        visiting the subcomponents before visiting the next
//        component in declaration order.
//  - then, components in declaration order, similarly to components of parent
//    type.
//  Here, the parent component is visited first so that search for a component
//  verifying a property will never descend into a component that already
//  verifies the property (this helps giving clearer feedback).
//
// ComponentIterator::const_iterator remain valid during the whole lifetime of
// the DerivedTypeSpec passed by reference to the ComponentIterator factory.
// Their validity is independent of the ComponentIterator factory lifetime.
//
// For safety and simplicity, the iterators are read only and can only be
// incremented. This could be changed if desired.
//
// Note that iterators are made in such a way that one can easily test and build
// info message in the following way:
//    ComponentIterator<ComponentKind::...> comp{derived}
//    if (auto it{std::find_if(comp.begin(), comp.end(), predicate)}) {
//       msg = it.BuildResultDesignatorName() + " verifies predicates";
//       const Symbol *component{*it};
//       ....
//    }

ENUM_CLASS(ComponentKind, Ordered, Direct, Ultimate, Potential, Scope,
    PotentialAndPointer)

template <ComponentKind componentKind> class ComponentIterator {
public:
  ComponentIterator(const DerivedTypeSpec &derived) : derived_{derived} {}
  class const_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = SymbolRef;
    using difference_type = void;
    using pointer = const Symbol *;
    using reference = const Symbol &;

    static const_iterator Create(const DerivedTypeSpec &);

    const_iterator &operator++() {
      Increment();
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp(*this);
      Increment();
      return tmp;
    }
    reference operator*() const {
      CHECK(!componentPath_.empty());
      return DEREF(componentPath_.back().component());
    }
    pointer operator->() const { return &**this; }

    bool operator==(const const_iterator &other) const {
      return componentPath_ == other.componentPath_;
    }
    bool operator!=(const const_iterator &other) const {
      return !(*this == other);
    }

    // bool() operator indicates if the iterator can be dereferenced without
    // having to check against an end() iterator.
    explicit operator bool() const { return !componentPath_.empty(); }

    // Returns the current sequence of components, including parent components.
    SymbolVector GetComponentPath() const;

    // Builds a designator name of the referenced component for messages.
    // The designator helps when the component referred to by the iterator
    // may be "buried" into other components. This gives the full
    // path inside the iterated derived type: e.g "%a%b%c%ultimate"
    // when it->name() only gives "ultimate". Parent components are
    // part of the path for clarity, even though they could be
    // skipped.
    std::string BuildResultDesignatorName() const;

  private:
    using name_iterator =
        std::conditional_t<componentKind == ComponentKind::Scope,
            typename Scope::const_iterator,
            typename std::list<SourceName>::const_iterator>;

    class ComponentPathNode {
    public:
      explicit ComponentPathNode(const DerivedTypeSpec &derived)
          : derived_{derived} {
        if constexpr (componentKind == ComponentKind::Scope) {
          const Scope &scope{DEREF(derived.GetScope())};
          nameIterator_ = scope.cbegin();
          nameEnd_ = scope.cend();
        } else {
          const std::list<SourceName> &nameList{
              derived.typeSymbol().get<DerivedTypeDetails>().componentNames()};
          nameIterator_ = nameList.cbegin();
          nameEnd_ = nameList.cend();
        }
      }
      const Symbol *component() const { return component_; }
      void set_component(const Symbol &component) { component_ = &component; }
      bool visited() const { return visited_; }
      void set_visited(bool yes) { visited_ = yes; }
      bool descended() const { return descended_; }
      void set_descended(bool yes) { descended_ = yes; }
      name_iterator &nameIterator() { return nameIterator_; }
      name_iterator nameEnd() { return nameEnd_; }
      const Symbol &GetTypeSymbol() const { return derived_->typeSymbol(); }
      const Scope &GetScope() const {
        return derived_->scope() ? *derived_->scope()
                                 : DEREF(GetTypeSymbol().scope());
      }
      bool operator==(const ComponentPathNode &that) const {
        return &*derived_ == &*that.derived_ &&
            nameIterator_ == that.nameIterator_ &&
            component_ == that.component_;
      }

    private:
      common::Reference<const DerivedTypeSpec> derived_;
      name_iterator nameEnd_;
      name_iterator nameIterator_;
      const Symbol *component_{nullptr}; // until Increment()
      bool visited_{false};
      bool descended_{false};
    };

    const DerivedTypeSpec *PlanComponentTraversal(
        const Symbol &component) const;
    // Advances to the next relevant symbol, if any.  Afterwards, the
    // iterator will either be at its end or contain no null component().
    void Increment();

    std::vector<ComponentPathNode> componentPath_;
  };

  const_iterator begin() { return cbegin(); }
  const_iterator end() { return cend(); }
  const_iterator cbegin() { return const_iterator::Create(derived_); }
  const_iterator cend() { return const_iterator{}; }

private:
  const DerivedTypeSpec &derived_;
};

extern template class ComponentIterator<ComponentKind::Ordered>;
extern template class ComponentIterator<ComponentKind::Direct>;
extern template class ComponentIterator<ComponentKind::Ultimate>;
extern template class ComponentIterator<ComponentKind::Potential>;
extern template class ComponentIterator<ComponentKind::Scope>;
extern template class ComponentIterator<ComponentKind::PotentialAndPointer>;
using OrderedComponentIterator = ComponentIterator<ComponentKind::Ordered>;
using DirectComponentIterator = ComponentIterator<ComponentKind::Direct>;
using UltimateComponentIterator = ComponentIterator<ComponentKind::Ultimate>;
using PotentialComponentIterator = ComponentIterator<ComponentKind::Potential>;
using ScopeComponentIterator = ComponentIterator<ComponentKind::Scope>;
using PotentialAndPointerComponentIterator =
    ComponentIterator<ComponentKind::PotentialAndPointer>;

// Common component searches, the iterator returned is referring to the first
// component, according to the order defined for the related ComponentIterator,
// that verifies the property from the name.
// If no component verifies the property, an end iterator (casting to false)
// is returned. Otherwise, the returned iterator casts to true and can be
// dereferenced.
PotentialComponentIterator::const_iterator FindEventOrLockPotentialComponent(
    const DerivedTypeSpec &, bool ignoreCoarrays = false);
PotentialComponentIterator::const_iterator FindCoarrayPotentialComponent(
    const DerivedTypeSpec &);
PotentialAndPointerComponentIterator::const_iterator
FindPointerPotentialComponent(const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator FindCoarrayUltimateComponent(
    const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator FindPointerUltimateComponent(
    const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator FindAllocatableUltimateComponent(
    const DerivedTypeSpec &);
DirectComponentIterator::const_iterator FindAllocatableOrPointerDirectComponent(
    const DerivedTypeSpec &);
PotentialComponentIterator::const_iterator
FindPolymorphicAllocatablePotentialComponent(const DerivedTypeSpec &);
UltimateComponentIterator::const_iterator
FindCUDADeviceAllocatableUltimateComponent(const DerivedTypeSpec &);

// The LabelEnforce class (given a set of labels) provides an error message if
// there is a branch to a label which is not in the given set.
class LabelEnforce {
public:
  LabelEnforce(SemanticsContext &context, std::set<parser::Label> &&labels,
      parser::CharBlock constructSourcePosition, const char *construct)
      : context_{context}, labels_{labels},
        constructSourcePosition_{constructSourcePosition}, construct_{
                                                               construct} {}
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    return true;
  }

  template <typename T> void Post(const T &) {}

  void Post(const parser::GotoStmt &gotoStmt);
  void Post(const parser::ComputedGotoStmt &computedGotoStmt);
  void Post(const parser::ArithmeticIfStmt &arithmeticIfStmt);
  void Post(const parser::AssignStmt &assignStmt);
  void Post(const parser::AssignedGotoStmt &assignedGotoStmt);
  void Post(const parser::AltReturnSpec &altReturnSpec);
  void Post(const parser::ErrLabel &errLabel);
  void Post(const parser::EndLabel &endLabel);
  void Post(const parser::EorLabel &eorLabel);
  void CheckLabelUse(const parser::Label &labelUsed);

private:
  SemanticsContext &context_;
  std::set<parser::Label> labels_;
  parser::CharBlock currentStatementSourcePosition_{nullptr};
  parser::CharBlock constructSourcePosition_{nullptr};
  const char *construct_{nullptr};

  parser::MessageFormattedText GetEnclosingConstructMsg();
  void SayWithConstruct(SemanticsContext &context,
      parser::CharBlock stmtLocation, parser::MessageFormattedText &&message,
      parser::CharBlock constructLocation);
};
// Return the (possibly null) name of the ConstructNode
const std::optional<parser::Name> &MaybeGetNodeName(
    const ConstructNode &construct);

// Convert evaluate::GetShape() result into an ArraySpec
std::optional<ArraySpec> ToArraySpec(
    evaluate::FoldingContext &, const evaluate::Shape &);
std::optional<ArraySpec> ToArraySpec(
    evaluate::FoldingContext &, const std::optional<evaluate::Shape> &);

// Searches a derived type and a scope for a particular defined I/O procedure.
bool HasDefinedIo(
    common::DefinedIo, const DerivedTypeSpec &, const Scope * = nullptr);

// Some intrinsic operators have more than one name (e.g. `operator(.eq.)` and
// `operator(==)`). GetAllNames() returns them all, including symbolName.
std::forward_list<std::string> GetAllNames(
    const SemanticsContext &, const SourceName &);

// Determines the derived type of a procedure's initial "dtv" dummy argument,
// assuming that the procedure is a specific procedure of a defined I/O
// generic interface,
const DerivedTypeSpec *GetDtvArgDerivedType(const Symbol &);

// If "expr" exists and is a designator for a deferred length
// character allocatable whose semantics might change under Fortran 202X,
// emit a portability warning.
void WarnOnDeferredLengthCharacterScalar(SemanticsContext &, const SomeExpr *,
    parser::CharBlock at, const char *what);

inline const parser::Name *getDesignatorNameIfDataRef(
    const parser::Designator &designator) {
  const auto *dataRef{std::get_if<parser::DataRef>(&designator.u)};
  return dataRef ? std::get_if<parser::Name>(&dataRef->u) : nullptr;
}

bool CouldBeDataPointerValuedFunction(const Symbol *);

template <typename R, typename T>
std::optional<R> GetConstExpr(SemanticsContext &semanticsContext, const T &x) {
  using DefaultCharConstantType = evaluate::Ascii;
  if (const auto *expr{GetExpr(semanticsContext, x)}) {
    const auto foldExpr{evaluate::Fold(
        semanticsContext.foldingContext(), common::Clone(*expr))};
    if constexpr (std::is_same_v<R, std::string>) {
      return evaluate::GetScalarConstantValue<DefaultCharConstantType>(
          foldExpr);
    }
  }
  return std::nullopt;
}

// Returns "m" for a module, "m:sm" for a submodule.
std::string GetModuleOrSubmoduleName(const Symbol &);

// Return the assembly name emitted for a common block.
std::string GetCommonBlockObjectName(const Symbol &, bool underscoring);

// Check for ambiguous USE associations
bool HadUseError(SemanticsContext &, SourceName at, const Symbol *);

bool AreSameModuleSymbol(const Symbol &, const Symbol &);

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_TOOLS_H_
