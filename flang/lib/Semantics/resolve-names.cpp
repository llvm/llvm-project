//===-- lib/Semantics/resolve-names.cpp -----------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "resolve-names.h"
#include "assignment.h"
#include "mod-file.h"
#include "pointer-assignment.h"
#include "program-tree.h"
#include "resolve-directives.h"
#include "resolve-names-utils.h"
#include "rewrite-parse-tree.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/default-kinds.h"
#include "flang/Common/indirection.h"
#include "flang/Common/restorer.h"
#include "flang/Common/visit.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/fold-designator.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "llvm/Support/raw_ostream.h"
#include <list>
#include <map>
#include <set>
#include <stack>

namespace Fortran::semantics {

using namespace parser::literals;

template <typename T> using Indirection = common::Indirection<T>;
using Message = parser::Message;
using Messages = parser::Messages;
using MessageFixedText = parser::MessageFixedText;
using MessageFormattedText = parser::MessageFormattedText;

class ResolveNamesVisitor;
class ScopeHandler;

// ImplicitRules maps initial character of identifier to the DeclTypeSpec
// representing the implicit type; std::nullopt if none.
// It also records the presence of IMPLICIT NONE statements.
// When inheritFromParent is set, defaults come from the parent rules.
class ImplicitRules {
public:
  ImplicitRules(SemanticsContext &context, ImplicitRules *parent)
      : parent_{parent}, context_{context} {
    inheritFromParent_ = parent != nullptr;
  }
  bool isImplicitNoneType() const;
  bool isImplicitNoneExternal() const;
  void set_isImplicitNoneType(bool x) { isImplicitNoneType_ = x; }
  void set_isImplicitNoneExternal(bool x) { isImplicitNoneExternal_ = x; }
  void set_inheritFromParent(bool x) { inheritFromParent_ = x; }
  // Get the implicit type for this name. May be null.
  const DeclTypeSpec *GetType(
      SourceName, bool respectImplicitNone = true) const;
  // Record the implicit type for the range of characters [fromLetter,
  // toLetter].
  void SetTypeMapping(const DeclTypeSpec &type, parser::Location fromLetter,
      parser::Location toLetter);

private:
  static char Incr(char ch);

  ImplicitRules *parent_;
  SemanticsContext &context_;
  bool inheritFromParent_{false}; // look in parent if not specified here
  bool isImplicitNoneType_{
      context_.IsEnabled(common::LanguageFeature::ImplicitNoneTypeAlways)};
  bool isImplicitNoneExternal_{false};
  // map_ contains the mapping between letters and types that were defined
  // by the IMPLICIT statements of the related scope. It does not contain
  // the default Fortran mappings nor the mapping defined in parents.
  std::map<char, common::Reference<const DeclTypeSpec>> map_;

  friend llvm::raw_ostream &operator<<(
      llvm::raw_ostream &, const ImplicitRules &);
  friend void ShowImplicitRule(
      llvm::raw_ostream &, const ImplicitRules &, char);
};

// scope -> implicit rules for that scope
using ImplicitRulesMap = std::map<const Scope *, ImplicitRules>;

// Track statement source locations and save messages.
class MessageHandler {
public:
  MessageHandler() { DIE("MessageHandler: default-constructed"); }
  explicit MessageHandler(SemanticsContext &c) : context_{&c} {}
  Messages &messages() { return context_->messages(); };
  const std::optional<SourceName> &currStmtSource() {
    return context_->location();
  }
  void set_currStmtSource(const std::optional<SourceName> &source) {
    context_->set_location(source);
  }

  // Emit a message associated with the current statement source.
  Message &Say(MessageFixedText &&);
  Message &Say(MessageFormattedText &&);
  // Emit a message about a SourceName
  Message &Say(const SourceName &, MessageFixedText &&);
  // Emit a formatted message associated with a source location.
  template <typename... A>
  Message &Say(const SourceName &source, MessageFixedText &&msg, A &&...args) {
    return context_->Say(source, std::move(msg), std::forward<A>(args)...);
  }

private:
  SemanticsContext *context_;
};

// Inheritance graph for the parse tree visitation classes that follow:
//   BaseVisitor
//   + AttrsVisitor
//   | + DeclTypeSpecVisitor
//   |   + ImplicitRulesVisitor
//   |     + ScopeHandler -----------+--+
//   |       + ModuleVisitor ========|==+
//   |       + InterfaceVisitor      |  |
//   |       +-+ SubprogramVisitor ==|==+
//   + ArraySpecVisitor              |  |
//     + DeclarationVisitor <--------+  |
//       + ConstructVisitor             |
//         + ResolveNamesVisitor <------+

class BaseVisitor {
public:
  BaseVisitor() { DIE("BaseVisitor: default-constructed"); }
  BaseVisitor(
      SemanticsContext &c, ResolveNamesVisitor &v, ImplicitRulesMap &rules)
      : implicitRulesMap_{&rules}, this_{&v}, context_{&c}, messageHandler_{c} {
  }
  template <typename T> void Walk(const T &);

  MessageHandler &messageHandler() { return messageHandler_; }
  const std::optional<SourceName> &currStmtSource() {
    return context_->location();
  }
  SemanticsContext &context() const { return *context_; }
  evaluate::FoldingContext &GetFoldingContext() const {
    return context_->foldingContext();
  }
  bool IsIntrinsic(
      const SourceName &name, std::optional<Symbol::Flag> flag) const {
    if (!flag) {
      return context_->intrinsics().IsIntrinsic(name.ToString());
    } else if (flag == Symbol::Flag::Function) {
      return context_->intrinsics().IsIntrinsicFunction(name.ToString());
    } else if (flag == Symbol::Flag::Subroutine) {
      return context_->intrinsics().IsIntrinsicSubroutine(name.ToString());
    } else {
      DIE("expected Subroutine or Function flag");
    }
  }

  bool InModuleFile() const { return GetFoldingContext().inModuleFile(); }

  // Make a placeholder symbol for a Name that otherwise wouldn't have one.
  // It is not in any scope and always has MiscDetails.
  void MakePlaceholder(const parser::Name &, MiscDetails::Kind);

  template <typename T> common::IfNoLvalue<T, T> FoldExpr(T &&expr) {
    return evaluate::Fold(GetFoldingContext(), std::move(expr));
  }

  template <typename T> MaybeExpr EvaluateExpr(const T &expr) {
    return FoldExpr(AnalyzeExpr(*context_, expr));
  }

  template <typename T>
  MaybeExpr EvaluateNonPointerInitializer(
      const Symbol &symbol, const T &expr, parser::CharBlock source) {
    if (!context().HasError(symbol)) {
      if (auto maybeExpr{AnalyzeExpr(*context_, expr)}) {
        auto restorer{GetFoldingContext().messages().SetLocation(source)};
        return evaluate::NonPointerInitializationExpr(
            symbol, std::move(*maybeExpr), GetFoldingContext());
      }
    }
    return std::nullopt;
  }

  template <typename T> MaybeIntExpr EvaluateIntExpr(const T &expr) {
    return semantics::EvaluateIntExpr(*context_, expr);
  }

  template <typename T>
  MaybeSubscriptIntExpr EvaluateSubscriptIntExpr(const T &expr) {
    if (MaybeIntExpr maybeIntExpr{EvaluateIntExpr(expr)}) {
      return FoldExpr(evaluate::ConvertToType<evaluate::SubscriptInteger>(
          std::move(*maybeIntExpr)));
    } else {
      return std::nullopt;
    }
  }

  template <typename... A> Message &Say(A &&...args) {
    return messageHandler_.Say(std::forward<A>(args)...);
  }
  template <typename... A>
  Message &Say(
      const parser::Name &name, MessageFixedText &&text, const A &...args) {
    return messageHandler_.Say(name.source, std::move(text), args...);
  }

protected:
  ImplicitRulesMap *implicitRulesMap_{nullptr};

private:
  ResolveNamesVisitor *this_;
  SemanticsContext *context_;
  MessageHandler messageHandler_;
};

// Provide Post methods to collect attributes into a member variable.
class AttrsVisitor : public virtual BaseVisitor {
public:
  bool BeginAttrs(); // always returns true
  Attrs GetAttrs();
  Attrs EndAttrs();
  bool SetPassNameOn(Symbol &);
  void SetBindNameOn(Symbol &);
  void Post(const parser::LanguageBindingSpec &);
  bool Pre(const parser::IntentSpec &);
  bool Pre(const parser::Pass &);

  bool CheckAndSet(Attr);

// Simple case: encountering CLASSNAME causes ATTRNAME to be set.
#define HANDLE_ATTR_CLASS(CLASSNAME, ATTRNAME) \
  bool Pre(const parser::CLASSNAME &) { \
    CheckAndSet(Attr::ATTRNAME); \
    return false; \
  }
  HANDLE_ATTR_CLASS(PrefixSpec::Elemental, ELEMENTAL)
  HANDLE_ATTR_CLASS(PrefixSpec::Impure, IMPURE)
  HANDLE_ATTR_CLASS(PrefixSpec::Module, MODULE)
  HANDLE_ATTR_CLASS(PrefixSpec::Non_Recursive, NON_RECURSIVE)
  HANDLE_ATTR_CLASS(PrefixSpec::Pure, PURE)
  HANDLE_ATTR_CLASS(PrefixSpec::Recursive, RECURSIVE)
  HANDLE_ATTR_CLASS(TypeAttrSpec::BindC, BIND_C)
  HANDLE_ATTR_CLASS(BindAttr::Deferred, DEFERRED)
  HANDLE_ATTR_CLASS(BindAttr::Non_Overridable, NON_OVERRIDABLE)
  HANDLE_ATTR_CLASS(Abstract, ABSTRACT)
  HANDLE_ATTR_CLASS(Allocatable, ALLOCATABLE)
  HANDLE_ATTR_CLASS(Asynchronous, ASYNCHRONOUS)
  HANDLE_ATTR_CLASS(Contiguous, CONTIGUOUS)
  HANDLE_ATTR_CLASS(External, EXTERNAL)
  HANDLE_ATTR_CLASS(Intrinsic, INTRINSIC)
  HANDLE_ATTR_CLASS(NoPass, NOPASS)
  HANDLE_ATTR_CLASS(Optional, OPTIONAL)
  HANDLE_ATTR_CLASS(Parameter, PARAMETER)
  HANDLE_ATTR_CLASS(Pointer, POINTER)
  HANDLE_ATTR_CLASS(Protected, PROTECTED)
  HANDLE_ATTR_CLASS(Save, SAVE)
  HANDLE_ATTR_CLASS(Target, TARGET)
  HANDLE_ATTR_CLASS(Value, VALUE)
  HANDLE_ATTR_CLASS(Volatile, VOLATILE)
#undef HANDLE_ATTR_CLASS

protected:
  std::optional<Attrs> attrs_;

  Attr AccessSpecToAttr(const parser::AccessSpec &x) {
    switch (x.v) {
    case parser::AccessSpec::Kind::Public:
      return Attr::PUBLIC;
    case parser::AccessSpec::Kind::Private:
      return Attr::PRIVATE;
    }
    llvm_unreachable("Switch covers all cases"); // suppress g++ warning
  }
  Attr IntentSpecToAttr(const parser::IntentSpec &x) {
    switch (x.v) {
    case parser::IntentSpec::Intent::In:
      return Attr::INTENT_IN;
    case parser::IntentSpec::Intent::Out:
      return Attr::INTENT_OUT;
    case parser::IntentSpec::Intent::InOut:
      return Attr::INTENT_INOUT;
    }
    llvm_unreachable("Switch covers all cases"); // suppress g++ warning
  }

private:
  bool IsDuplicateAttr(Attr);
  bool HaveAttrConflict(Attr, Attr, Attr);
  bool IsConflictingAttr(Attr);

  MaybeExpr bindName_; // from BIND(C, NAME="...")
  std::optional<SourceName> passName_; // from PASS(...)
};

// Find and create types from declaration-type-spec nodes.
class DeclTypeSpecVisitor : public AttrsVisitor {
public:
  using AttrsVisitor::Post;
  using AttrsVisitor::Pre;
  void Post(const parser::IntrinsicTypeSpec::DoublePrecision &);
  void Post(const parser::IntrinsicTypeSpec::DoubleComplex &);
  void Post(const parser::DeclarationTypeSpec::ClassStar &);
  void Post(const parser::DeclarationTypeSpec::TypeStar &);
  bool Pre(const parser::TypeGuardStmt &);
  void Post(const parser::TypeGuardStmt &);
  void Post(const parser::TypeSpec &);

  // Walk the parse tree of a type spec and return the DeclTypeSpec for it.
  template <typename T>
  const DeclTypeSpec *ProcessTypeSpec(const T &x, bool allowForward = false) {
    auto restorer{common::ScopedSet(state_, State{})};
    set_allowForwardReferenceToDerivedType(allowForward);
    BeginDeclTypeSpec();
    Walk(x);
    const auto *type{GetDeclTypeSpec()};
    EndDeclTypeSpec();
    return type;
  }

protected:
  struct State {
    bool expectDeclTypeSpec{false}; // should see decl-type-spec only when true
    const DeclTypeSpec *declTypeSpec{nullptr};
    struct {
      DerivedTypeSpec *type{nullptr};
      DeclTypeSpec::Category category{DeclTypeSpec::TypeDerived};
    } derived;
    bool allowForwardReferenceToDerivedType{false};
  };

  bool allowForwardReferenceToDerivedType() const {
    return state_.allowForwardReferenceToDerivedType;
  }
  void set_allowForwardReferenceToDerivedType(bool yes) {
    state_.allowForwardReferenceToDerivedType = yes;
  }

  const DeclTypeSpec *GetDeclTypeSpec();
  void BeginDeclTypeSpec();
  void EndDeclTypeSpec();
  void SetDeclTypeSpec(const DeclTypeSpec &);
  void SetDeclTypeSpecCategory(DeclTypeSpec::Category);
  DeclTypeSpec::Category GetDeclTypeSpecCategory() const {
    return state_.derived.category;
  }
  KindExpr GetKindParamExpr(
      TypeCategory, const std::optional<parser::KindSelector> &);
  void CheckForAbstractType(const Symbol &typeSymbol);

private:
  State state_;

  void MakeNumericType(TypeCategory, int kind);
};

// Visit ImplicitStmt and related parse tree nodes and updates implicit rules.
class ImplicitRulesVisitor : public DeclTypeSpecVisitor {
public:
  using DeclTypeSpecVisitor::Post;
  using DeclTypeSpecVisitor::Pre;
  using ImplicitNoneNameSpec = parser::ImplicitStmt::ImplicitNoneNameSpec;

  void Post(const parser::ParameterStmt &);
  bool Pre(const parser::ImplicitStmt &);
  bool Pre(const parser::LetterSpec &);
  bool Pre(const parser::ImplicitSpec &);
  void Post(const parser::ImplicitSpec &);

  const DeclTypeSpec *GetType(
      SourceName name, bool respectImplicitNoneType = true) {
    return implicitRules_->GetType(name, respectImplicitNoneType);
  }
  bool isImplicitNoneType() const {
    return implicitRules_->isImplicitNoneType();
  }
  bool isImplicitNoneType(const Scope &scope) const {
    return implicitRulesMap_->at(&scope).isImplicitNoneType();
  }
  bool isImplicitNoneExternal() const {
    return implicitRules_->isImplicitNoneExternal();
  }
  void set_inheritFromParent(bool x) {
    implicitRules_->set_inheritFromParent(x);
  }

protected:
  void BeginScope(const Scope &);
  void SetScope(const Scope &);

private:
  // implicit rules in effect for current scope
  ImplicitRules *implicitRules_{nullptr};
  std::optional<SourceName> prevImplicit_;
  std::optional<SourceName> prevImplicitNone_;
  std::optional<SourceName> prevImplicitNoneType_;
  std::optional<SourceName> prevParameterStmt_;

  bool HandleImplicitNone(const std::list<ImplicitNoneNameSpec> &nameSpecs);
};

// Track array specifications. They can occur in AttrSpec, EntityDecl,
// ObjectDecl, DimensionStmt, CommonBlockObject, or BasedPointerStmt.
// 1. INTEGER, DIMENSION(10) :: x
// 2. INTEGER :: x(10)
// 3. ALLOCATABLE :: x(:)
// 4. DIMENSION :: x(10)
// 5. COMMON x(10)
// 6. BasedPointerStmt
class ArraySpecVisitor : public virtual BaseVisitor {
public:
  void Post(const parser::ArraySpec &);
  void Post(const parser::ComponentArraySpec &);
  void Post(const parser::CoarraySpec &);
  void Post(const parser::AttrSpec &) { PostAttrSpec(); }
  void Post(const parser::ComponentAttrSpec &) { PostAttrSpec(); }

protected:
  const ArraySpec &arraySpec();
  void set_arraySpec(const ArraySpec arraySpec) { arraySpec_ = arraySpec; }
  const ArraySpec &coarraySpec();
  void BeginArraySpec();
  void EndArraySpec();
  void ClearArraySpec() { arraySpec_.clear(); }
  void ClearCoarraySpec() { coarraySpec_.clear(); }

private:
  // arraySpec_/coarraySpec_ are populated from any ArraySpec/CoarraySpec
  ArraySpec arraySpec_;
  ArraySpec coarraySpec_;
  // When an ArraySpec is under an AttrSpec or ComponentAttrSpec, it is moved
  // into attrArraySpec_
  ArraySpec attrArraySpec_;
  ArraySpec attrCoarraySpec_;

  void PostAttrSpec();
};

// Manages a stack of function result information.  We defer the processing
// of a type specification that appears in the prefix of a FUNCTION statement
// until the function result variable appears in the specification part
// or the end of the specification part.  This allows for forward references
// in the type specification to resolve to local names.
class FuncResultStack {
public:
  explicit FuncResultStack(ScopeHandler &scopeHandler)
      : scopeHandler_{scopeHandler} {}
  ~FuncResultStack();

  struct FuncInfo {
    explicit FuncInfo(const Scope &s) : scope{s} {}
    const Scope &scope;
    // Parse tree of the type specification in the FUNCTION prefix
    const parser::DeclarationTypeSpec *parsedType{nullptr};
    // Name of the function RESULT in the FUNCTION suffix, if any
    const parser::Name *resultName{nullptr};
    // Result symbol
    Symbol *resultSymbol{nullptr};
    std::optional<SourceName> source;
    bool inFunctionStmt{false}; // true between Pre/Post of FunctionStmt
  };

  // Completes the definition of the top function's result.
  void CompleteFunctionResultType();
  // Completes the definition of a symbol if it is the top function's result.
  void CompleteTypeIfFunctionResult(Symbol &);

  FuncInfo *Top() { return stack_.empty() ? nullptr : &stack_.back(); }
  FuncInfo &Push(const Scope &scope) { return stack_.emplace_back(scope); }
  void Pop();

private:
  ScopeHandler &scopeHandler_;
  std::vector<FuncInfo> stack_;
};

// Manage a stack of Scopes
class ScopeHandler : public ImplicitRulesVisitor {
public:
  using ImplicitRulesVisitor::Post;
  using ImplicitRulesVisitor::Pre;

  Scope &currScope() { return DEREF(currScope_); }
  // The enclosing host procedure if current scope is in an internal procedure
  Scope *GetHostProcedure();
  // The innermost enclosing program unit scope, ignoring BLOCK and other
  // construct scopes.
  Scope &InclusiveScope();
  // The enclosing scope, skipping derived types.
  Scope &NonDerivedTypeScope();

  // Create a new scope and push it on the scope stack.
  void PushScope(Scope::Kind kind, Symbol *symbol);
  void PushScope(Scope &scope);
  void PopScope();
  void SetScope(Scope &);

  template <typename T> bool Pre(const parser::Statement<T> &x) {
    messageHandler().set_currStmtSource(x.source);
    currScope_->AddSourceRange(x.source);
    return true;
  }
  template <typename T> void Post(const parser::Statement<T> &) {
    messageHandler().set_currStmtSource(std::nullopt);
  }

  // Special messages: already declared; referencing symbol's declaration;
  // about a type; two names & locations
  void SayAlreadyDeclared(const parser::Name &, Symbol &);
  void SayAlreadyDeclared(const SourceName &, Symbol &);
  void SayAlreadyDeclared(const SourceName &, const SourceName &);
  void SayWithReason(
      const parser::Name &, Symbol &, MessageFixedText &&, Message &&);
  void SayWithDecl(const parser::Name &, Symbol &, MessageFixedText &&);
  void SayLocalMustBeVariable(const parser::Name &, Symbol &);
  void SayDerivedType(const SourceName &, MessageFixedText &&, const Scope &);
  void Say2(const SourceName &, MessageFixedText &&, const SourceName &,
      MessageFixedText &&);
  void Say2(
      const SourceName &, MessageFixedText &&, Symbol &, MessageFixedText &&);
  void Say2(
      const parser::Name &, MessageFixedText &&, Symbol &, MessageFixedText &&);

  // Search for symbol by name in current, parent derived type, and
  // containing scopes
  Symbol *FindSymbol(const parser::Name &);
  Symbol *FindSymbol(const Scope &, const parser::Name &);
  // Search for name only in scope, not in enclosing scopes.
  Symbol *FindInScope(const Scope &, const parser::Name &);
  Symbol *FindInScope(const Scope &, const SourceName &);
  template <typename T> Symbol *FindInScope(const T &name) {
    return FindInScope(currScope(), name);
  }
  // Search for name in a derived type scope and its parents.
  Symbol *FindInTypeOrParents(const Scope &, const parser::Name &);
  Symbol *FindInTypeOrParents(const parser::Name &);
  Symbol *FindSeparateModuleProcedureInterface(const parser::Name &);
  void EraseSymbol(const parser::Name &);
  void EraseSymbol(const Symbol &symbol) { currScope().erase(symbol.name()); }
  // Make a new symbol with the name and attrs of an existing one
  Symbol &CopySymbol(const SourceName &, const Symbol &);

  // Make symbols in the current or named scope
  Symbol &MakeSymbol(Scope &, const SourceName &, Attrs);
  Symbol &MakeSymbol(const SourceName &, Attrs = Attrs{});
  Symbol &MakeSymbol(const parser::Name &, Attrs = Attrs{});
  Symbol &MakeHostAssocSymbol(const parser::Name &, const Symbol &);

  template <typename D>
  common::IfNoLvalue<Symbol &, D> MakeSymbol(
      const parser::Name &name, D &&details) {
    return MakeSymbol(name, Attrs{}, std::move(details));
  }

  template <typename D>
  common::IfNoLvalue<Symbol &, D> MakeSymbol(
      const parser::Name &name, const Attrs &attrs, D &&details) {
    return Resolve(name, MakeSymbol(name.source, attrs, std::move(details)));
  }

  template <typename D>
  common::IfNoLvalue<Symbol &, D> MakeSymbol(
      const SourceName &name, const Attrs &attrs, D &&details) {
    // Note: don't use FindSymbol here. If this is a derived type scope,
    // we want to detect whether the name is already declared as a component.
    auto *symbol{FindInScope(name)};
    if (!symbol) {
      symbol = &MakeSymbol(name, attrs);
      symbol->set_details(std::move(details));
      return *symbol;
    }
    if constexpr (std::is_same_v<DerivedTypeDetails, D>) {
      if (auto *d{symbol->detailsIf<GenericDetails>()}) {
        if (!d->specific()) {
          // derived type with same name as a generic
          auto *derivedType{d->derivedType()};
          if (!derivedType) {
            derivedType =
                &currScope().MakeSymbol(name, attrs, std::move(details));
            d->set_derivedType(*derivedType);
          } else if (derivedType->CanReplaceDetails(details)) {
            // was forward-referenced
            derivedType->attrs() |= attrs;
            derivedType->set_details(std::move(details));
          } else {
            SayAlreadyDeclared(name, *derivedType);
          }
          return *derivedType;
        }
      }
    }
    if (symbol->CanReplaceDetails(details)) {
      // update the existing symbol
      symbol->attrs() |= attrs;
      if constexpr (std::is_same_v<SubprogramDetails, D>) {
        // Dummy argument defined by explicit interface?
        details.set_isDummy(IsDummy(*symbol));
      }
      symbol->set_details(std::move(details));
      return *symbol;
    } else if constexpr (std::is_same_v<UnknownDetails, D>) {
      symbol->attrs() |= attrs;
      return *symbol;
    } else {
      if (!CheckPossibleBadForwardRef(*symbol)) {
        if (name.empty() && symbol->name().empty()) {
          // report the error elsewhere
          return *symbol;
        }
        SayAlreadyDeclared(name, *symbol);
      }
      // replace the old symbol with a new one with correct details
      EraseSymbol(*symbol);
      auto &result{MakeSymbol(name, attrs, std::move(details))};
      context().SetError(result);
      return result;
    }
  }

  void MakeExternal(Symbol &);

protected:
  FuncResultStack &funcResultStack() { return funcResultStack_; }

  // Apply the implicit type rules to this symbol.
  void ApplyImplicitRules(Symbol &, bool allowForwardReference = false);
  bool ImplicitlyTypeForwardRef(Symbol &);
  void AcquireIntrinsicProcedureFlags(Symbol &);
  const DeclTypeSpec *GetImplicitType(
      Symbol &, bool respectImplicitNoneType = true);
  bool ConvertToObjectEntity(Symbol &);
  bool ConvertToProcEntity(Symbol &);

  const DeclTypeSpec &MakeNumericType(
      TypeCategory, const std::optional<parser::KindSelector> &);
  const DeclTypeSpec &MakeLogicalType(
      const std::optional<parser::KindSelector> &);
  void NotePossibleBadForwardRef(const parser::Name &);
  std::optional<SourceName> HadForwardRef(const Symbol &) const;
  bool CheckPossibleBadForwardRef(const Symbol &);

  bool inSpecificationPart_{false};
  bool inEquivalenceStmt_{false};

  // Some information is collected from a specification part for deferred
  // processing in DeclarationPartVisitor functions (e.g., CheckSaveStmts())
  // that are called by ResolveNamesVisitor::FinishSpecificationPart().  Since
  // specification parts can nest (e.g., INTERFACE bodies), the collected
  // information that is not contained in the scope needs to be packaged
  // and restorable.
  struct SpecificationPartState {
    std::set<SourceName> forwardRefs;
    // Collect equivalence sets and process at end of specification part
    std::vector<const std::list<parser::EquivalenceObject> *> equivalenceSets;
    // Names of all common block objects in the scope
    std::set<SourceName> commonBlockObjects;
    // Info about about SAVE statements and attributes in current scope
    struct {
      std::optional<SourceName> saveAll; // "SAVE" without entity list
      std::set<SourceName> entities; // names of entities with save attr
      std::set<SourceName> commons; // names of common blocks with save attr
    } saveInfo;
  } specPartState_;

  // Some declaration processing can and should be deferred to
  // ResolveExecutionParts() to avoid prematurely creating implicitly-typed
  // local symbols that should be host associations.
  struct DeferredDeclarationState {
    // The content of each namelist group
    std::list<const parser::NamelistStmt::Group *> namelistGroups;
  };
  DeferredDeclarationState *GetDeferredDeclarationState(bool add = false) {
    if (!add && deferred_.find(&currScope()) == deferred_.end()) {
      return nullptr;
    } else {
      return &deferred_.emplace(&currScope(), DeferredDeclarationState{})
                  .first->second;
    }
  }

private:
  Scope *currScope_{nullptr};
  FuncResultStack funcResultStack_{*this};
  std::map<Scope *, DeferredDeclarationState> deferred_;
};

class ModuleVisitor : public virtual ScopeHandler {
public:
  bool Pre(const parser::AccessStmt &);
  bool Pre(const parser::Only &);
  bool Pre(const parser::Rename::Names &);
  bool Pre(const parser::Rename::Operators &);
  bool Pre(const parser::UseStmt &);
  void Post(const parser::UseStmt &);

  void BeginModule(const parser::Name &, bool isSubmodule);
  bool BeginSubmodule(const parser::Name &, const parser::ParentIdentifier &);
  void ApplyDefaultAccess();
  Symbol &AddGenericUse(GenericDetails &, const SourceName &, const Symbol &);
  void AddAndCheckExplicitIntrinsicUse(SourceName, bool isIntrinsic);
  void ClearUseRenames() { useRenames_.clear(); }
  void ClearUseOnly() { useOnly_.clear(); }
  void ClearExplicitIntrinsicUses() {
    explicitIntrinsicUses_.clear();
    explicitNonIntrinsicUses_.clear();
  }

private:
  // The default access spec for this module.
  Attr defaultAccess_{Attr::PUBLIC};
  // The location of the last AccessStmt without access-ids, if any.
  std::optional<SourceName> prevAccessStmt_;
  // The scope of the module during a UseStmt
  Scope *useModuleScope_{nullptr};
  // Names that have appeared in a rename clause of a USE statement
  std::set<std::pair<SourceName, Scope *>> useRenames_;
  // Names that have appeared in an ONLY clause of a USE statement
  std::set<std::pair<SourceName, Scope *>> useOnly_;
  // Module names that have appeared in USE statements with explicit
  // INTRINSIC or NON_INTRINSIC keywords
  std::set<SourceName> explicitIntrinsicUses_;
  std::set<SourceName> explicitNonIntrinsicUses_;

  Symbol &SetAccess(const SourceName &, Attr attr, Symbol * = nullptr);
  // A rename in a USE statement: local => use
  struct SymbolRename {
    Symbol *local{nullptr};
    Symbol *use{nullptr};
  };
  // Record a use from useModuleScope_ of use Name/Symbol as local Name/Symbol
  SymbolRename AddUse(const SourceName &localName, const SourceName &useName);
  SymbolRename AddUse(const SourceName &, const SourceName &, Symbol *);
  void DoAddUse(
      SourceName, SourceName, Symbol &localSymbol, const Symbol &useSymbol);
  void AddUse(const GenericSpecInfo &);
  // If appropriate, erase a previously USE-associated symbol
  void EraseRenamedSymbol(const Symbol &);
  // Record a name appearing in a USE rename clause
  void AddUseRename(const SourceName &name) {
    useRenames_.emplace(std::make_pair(name, useModuleScope_));
  }
  bool IsUseRenamed(const SourceName &name) const {
    return useRenames_.find({name, useModuleScope_}) != useRenames_.end();
  }
  // Record a name appearing in a USE ONLY clause
  void AddUseOnly(const SourceName &name) {
    useOnly_.emplace(std::make_pair(name, useModuleScope_));
  }
  bool IsUseOnly(const SourceName &name) const {
    return useOnly_.find({name, useModuleScope_}) != useOnly_.end();
  }
  Scope *FindModule(const parser::Name &, std::optional<bool> isIntrinsic,
      Scope *ancestor = nullptr);
};

class InterfaceVisitor : public virtual ScopeHandler {
public:
  bool Pre(const parser::InterfaceStmt &);
  void Post(const parser::InterfaceStmt &);
  void Post(const parser::EndInterfaceStmt &);
  bool Pre(const parser::GenericSpec &);
  bool Pre(const parser::ProcedureStmt &);
  bool Pre(const parser::GenericStmt &);
  void Post(const parser::GenericStmt &);

  bool inInterfaceBlock() const;
  bool isGeneric() const;
  bool isAbstract() const;

protected:
  Symbol &GetGenericSymbol() { return DEREF(genericInfo_.top().symbol); }
  // Add to generic the symbol for the subprogram with the same name
  void CheckGenericProcedures(Symbol &);

private:
  // A new GenericInfo is pushed for each interface block and generic stmt
  struct GenericInfo {
    GenericInfo(bool isInterface, bool isAbstract = false)
        : isInterface{isInterface}, isAbstract{isAbstract} {}
    bool isInterface; // in interface block
    bool isAbstract; // in abstract interface block
    Symbol *symbol{nullptr}; // the generic symbol being defined
  };
  std::stack<GenericInfo> genericInfo_;
  const GenericInfo &GetGenericInfo() const { return genericInfo_.top(); }
  void SetGenericSymbol(Symbol &symbol) { genericInfo_.top().symbol = &symbol; }

  using ProcedureKind = parser::ProcedureStmt::Kind;
  // mapping of generic to its specific proc names and kinds
  std::multimap<Symbol *, std::pair<const parser::Name *, ProcedureKind>>
      specificProcs_;

  void AddSpecificProcs(const std::list<parser::Name> &, ProcedureKind);
  void ResolveSpecificsInGeneric(Symbol &generic);
};

class SubprogramVisitor : public virtual ScopeHandler, public InterfaceVisitor {
public:
  bool HandleStmtFunction(const parser::StmtFunctionStmt &);
  bool Pre(const parser::SubroutineStmt &);
  bool Pre(const parser::FunctionStmt &);
  void Post(const parser::FunctionStmt &);
  bool Pre(const parser::EntryStmt &);
  void Post(const parser::EntryStmt &);
  bool Pre(const parser::InterfaceBody::Subroutine &);
  void Post(const parser::InterfaceBody::Subroutine &);
  bool Pre(const parser::InterfaceBody::Function &);
  void Post(const parser::InterfaceBody::Function &);
  bool Pre(const parser::Suffix &);
  bool Pre(const parser::PrefixSpec &);

  bool BeginSubprogram(const parser::Name &, Symbol::Flag,
      bool hasModulePrefix = false,
      const parser::LanguageBindingSpec * = nullptr,
      const ProgramTree::EntryStmtList * = nullptr);
  bool BeginMpSubprogram(const parser::Name &);
  void PushBlockDataScope(const parser::Name &);
  void EndSubprogram(std::optional<parser::CharBlock> stmtSource = std::nullopt,
      const std::optional<parser::LanguageBindingSpec> * = nullptr,
      const ProgramTree::EntryStmtList * = nullptr);

protected:
  // Set when we see a stmt function that is really an array element assignment
  bool badStmtFuncFound_{false};

private:
  // Edits an existing symbol created for earlier calls to a subprogram or ENTRY
  // so that it can be replaced by a later definition.
  bool HandlePreviousCalls(const parser::Name &, Symbol &, Symbol::Flag);
  void CheckExtantProc(const parser::Name &, Symbol::Flag);
  // Create a subprogram symbol in the current scope and push a new scope.
  Symbol &PushSubprogramScope(const parser::Name &, Symbol::Flag,
      const parser::LanguageBindingSpec * = nullptr);
  Symbol *GetSpecificFromGeneric(const parser::Name &);
  SubprogramDetails &PostSubprogramStmt(const parser::Name &);
  void CreateEntry(const parser::EntryStmt &stmt, Symbol &subprogram);
  void PostEntryStmt(const parser::EntryStmt &stmt);
  void HandleLanguageBinding(Symbol *,
      std::optional<parser::CharBlock> stmtSource,
      const std::optional<parser::LanguageBindingSpec> *);
};

class DeclarationVisitor : public ArraySpecVisitor,
                           public virtual ScopeHandler {
public:
  using ArraySpecVisitor::Post;
  using ScopeHandler::Post;
  using ScopeHandler::Pre;

  bool Pre(const parser::Initialization &);
  void Post(const parser::EntityDecl &);
  void Post(const parser::ObjectDecl &);
  void Post(const parser::PointerDecl &);
  bool Pre(const parser::BindStmt &) { return BeginAttrs(); }
  void Post(const parser::BindStmt &) { EndAttrs(); }
  bool Pre(const parser::BindEntity &);
  bool Pre(const parser::OldParameterStmt &);
  bool Pre(const parser::NamedConstantDef &);
  bool Pre(const parser::NamedConstant &);
  void Post(const parser::EnumDef &);
  bool Pre(const parser::Enumerator &);
  bool Pre(const parser::AccessSpec &);
  bool Pre(const parser::AsynchronousStmt &);
  bool Pre(const parser::ContiguousStmt &);
  bool Pre(const parser::ExternalStmt &);
  bool Pre(const parser::IntentStmt &);
  bool Pre(const parser::IntrinsicStmt &);
  bool Pre(const parser::OptionalStmt &);
  bool Pre(const parser::ProtectedStmt &);
  bool Pre(const parser::ValueStmt &);
  bool Pre(const parser::VolatileStmt &);
  bool Pre(const parser::AllocatableStmt &) {
    objectDeclAttr_ = Attr::ALLOCATABLE;
    return true;
  }
  void Post(const parser::AllocatableStmt &) { objectDeclAttr_ = std::nullopt; }
  bool Pre(const parser::TargetStmt &) {
    objectDeclAttr_ = Attr::TARGET;
    return true;
  }
  void Post(const parser::TargetStmt &) { objectDeclAttr_ = std::nullopt; }
  void Post(const parser::DimensionStmt::Declaration &);
  void Post(const parser::CodimensionDecl &);
  bool Pre(const parser::TypeDeclarationStmt &) { return BeginDecl(); }
  void Post(const parser::TypeDeclarationStmt &);
  void Post(const parser::IntegerTypeSpec &);
  void Post(const parser::IntrinsicTypeSpec::Real &);
  void Post(const parser::IntrinsicTypeSpec::Complex &);
  void Post(const parser::IntrinsicTypeSpec::Logical &);
  void Post(const parser::IntrinsicTypeSpec::Character &);
  void Post(const parser::CharSelector::LengthAndKind &);
  void Post(const parser::CharLength &);
  void Post(const parser::LengthSelector &);
  bool Pre(const parser::KindParam &);
  bool Pre(const parser::DeclarationTypeSpec::Type &);
  void Post(const parser::DeclarationTypeSpec::Type &);
  bool Pre(const parser::DeclarationTypeSpec::Class &);
  void Post(const parser::DeclarationTypeSpec::Class &);
  void Post(const parser::DeclarationTypeSpec::Record &);
  void Post(const parser::DerivedTypeSpec &);
  bool Pre(const parser::DerivedTypeDef &);
  bool Pre(const parser::DerivedTypeStmt &);
  void Post(const parser::DerivedTypeStmt &);
  bool Pre(const parser::TypeParamDefStmt &) { return BeginDecl(); }
  void Post(const parser::TypeParamDefStmt &);
  bool Pre(const parser::TypeAttrSpec::Extends &);
  bool Pre(const parser::PrivateStmt &);
  bool Pre(const parser::SequenceStmt &);
  bool Pre(const parser::ComponentDefStmt &) { return BeginDecl(); }
  void Post(const parser::ComponentDefStmt &) { EndDecl(); }
  void Post(const parser::ComponentDecl &);
  void Post(const parser::FillDecl &);
  bool Pre(const parser::ProcedureDeclarationStmt &);
  void Post(const parser::ProcedureDeclarationStmt &);
  bool Pre(const parser::DataComponentDefStmt &); // returns false
  bool Pre(const parser::ProcComponentDefStmt &);
  void Post(const parser::ProcComponentDefStmt &);
  bool Pre(const parser::ProcPointerInit &);
  void Post(const parser::ProcInterface &);
  void Post(const parser::ProcDecl &);
  bool Pre(const parser::TypeBoundProcedurePart &);
  void Post(const parser::TypeBoundProcedurePart &);
  void Post(const parser::ContainsStmt &);
  bool Pre(const parser::TypeBoundProcBinding &) { return BeginAttrs(); }
  void Post(const parser::TypeBoundProcBinding &) { EndAttrs(); }
  void Post(const parser::TypeBoundProcedureStmt::WithoutInterface &);
  void Post(const parser::TypeBoundProcedureStmt::WithInterface &);
  void Post(const parser::FinalProcedureStmt &);
  bool Pre(const parser::TypeBoundGenericStmt &);
  bool Pre(const parser::StructureDef &); // returns false
  bool Pre(const parser::Union::UnionStmt &);
  bool Pre(const parser::StructureField &);
  void Post(const parser::StructureField &);
  bool Pre(const parser::AllocateStmt &);
  void Post(const parser::AllocateStmt &);
  bool Pre(const parser::StructureConstructor &);
  bool Pre(const parser::NamelistStmt::Group &);
  bool Pre(const parser::IoControlSpec &);
  bool Pre(const parser::CommonStmt::Block &);
  bool Pre(const parser::CommonBlockObject &);
  void Post(const parser::CommonBlockObject &);
  bool Pre(const parser::EquivalenceStmt &);
  bool Pre(const parser::SaveStmt &);
  bool Pre(const parser::BasedPointerStmt &);

  void PointerInitialization(
      const parser::Name &, const parser::InitialDataTarget &);
  void PointerInitialization(
      const parser::Name &, const parser::ProcPointerInit &);
  void NonPointerInitialization(
      const parser::Name &, const parser::ConstantExpr &);
  void CheckExplicitInterface(const parser::Name &);
  void CheckBindings(const parser::TypeBoundProcedureStmt::WithoutInterface &);

  const parser::Name *ResolveDesignator(const parser::Designator &);

protected:
  bool BeginDecl();
  void EndDecl();
  Symbol &DeclareObjectEntity(const parser::Name &, Attrs = Attrs{});
  // Make sure that there's an entity in an enclosing scope called Name
  Symbol &FindOrDeclareEnclosingEntity(const parser::Name &);
  // Declare a LOCAL/LOCAL_INIT entity. If there isn't a type specified
  // it comes from the entity in the containing scope, or implicit rules.
  // Return pointer to the new symbol, or nullptr on error.
  Symbol *DeclareLocalEntity(const parser::Name &);
  // Declare a statement entity (i.e., an implied DO loop index for
  // a DATA statement or an array constructor).  If there isn't an explict
  // type specified, implicit rules apply. Return pointer to the new symbol,
  // or nullptr on error.
  Symbol *DeclareStatementEntity(const parser::DoVariable &,
      const std::optional<parser::IntegerTypeSpec> &);
  Symbol &MakeCommonBlockSymbol(const parser::Name &);
  Symbol &MakeCommonBlockSymbol(const std::optional<parser::Name> &);
  bool CheckUseError(const parser::Name &);
  void CheckAccessibility(const SourceName &, bool, Symbol &);
  void CheckCommonBlocks();
  void CheckSaveStmts();
  void CheckEquivalenceSets();
  bool CheckNotInBlock(const char *);
  bool NameIsKnownOrIntrinsic(const parser::Name &);
  void FinishNamelists();

  // Each of these returns a pointer to a resolved Name (i.e. with symbol)
  // or nullptr in case of error.
  const parser::Name *ResolveStructureComponent(
      const parser::StructureComponent &);
  const parser::Name *ResolveDataRef(const parser::DataRef &);
  const parser::Name *ResolveName(const parser::Name &);
  bool PassesSharedLocalityChecks(const parser::Name &name, Symbol &symbol);
  Symbol *NoteInterfaceName(const parser::Name &);
  bool IsUplevelReference(const Symbol &);

  std::optional<SourceName> BeginCheckOnIndexUseInOwnBounds(
      const parser::DoVariable &name) {
    std::optional<SourceName> result{checkIndexUseInOwnBounds_};
    checkIndexUseInOwnBounds_ = name.thing.thing.source;
    return result;
  }
  void EndCheckOnIndexUseInOwnBounds(const std::optional<SourceName> &restore) {
    checkIndexUseInOwnBounds_ = restore;
  }

private:
  // The attribute corresponding to the statement containing an ObjectDecl
  std::optional<Attr> objectDeclAttr_;
  // Info about current character type while walking DeclTypeSpec.
  // Also captures any "*length" specifier on an individual declaration.
  struct {
    std::optional<ParamValue> length;
    std::optional<KindExpr> kind;
  } charInfo_;
  // Info about current derived type or STRUCTURE while walking
  // DerivedTypeDef / StructureDef
  struct {
    const parser::Name *extends{nullptr}; // EXTENDS(name)
    bool privateComps{false}; // components are private by default
    bool privateBindings{false}; // bindings are private by default
    bool sawContains{false}; // currently processing bindings
    bool sequence{false}; // is a sequence type
    const Symbol *type{nullptr}; // derived type being defined
    bool isStructure{false}; // is a DEC STRUCTURE
  } derivedTypeInfo_;
  // In a ProcedureDeclarationStmt or ProcComponentDefStmt, this is
  // the interface name, if any.
  const parser::Name *interfaceName_{nullptr};
  // Map type-bound generic to binding names of its specific bindings
  std::multimap<Symbol *, const parser::Name *> genericBindings_;
  // Info about current ENUM
  struct EnumeratorState {
    // Enum value must hold inside a C_INT (7.6.2).
    std::optional<int> value{0};
  } enumerationState_;
  // Set for OldParameterStmt processing
  bool inOldStyleParameterStmt_{false};
  // Set when walking DATA & array constructor implied DO loop bounds
  // to warn about use of the implied DO intex therein.
  std::optional<SourceName> checkIndexUseInOwnBounds_;
  bool hasBindCName_{false};

  bool HandleAttributeStmt(Attr, const std::list<parser::Name> &);
  Symbol &HandleAttributeStmt(Attr, const parser::Name &);
  Symbol &DeclareUnknownEntity(const parser::Name &, Attrs);
  Symbol &DeclareProcEntity(const parser::Name &, Attrs, const ProcInterface &);
  void SetType(const parser::Name &, const DeclTypeSpec &);
  std::optional<DerivedTypeSpec> ResolveDerivedType(const parser::Name &);
  std::optional<DerivedTypeSpec> ResolveExtendsType(
      const parser::Name &, const parser::Name *);
  Symbol *MakeTypeSymbol(const SourceName &, Details &&);
  Symbol *MakeTypeSymbol(const parser::Name &, Details &&);
  bool OkToAddComponent(const parser::Name &, const Symbol * = nullptr);
  ParamValue GetParamValue(
      const parser::TypeParamValue &, common::TypeParamAttr attr);
  void CheckCommonBlockDerivedType(const SourceName &, const Symbol &);
  std::optional<MessageFixedText> CheckSaveAttr(const Symbol &);
  Attrs HandleSaveName(const SourceName &, Attrs);
  void AddSaveName(std::set<SourceName> &, const SourceName &);
  void SetSaveAttr(Symbol &);
  bool HandleUnrestrictedSpecificIntrinsicFunction(const parser::Name &);
  const parser::Name *FindComponent(const parser::Name *, const parser::Name &);
  void Initialization(const parser::Name &, const parser::Initialization &,
      bool inComponentDecl);
  bool PassesLocalityChecks(const parser::Name &name, Symbol &symbol);
  bool CheckForHostAssociatedImplicit(const parser::Name &);

  // Declare an object or procedure entity.
  // T is one of: EntityDetails, ObjectEntityDetails, ProcEntityDetails
  template <typename T>
  Symbol &DeclareEntity(const parser::Name &name, Attrs attrs) {
    Symbol &symbol{MakeSymbol(name, attrs)};
    if (context().HasError(symbol) || symbol.has<T>()) {
      return symbol; // OK or error already reported
    } else if (symbol.has<UnknownDetails>()) {
      symbol.set_details(T{});
      return symbol;
    } else if (auto *details{symbol.detailsIf<EntityDetails>()}) {
      symbol.set_details(T{std::move(*details)});
      return symbol;
    } else if (std::is_same_v<EntityDetails, T> &&
        (symbol.has<ObjectEntityDetails>() ||
            symbol.has<ProcEntityDetails>())) {
      return symbol; // OK
    } else if (auto *details{symbol.detailsIf<UseDetails>()}) {
      Say(name.source,
          "'%s' is use-associated from module '%s' and cannot be re-declared"_err_en_US,
          name.source, GetUsedModule(*details).name());
    } else if (auto *details{symbol.detailsIf<SubprogramNameDetails>()}) {
      if (details->kind() == SubprogramKind::Module) {
        Say2(name,
            "Declaration of '%s' conflicts with its use as module procedure"_err_en_US,
            symbol, "Module procedure definition"_en_US);
      } else if (details->kind() == SubprogramKind::Internal) {
        Say2(name,
            "Declaration of '%s' conflicts with its use as internal procedure"_err_en_US,
            symbol, "Internal procedure definition"_en_US);
      } else {
        DIE("unexpected kind");
      }
    } else if (std::is_same_v<ObjectEntityDetails, T> &&
        symbol.has<ProcEntityDetails>()) {
      SayWithDecl(
          name, symbol, "'%s' is already declared as a procedure"_err_en_US);
    } else if (std::is_same_v<ProcEntityDetails, T> &&
        symbol.has<ObjectEntityDetails>()) {
      if (InCommonBlock(symbol)) {
        SayWithDecl(name, symbol,
            "'%s' may not be a procedure as it is in a COMMON block"_err_en_US);
      } else {
        SayWithDecl(
            name, symbol, "'%s' is already declared as an object"_err_en_US);
      }
    } else if (!CheckPossibleBadForwardRef(symbol)) {
      SayAlreadyDeclared(name, symbol);
    }
    context().SetError(symbol);
    return symbol;
  }
  bool HasCycle(const Symbol &, const ProcInterface &);
};

// Resolve construct entities and statement entities.
// Check that construct names don't conflict with other names.
class ConstructVisitor : public virtual DeclarationVisitor {
public:
  bool Pre(const parser::ConcurrentHeader &);
  bool Pre(const parser::LocalitySpec::Local &);
  bool Pre(const parser::LocalitySpec::LocalInit &);
  bool Pre(const parser::LocalitySpec::Shared &);
  bool Pre(const parser::AcSpec &);
  bool Pre(const parser::AcImpliedDo &);
  bool Pre(const parser::DataImpliedDo &);
  bool Pre(const parser::DataIDoObject &);
  bool Pre(const parser::DataStmtObject &);
  bool Pre(const parser::DataStmtValue &);
  bool Pre(const parser::DoConstruct &);
  void Post(const parser::DoConstruct &);
  bool Pre(const parser::ForallConstruct &);
  void Post(const parser::ForallConstruct &);
  bool Pre(const parser::ForallStmt &);
  void Post(const parser::ForallStmt &);
  bool Pre(const parser::BlockStmt &);
  bool Pre(const parser::EndBlockStmt &);
  void Post(const parser::Selector &);
  void Post(const parser::AssociateStmt &);
  void Post(const parser::EndAssociateStmt &);
  bool Pre(const parser::Association &);
  void Post(const parser::SelectTypeStmt &);
  void Post(const parser::SelectRankStmt &);
  bool Pre(const parser::SelectTypeConstruct &);
  void Post(const parser::SelectTypeConstruct &);
  bool Pre(const parser::SelectTypeConstruct::TypeCase &);
  void Post(const parser::SelectTypeConstruct::TypeCase &);
  // Creates Block scopes with neither symbol name nor symbol details.
  bool Pre(const parser::SelectRankConstruct::RankCase &);
  void Post(const parser::SelectRankConstruct::RankCase &);
  void Post(const parser::TypeGuardStmt::Guard &);
  void Post(const parser::SelectRankCaseStmt::Rank &);
  bool Pre(const parser::ChangeTeamStmt &);
  void Post(const parser::EndChangeTeamStmt &);
  void Post(const parser::CoarrayAssociation &);

  // Definitions of construct names
  bool Pre(const parser::WhereConstructStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::ForallConstructStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::CriticalStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::LabelDoStmt &) {
    return false; // error recovery
  }
  bool Pre(const parser::NonLabelDoStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::IfThenStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::SelectCaseStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::SelectRankConstruct &);
  void Post(const parser::SelectRankConstruct &);
  bool Pre(const parser::SelectRankStmt &x) {
    return CheckDef(std::get<0>(x.t));
  }
  bool Pre(const parser::SelectTypeStmt &x) {
    return CheckDef(std::get<0>(x.t));
  }

  // References to construct names
  void Post(const parser::MaskedElsewhereStmt &x) { CheckRef(x.t); }
  void Post(const parser::ElsewhereStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndWhereStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndForallStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndCriticalStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndDoStmt &x) { CheckRef(x.v); }
  void Post(const parser::ElseIfStmt &x) { CheckRef(x.t); }
  void Post(const parser::ElseStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndIfStmt &x) { CheckRef(x.v); }
  void Post(const parser::CaseStmt &x) { CheckRef(x.t); }
  void Post(const parser::EndSelectStmt &x) { CheckRef(x.v); }
  void Post(const parser::SelectRankCaseStmt &x) { CheckRef(x.t); }
  void Post(const parser::TypeGuardStmt &x) { CheckRef(x.t); }
  void Post(const parser::CycleStmt &x) { CheckRef(x.v); }
  void Post(const parser::ExitStmt &x) { CheckRef(x.v); }

private:
  // R1105 selector -> expr | variable
  // expr is set in either case unless there were errors
  struct Selector {
    Selector() {}
    Selector(const SourceName &source, MaybeExpr &&expr)
        : source{source}, expr{std::move(expr)} {}
    operator bool() const { return expr.has_value(); }
    parser::CharBlock source;
    MaybeExpr expr;
  };
  // association -> [associate-name =>] selector
  struct Association {
    const parser::Name *name{nullptr};
    Selector selector;
  };
  std::vector<Association> associationStack_;
  Association *currentAssociation_{nullptr};

  template <typename T> bool CheckDef(const T &t) {
    return CheckDef(std::get<std::optional<parser::Name>>(t));
  }
  template <typename T> void CheckRef(const T &t) {
    CheckRef(std::get<std::optional<parser::Name>>(t));
  }
  bool CheckDef(const std::optional<parser::Name> &);
  void CheckRef(const std::optional<parser::Name> &);
  const DeclTypeSpec &ToDeclTypeSpec(evaluate::DynamicType &&);
  const DeclTypeSpec &ToDeclTypeSpec(
      evaluate::DynamicType &&, MaybeSubscriptIntExpr &&length);
  Symbol *MakeAssocEntity();
  void SetTypeFromAssociation(Symbol &);
  void SetAttrsFromAssociation(Symbol &);
  Selector ResolveSelector(const parser::Selector &);
  void ResolveIndexName(const parser::ConcurrentControl &control);
  void SetCurrentAssociation(std::size_t n);
  Association &GetCurrentAssociation();
  void PushAssociation();
  void PopAssociation(std::size_t count = 1);
};

// Create scopes for OpenACC constructs
class AccVisitor : public virtual DeclarationVisitor {
public:
  void AddAccSourceRange(const parser::CharBlock &);

  static bool NeedsScope(const parser::OpenACCBlockConstruct &);

  bool Pre(const parser::OpenACCBlockConstruct &);
  void Post(const parser::OpenACCBlockConstruct &);
  bool Pre(const parser::AccBeginBlockDirective &x) {
    AddAccSourceRange(x.source);
    return true;
  }
  void Post(const parser::AccBeginBlockDirective &) {
    messageHandler().set_currStmtSource(std::nullopt);
  }
  bool Pre(const parser::AccEndBlockDirective &x) {
    AddAccSourceRange(x.source);
    return true;
  }
  void Post(const parser::AccEndBlockDirective &) {
    messageHandler().set_currStmtSource(std::nullopt);
  }
  bool Pre(const parser::AccBeginLoopDirective &x) {
    AddAccSourceRange(x.source);
    return true;
  }
  void Post(const parser::AccBeginLoopDirective &x) {
    messageHandler().set_currStmtSource(std::nullopt);
  }
};

bool AccVisitor::NeedsScope(const parser::OpenACCBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginBlockDirective>(x.t)};
  const auto &beginDir{std::get<parser::AccBlockDirective>(beginBlockDir.t)};
  switch (beginDir.v) {
  case llvm::acc::Directive::ACCD_data:
  case llvm::acc::Directive::ACCD_host_data:
  case llvm::acc::Directive::ACCD_kernels:
  case llvm::acc::Directive::ACCD_parallel:
  case llvm::acc::Directive::ACCD_serial:
    return true;
  default:
    return false;
  }
}

void AccVisitor::AddAccSourceRange(const parser::CharBlock &source) {
  messageHandler().set_currStmtSource(source);
  currScope().AddSourceRange(source);
}

bool AccVisitor::Pre(const parser::OpenACCBlockConstruct &x) {
  if (NeedsScope(x)) {
    PushScope(Scope::Kind::OtherConstruct, nullptr);
  }
  return true;
}

void AccVisitor::Post(const parser::OpenACCBlockConstruct &x) {
  if (NeedsScope(x)) {
    PopScope();
  }
}

// Create scopes for OpenMP constructs
class OmpVisitor : public virtual DeclarationVisitor {
public:
  void AddOmpSourceRange(const parser::CharBlock &);

  static bool NeedsScope(const parser::OpenMPBlockConstruct &);

  bool Pre(const parser::OpenMPBlockConstruct &);
  void Post(const parser::OpenMPBlockConstruct &);
  bool Pre(const parser::OmpBeginBlockDirective &x) {
    AddOmpSourceRange(x.source);
    return true;
  }
  void Post(const parser::OmpBeginBlockDirective &) {
    messageHandler().set_currStmtSource(std::nullopt);
  }
  bool Pre(const parser::OmpEndBlockDirective &x) {
    AddOmpSourceRange(x.source);
    return true;
  }
  void Post(const parser::OmpEndBlockDirective &) {
    messageHandler().set_currStmtSource(std::nullopt);
  }

  bool Pre(const parser::OpenMPLoopConstruct &) {
    PushScope(Scope::Kind::OtherConstruct, nullptr);
    return true;
  }
  void Post(const parser::OpenMPLoopConstruct &) { PopScope(); }
  bool Pre(const parser::OmpBeginLoopDirective &x) {
    AddOmpSourceRange(x.source);
    return true;
  }
  void Post(const parser::OmpBeginLoopDirective &) {
    messageHandler().set_currStmtSource(std::nullopt);
  }
  bool Pre(const parser::OmpEndLoopDirective &x) {
    AddOmpSourceRange(x.source);
    return true;
  }
  void Post(const parser::OmpEndLoopDirective &) {
    messageHandler().set_currStmtSource(std::nullopt);
  }

  bool Pre(const parser::OpenMPSectionsConstruct &) {
    PushScope(Scope::Kind::OtherConstruct, nullptr);
    return true;
  }
  void Post(const parser::OpenMPSectionsConstruct &) { PopScope(); }
  bool Pre(const parser::OmpBeginSectionsDirective &x) {
    AddOmpSourceRange(x.source);
    return true;
  }
  void Post(const parser::OmpBeginSectionsDirective &) {
    messageHandler().set_currStmtSource(std::nullopt);
  }
  bool Pre(const parser::OmpEndSectionsDirective &x) {
    AddOmpSourceRange(x.source);
    return true;
  }
  void Post(const parser::OmpEndSectionsDirective &) {
    messageHandler().set_currStmtSource(std::nullopt);
  }
};

bool OmpVisitor::NeedsScope(const parser::OpenMPBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::OmpBeginBlockDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
  switch (beginDir.v) {
  case llvm::omp::Directive::OMPD_target_data:
  case llvm::omp::Directive::OMPD_master:
  case llvm::omp::Directive::OMPD_ordered:
  case llvm::omp::Directive::OMPD_taskgroup:
    return false;
  default:
    return true;
  }
}

void OmpVisitor::AddOmpSourceRange(const parser::CharBlock &source) {
  messageHandler().set_currStmtSource(source);
  currScope().AddSourceRange(source);
}

bool OmpVisitor::Pre(const parser::OpenMPBlockConstruct &x) {
  if (NeedsScope(x)) {
    PushScope(Scope::Kind::OtherConstruct, nullptr);
  }
  return true;
}

void OmpVisitor::Post(const parser::OpenMPBlockConstruct &x) {
  if (NeedsScope(x)) {
    PopScope();
  }
}

// Walk the parse tree and resolve names to symbols.
class ResolveNamesVisitor : public virtual ScopeHandler,
                            public ModuleVisitor,
                            public SubprogramVisitor,
                            public ConstructVisitor,
                            public OmpVisitor,
                            public AccVisitor {
public:
  using AccVisitor::Post;
  using AccVisitor::Pre;
  using ArraySpecVisitor::Post;
  using ConstructVisitor::Post;
  using ConstructVisitor::Pre;
  using DeclarationVisitor::Post;
  using DeclarationVisitor::Pre;
  using ImplicitRulesVisitor::Post;
  using ImplicitRulesVisitor::Pre;
  using InterfaceVisitor::Post;
  using InterfaceVisitor::Pre;
  using ModuleVisitor::Post;
  using ModuleVisitor::Pre;
  using OmpVisitor::Post;
  using OmpVisitor::Pre;
  using ScopeHandler::Post;
  using ScopeHandler::Pre;
  using SubprogramVisitor::Post;
  using SubprogramVisitor::Pre;

  ResolveNamesVisitor(
      SemanticsContext &context, ImplicitRulesMap &rules, Scope &top)
      : BaseVisitor{context, *this, rules}, topScope_{top} {
    PushScope(top);
  }

  Scope &topScope() const { return topScope_; }

  // Default action for a parse tree node is to visit children.
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::SpecificationPart &);
  void Post(const parser::Program &);
  bool Pre(const parser::ImplicitStmt &);
  void Post(const parser::PointerObject &);
  void Post(const parser::AllocateObject &);
  bool Pre(const parser::PointerAssignmentStmt &);
  void Post(const parser::Designator &);
  void Post(const parser::SubstringInquiry &);
  template <typename A, typename B>
  void Post(const parser::LoopBounds<A, B> &x) {
    ResolveName(*parser::Unwrap<parser::Name>(x.name));
  }
  void Post(const parser::ProcComponentRef &);
  bool Pre(const parser::FunctionReference &);
  bool Pre(const parser::CallStmt &);
  bool Pre(const parser::ImportStmt &);
  void Post(const parser::TypeGuardStmt &);
  bool Pre(const parser::StmtFunctionStmt &);
  bool Pre(const parser::DefinedOpName &);
  bool Pre(const parser::ProgramUnit &);
  void Post(const parser::AssignStmt &);
  void Post(const parser::AssignedGotoStmt &);

  // These nodes should never be reached: they are handled in ProgramUnit
  bool Pre(const parser::MainProgram &) {
    llvm_unreachable("This node is handled in ProgramUnit");
  }
  bool Pre(const parser::FunctionSubprogram &) {
    llvm_unreachable("This node is handled in ProgramUnit");
  }
  bool Pre(const parser::SubroutineSubprogram &) {
    llvm_unreachable("This node is handled in ProgramUnit");
  }
  bool Pre(const parser::SeparateModuleSubprogram &) {
    llvm_unreachable("This node is handled in ProgramUnit");
  }
  bool Pre(const parser::Module &) {
    llvm_unreachable("This node is handled in ProgramUnit");
  }
  bool Pre(const parser::Submodule &) {
    llvm_unreachable("This node is handled in ProgramUnit");
  }
  bool Pre(const parser::BlockData &) {
    llvm_unreachable("This node is handled in ProgramUnit");
  }

  void NoteExecutablePartCall(Symbol::Flag, const parser::Call &);

  friend void ResolveSpecificationParts(SemanticsContext &, const Symbol &);

private:
  // Kind of procedure we are expecting to see in a ProcedureDesignator
  std::optional<Symbol::Flag> expectedProcFlag_;
  std::optional<SourceName> prevImportStmt_;
  Scope &topScope_;

  void PreSpecificationConstruct(const parser::SpecificationConstruct &);
  void CreateCommonBlockSymbols(const parser::CommonStmt &);
  void CreateGeneric(const parser::GenericSpec &);
  void FinishSpecificationPart(const std::list<parser::DeclarationConstruct> &);
  void AnalyzeStmtFunctionStmt(const parser::StmtFunctionStmt &);
  void CheckImports();
  void CheckImport(const SourceName &, const SourceName &);
  void HandleCall(Symbol::Flag, const parser::Call &);
  void HandleProcedureName(Symbol::Flag, const parser::Name &);
  bool CheckImplicitNoneExternal(const SourceName &, const Symbol &);
  bool SetProcFlag(const parser::Name &, Symbol &, Symbol::Flag);
  void ResolveSpecificationParts(ProgramTree &);
  void AddSubpNames(ProgramTree &);
  bool BeginScopeForNode(const ProgramTree &);
  void EndScopeForNode(const ProgramTree &);
  void FinishSpecificationParts(const ProgramTree &);
  void FinishDerivedTypeInstantiation(Scope &);
  void ResolveExecutionParts(const ProgramTree &);
};

// ImplicitRules implementation

bool ImplicitRules::isImplicitNoneType() const {
  if (isImplicitNoneType_) {
    return true;
  } else if (map_.empty() && inheritFromParent_) {
    return parent_->isImplicitNoneType();
  } else {
    return false; // default if not specified
  }
}

bool ImplicitRules::isImplicitNoneExternal() const {
  if (isImplicitNoneExternal_) {
    return true;
  } else if (inheritFromParent_) {
    return parent_->isImplicitNoneExternal();
  } else {
    return false; // default if not specified
  }
}

const DeclTypeSpec *ImplicitRules::GetType(
    SourceName name, bool respectImplicitNoneType) const {
  char ch{name.begin()[0]};
  if (isImplicitNoneType_ && respectImplicitNoneType) {
    return nullptr;
  } else if (auto it{map_.find(ch)}; it != map_.end()) {
    return &*it->second;
  } else if (inheritFromParent_) {
    return parent_->GetType(name, respectImplicitNoneType);
  } else if (ch >= 'i' && ch <= 'n') {
    return &context_.MakeNumericType(TypeCategory::Integer);
  } else if (ch >= 'a' && ch <= 'z') {
    return &context_.MakeNumericType(TypeCategory::Real);
  } else {
    return nullptr;
  }
}

void ImplicitRules::SetTypeMapping(const DeclTypeSpec &type,
    parser::Location fromLetter, parser::Location toLetter) {
  for (char ch = *fromLetter; ch; ch = ImplicitRules::Incr(ch)) {
    auto res{map_.emplace(ch, type)};
    if (!res.second) {
      context_.Say(parser::CharBlock{fromLetter},
          "More than one implicit type specified for '%c'"_err_en_US, ch);
    }
    if (ch == *toLetter) {
      break;
    }
  }
}

// Return the next char after ch in a way that works for ASCII or EBCDIC.
// Return '\0' for the char after 'z'.
char ImplicitRules::Incr(char ch) {
  switch (ch) {
  case 'i':
    return 'j';
  case 'r':
    return 's';
  case 'z':
    return '\0';
  default:
    return ch + 1;
  }
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &o, const ImplicitRules &implicitRules) {
  o << "ImplicitRules:\n";
  for (char ch = 'a'; ch; ch = ImplicitRules::Incr(ch)) {
    ShowImplicitRule(o, implicitRules, ch);
  }
  ShowImplicitRule(o, implicitRules, '_');
  ShowImplicitRule(o, implicitRules, '$');
  ShowImplicitRule(o, implicitRules, '@');
  return o;
}
void ShowImplicitRule(
    llvm::raw_ostream &o, const ImplicitRules &implicitRules, char ch) {
  auto it{implicitRules.map_.find(ch)};
  if (it != implicitRules.map_.end()) {
    o << "  " << ch << ": " << *it->second << '\n';
  }
}

template <typename T> void BaseVisitor::Walk(const T &x) {
  parser::Walk(x, *this_);
}

void BaseVisitor::MakePlaceholder(
    const parser::Name &name, MiscDetails::Kind kind) {
  if (!name.symbol) {
    name.symbol = &context_->globalScope().MakeSymbol(
        name.source, Attrs{}, MiscDetails{kind});
  }
}

// AttrsVisitor implementation

bool AttrsVisitor::BeginAttrs() {
  CHECK(!attrs_);
  attrs_ = std::make_optional<Attrs>();
  return true;
}
Attrs AttrsVisitor::GetAttrs() {
  CHECK(attrs_);
  return *attrs_;
}
Attrs AttrsVisitor::EndAttrs() {
  Attrs result{GetAttrs()};
  attrs_.reset();
  passName_ = std::nullopt;
  bindName_.reset();
  return result;
}

bool AttrsVisitor::SetPassNameOn(Symbol &symbol) {
  if (!passName_) {
    return false;
  }
  common::visit(common::visitors{
                    [&](ProcEntityDetails &x) { x.set_passName(*passName_); },
                    [&](ProcBindingDetails &x) { x.set_passName(*passName_); },
                    [](auto &) { common::die("unexpected pass name"); },
                },
      symbol.details());
  return true;
}

void AttrsVisitor::SetBindNameOn(Symbol &symbol) {
  if (!attrs_ || !attrs_->test(Attr::BIND_C)) {
    return;
  }
  std::optional<std::string> label{
      evaluate::GetScalarConstantValue<evaluate::Ascii>(bindName_)};
  if (ClassifyProcedure(symbol) == ProcedureDefinitionClass::Internal) {
    if (label) { // C1552: no NAME= allowed even if null
      Say(symbol.name(),
          "An internal procedure may not have a BIND(C,NAME=) binding label"_err_en_US);
    }
    return;
  }
  // 18.9.2(2): discard leading and trailing blanks
  if (label) {
    auto first{label->find_first_not_of(" ")};
    if (first == std::string::npos) {
      // Empty NAME= means no binding at all (18.10.2p2)
      return;
    }
    auto last{label->find_last_not_of(" ")};
    label = label->substr(first, last - first + 1);
  } else {
    label = parser::ToLowerCaseLetters(symbol.name().ToString());
  }
  // Check if a symbol has two Bind names.
  std::string oldBindName;
  if (symbol.GetBindName()) {
    oldBindName = *symbol.GetBindName();
  }
  symbol.SetBindName(std::move(*label));
  if (!oldBindName.empty()) {
    if (const std::string * newBindName{symbol.GetBindName()}) {
      if (oldBindName != *newBindName) {
        Say(symbol.name(), "The entity '%s' has multiple BIND names"_err_en_US);
      }
    }
  }
}

void AttrsVisitor::Post(const parser::LanguageBindingSpec &x) {
  CHECK(attrs_);
  if (CheckAndSet(Attr::BIND_C)) {
    if (x.v) {
      bindName_ = EvaluateExpr(*x.v);
    }
  }
}
bool AttrsVisitor::Pre(const parser::IntentSpec &x) {
  CHECK(attrs_);
  CheckAndSet(IntentSpecToAttr(x));
  return false;
}
bool AttrsVisitor::Pre(const parser::Pass &x) {
  if (CheckAndSet(Attr::PASS)) {
    if (x.v) {
      passName_ = x.v->source;
      MakePlaceholder(*x.v, MiscDetails::Kind::PassName);
    }
  }
  return false;
}

// C730, C743, C755, C778, C1543 say no attribute or prefix repetitions
bool AttrsVisitor::IsDuplicateAttr(Attr attrName) {
  if (attrs_->test(attrName)) {
    Say(currStmtSource().value(),
        "Attribute '%s' cannot be used more than once"_warn_en_US,
        AttrToString(attrName));
    return true;
  }
  return false;
}

// See if attrName violates a constraint cause by a conflict.  attr1 and attr2
// name attributes that cannot be used on the same declaration
bool AttrsVisitor::HaveAttrConflict(Attr attrName, Attr attr1, Attr attr2) {
  if ((attrName == attr1 && attrs_->test(attr2)) ||
      (attrName == attr2 && attrs_->test(attr1))) {
    Say(currStmtSource().value(),
        "Attributes '%s' and '%s' conflict with each other"_err_en_US,
        AttrToString(attr1), AttrToString(attr2));
    return true;
  }
  return false;
}
// C759, C1543
bool AttrsVisitor::IsConflictingAttr(Attr attrName) {
  return HaveAttrConflict(attrName, Attr::INTENT_IN, Attr::INTENT_INOUT) ||
      HaveAttrConflict(attrName, Attr::INTENT_IN, Attr::INTENT_OUT) ||
      HaveAttrConflict(attrName, Attr::INTENT_INOUT, Attr::INTENT_OUT) ||
      HaveAttrConflict(attrName, Attr::PASS, Attr::NOPASS) || // C781
      HaveAttrConflict(attrName, Attr::PURE, Attr::IMPURE) ||
      HaveAttrConflict(attrName, Attr::PUBLIC, Attr::PRIVATE) ||
      HaveAttrConflict(attrName, Attr::RECURSIVE, Attr::NON_RECURSIVE);
}
bool AttrsVisitor::CheckAndSet(Attr attrName) {
  CHECK(attrs_);
  if (IsConflictingAttr(attrName) || IsDuplicateAttr(attrName)) {
    return false;
  }
  attrs_->set(attrName);
  return true;
}

// DeclTypeSpecVisitor implementation

const DeclTypeSpec *DeclTypeSpecVisitor::GetDeclTypeSpec() {
  return state_.declTypeSpec;
}

void DeclTypeSpecVisitor::BeginDeclTypeSpec() {
  CHECK(!state_.expectDeclTypeSpec);
  CHECK(!state_.declTypeSpec);
  state_.expectDeclTypeSpec = true;
}
void DeclTypeSpecVisitor::EndDeclTypeSpec() {
  CHECK(state_.expectDeclTypeSpec);
  state_ = {};
}

void DeclTypeSpecVisitor::SetDeclTypeSpecCategory(
    DeclTypeSpec::Category category) {
  CHECK(state_.expectDeclTypeSpec);
  state_.derived.category = category;
}

bool DeclTypeSpecVisitor::Pre(const parser::TypeGuardStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclTypeSpecVisitor::Post(const parser::TypeGuardStmt &) {
  EndDeclTypeSpec();
}

void DeclTypeSpecVisitor::Post(const parser::TypeSpec &typeSpec) {
  // Record the resolved DeclTypeSpec in the parse tree for use by
  // expression semantics if the DeclTypeSpec is a valid TypeSpec.
  // The grammar ensures that it's an intrinsic or derived type spec,
  // not TYPE(*) or CLASS(*) or CLASS(T).
  if (const DeclTypeSpec * spec{state_.declTypeSpec}) {
    switch (spec->category()) {
    case DeclTypeSpec::Numeric:
    case DeclTypeSpec::Logical:
    case DeclTypeSpec::Character:
      typeSpec.declTypeSpec = spec;
      break;
    case DeclTypeSpec::TypeDerived:
      if (const DerivedTypeSpec * derived{spec->AsDerived()}) {
        CheckForAbstractType(derived->typeSymbol()); // C703
        typeSpec.declTypeSpec = spec;
      }
      break;
    default:
      CRASH_NO_CASE;
    }
  }
}

void DeclTypeSpecVisitor::Post(
    const parser::IntrinsicTypeSpec::DoublePrecision &) {
  MakeNumericType(TypeCategory::Real, context().doublePrecisionKind());
}
void DeclTypeSpecVisitor::Post(
    const parser::IntrinsicTypeSpec::DoubleComplex &) {
  MakeNumericType(TypeCategory::Complex, context().doublePrecisionKind());
}
void DeclTypeSpecVisitor::MakeNumericType(TypeCategory category, int kind) {
  SetDeclTypeSpec(context().MakeNumericType(category, kind));
}

void DeclTypeSpecVisitor::CheckForAbstractType(const Symbol &typeSymbol) {
  if (typeSymbol.attrs().test(Attr::ABSTRACT)) {
    Say("ABSTRACT derived type may not be used here"_err_en_US);
  }
}

void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::ClassStar &) {
  SetDeclTypeSpec(context().globalScope().MakeClassStarType());
}
void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::TypeStar &) {
  SetDeclTypeSpec(context().globalScope().MakeTypeStarType());
}

// Check that we're expecting to see a DeclTypeSpec (and haven't seen one yet)
// and save it in state_.declTypeSpec.
void DeclTypeSpecVisitor::SetDeclTypeSpec(const DeclTypeSpec &declTypeSpec) {
  CHECK(state_.expectDeclTypeSpec);
  CHECK(!state_.declTypeSpec);
  state_.declTypeSpec = &declTypeSpec;
}

KindExpr DeclTypeSpecVisitor::GetKindParamExpr(
    TypeCategory category, const std::optional<parser::KindSelector> &kind) {
  return AnalyzeKindSelector(context(), category, kind);
}

// MessageHandler implementation

Message &MessageHandler::Say(MessageFixedText &&msg) {
  return context_->Say(currStmtSource().value(), std::move(msg));
}
Message &MessageHandler::Say(MessageFormattedText &&msg) {
  return context_->Say(currStmtSource().value(), std::move(msg));
}
Message &MessageHandler::Say(const SourceName &name, MessageFixedText &&msg) {
  return Say(name, std::move(msg), name);
}

// ImplicitRulesVisitor implementation

void ImplicitRulesVisitor::Post(const parser::ParameterStmt &) {
  prevParameterStmt_ = currStmtSource();
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitStmt &x) {
  bool result{
      common::visit(common::visitors{
                        [&](const std::list<ImplicitNoneNameSpec> &y) {
                          return HandleImplicitNone(y);
                        },
                        [&](const std::list<parser::ImplicitSpec> &) {
                          if (prevImplicitNoneType_) {
                            Say("IMPLICIT statement after IMPLICIT NONE or "
                                "IMPLICIT NONE(TYPE) statement"_err_en_US);
                            return false;
                          }
                          implicitRules_->set_isImplicitNoneType(false);
                          return true;
                        },
                    },
          x.u)};
  prevImplicit_ = currStmtSource();
  return result;
}

bool ImplicitRulesVisitor::Pre(const parser::LetterSpec &x) {
  auto loLoc{std::get<parser::Location>(x.t)};
  auto hiLoc{loLoc};
  if (auto hiLocOpt{std::get<std::optional<parser::Location>>(x.t)}) {
    hiLoc = *hiLocOpt;
    if (*hiLoc < *loLoc) {
      Say(hiLoc, "'%s' does not follow '%s' alphabetically"_err_en_US,
          std::string(hiLoc, 1), std::string(loLoc, 1));
      return false;
    }
  }
  implicitRules_->SetTypeMapping(*GetDeclTypeSpec(), loLoc, hiLoc);
  return false;
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitSpec &) {
  BeginDeclTypeSpec();
  set_allowForwardReferenceToDerivedType(true);
  return true;
}

void ImplicitRulesVisitor::Post(const parser::ImplicitSpec &) {
  EndDeclTypeSpec();
}

void ImplicitRulesVisitor::SetScope(const Scope &scope) {
  implicitRules_ = &DEREF(implicitRulesMap_).at(&scope);
  prevImplicit_ = std::nullopt;
  prevImplicitNone_ = std::nullopt;
  prevImplicitNoneType_ = std::nullopt;
  prevParameterStmt_ = std::nullopt;
}
void ImplicitRulesVisitor::BeginScope(const Scope &scope) {
  // find or create implicit rules for this scope
  DEREF(implicitRulesMap_).try_emplace(&scope, context(), implicitRules_);
  SetScope(scope);
}

// TODO: for all of these errors, reference previous statement too
bool ImplicitRulesVisitor::HandleImplicitNone(
    const std::list<ImplicitNoneNameSpec> &nameSpecs) {
  if (prevImplicitNone_) {
    Say("More than one IMPLICIT NONE statement"_err_en_US);
    Say(*prevImplicitNone_, "Previous IMPLICIT NONE statement"_en_US);
    return false;
  }
  if (prevParameterStmt_) {
    Say("IMPLICIT NONE statement after PARAMETER statement"_err_en_US);
    return false;
  }
  prevImplicitNone_ = currStmtSource();
  bool implicitNoneTypeNever{
      context().IsEnabled(common::LanguageFeature::ImplicitNoneTypeNever)};
  if (nameSpecs.empty()) {
    if (!implicitNoneTypeNever) {
      prevImplicitNoneType_ = currStmtSource();
      implicitRules_->set_isImplicitNoneType(true);
      if (prevImplicit_) {
        Say("IMPLICIT NONE statement after IMPLICIT statement"_err_en_US);
        return false;
      }
    }
  } else {
    int sawType{0};
    int sawExternal{0};
    for (const auto noneSpec : nameSpecs) {
      switch (noneSpec) {
      case ImplicitNoneNameSpec::External:
        implicitRules_->set_isImplicitNoneExternal(true);
        ++sawExternal;
        break;
      case ImplicitNoneNameSpec::Type:
        if (!implicitNoneTypeNever) {
          prevImplicitNoneType_ = currStmtSource();
          implicitRules_->set_isImplicitNoneType(true);
          if (prevImplicit_) {
            Say("IMPLICIT NONE(TYPE) after IMPLICIT statement"_err_en_US);
            return false;
          }
          ++sawType;
        }
        break;
      }
    }
    if (sawType > 1) {
      Say("TYPE specified more than once in IMPLICIT NONE statement"_err_en_US);
      return false;
    }
    if (sawExternal > 1) {
      Say("EXTERNAL specified more than once in IMPLICIT NONE statement"_err_en_US);
      return false;
    }
  }
  return true;
}

// ArraySpecVisitor implementation

void ArraySpecVisitor::Post(const parser::ArraySpec &x) {
  CHECK(arraySpec_.empty());
  arraySpec_ = AnalyzeArraySpec(context(), x);
}
void ArraySpecVisitor::Post(const parser::ComponentArraySpec &x) {
  CHECK(arraySpec_.empty());
  arraySpec_ = AnalyzeArraySpec(context(), x);
}
void ArraySpecVisitor::Post(const parser::CoarraySpec &x) {
  CHECK(coarraySpec_.empty());
  coarraySpec_ = AnalyzeCoarraySpec(context(), x);
}

const ArraySpec &ArraySpecVisitor::arraySpec() {
  return !arraySpec_.empty() ? arraySpec_ : attrArraySpec_;
}
const ArraySpec &ArraySpecVisitor::coarraySpec() {
  return !coarraySpec_.empty() ? coarraySpec_ : attrCoarraySpec_;
}
void ArraySpecVisitor::BeginArraySpec() {
  CHECK(arraySpec_.empty());
  CHECK(coarraySpec_.empty());
  CHECK(attrArraySpec_.empty());
  CHECK(attrCoarraySpec_.empty());
}
void ArraySpecVisitor::EndArraySpec() {
  CHECK(arraySpec_.empty());
  CHECK(coarraySpec_.empty());
  attrArraySpec_.clear();
  attrCoarraySpec_.clear();
}
void ArraySpecVisitor::PostAttrSpec() {
  // Save dimension/codimension from attrs so we can process array/coarray-spec
  // on the entity-decl
  if (!arraySpec_.empty()) {
    if (attrArraySpec_.empty()) {
      attrArraySpec_ = arraySpec_;
      arraySpec_.clear();
    } else {
      Say(currStmtSource().value(),
          "Attribute 'DIMENSION' cannot be used more than once"_err_en_US);
    }
  }
  if (!coarraySpec_.empty()) {
    if (attrCoarraySpec_.empty()) {
      attrCoarraySpec_ = coarraySpec_;
      coarraySpec_.clear();
    } else {
      Say(currStmtSource().value(),
          "Attribute 'CODIMENSION' cannot be used more than once"_err_en_US);
    }
  }
}

// FuncResultStack implementation

FuncResultStack::~FuncResultStack() { CHECK(stack_.empty()); }

void FuncResultStack::CompleteFunctionResultType() {
  // If the function has a type in the prefix, process it now.
  FuncInfo *info{Top()};
  if (info && &info->scope == &scopeHandler_.currScope()) {
    if (info->parsedType) {
      scopeHandler_.messageHandler().set_currStmtSource(info->source);
      if (const auto *type{
              scopeHandler_.ProcessTypeSpec(*info->parsedType, true)}) {
        if (!scopeHandler_.context().HasError(info->resultSymbol)) {
          info->resultSymbol->SetType(*type);
        }
      }
      info->parsedType = nullptr;
    }
  }
}

// Called from ConvertTo{Object/Proc}Entity to cope with any appearance
// of the function result in a specification expression.
void FuncResultStack::CompleteTypeIfFunctionResult(Symbol &symbol) {
  if (FuncInfo * info{Top()}) {
    if (info->resultSymbol == &symbol) {
      CompleteFunctionResultType();
    }
  }
}

void FuncResultStack::Pop() {
  if (!stack_.empty() && &stack_.back().scope == &scopeHandler_.currScope()) {
    stack_.pop_back();
  }
}

// ScopeHandler implementation

void ScopeHandler::SayAlreadyDeclared(const parser::Name &name, Symbol &prev) {
  SayAlreadyDeclared(name.source, prev);
}
void ScopeHandler::SayAlreadyDeclared(const SourceName &name, Symbol &prev) {
  if (context().HasError(prev)) {
    // don't report another error about prev
  } else {
    if (const auto *details{prev.detailsIf<UseDetails>()}) {
      Say(name, "'%s' is already declared in this scoping unit"_err_en_US)
          .Attach(details->location(),
              "It is use-associated with '%s' in module '%s'"_en_US,
              details->symbol().name(), GetUsedModule(*details).name());
    } else {
      SayAlreadyDeclared(name, prev.name());
    }
    context().SetError(prev);
  }
}
void ScopeHandler::SayAlreadyDeclared(
    const SourceName &name1, const SourceName &name2) {
  if (name1.begin() < name2.begin()) {
    SayAlreadyDeclared(name2, name1);
  } else {
    Say(name1, "'%s' is already declared in this scoping unit"_err_en_US)
        .Attach(name2, "Previous declaration of '%s'"_en_US, name2);
  }
}

void ScopeHandler::SayWithReason(const parser::Name &name, Symbol &symbol,
    MessageFixedText &&msg1, Message &&msg2) {
  Say(name, std::move(msg1), symbol.name()).Attach(std::move(msg2));
  context().SetError(symbol, msg1.isFatal());
}

void ScopeHandler::SayWithDecl(
    const parser::Name &name, Symbol &symbol, MessageFixedText &&msg) {
  Say(name, std::move(msg), symbol.name())
      .Attach(Message{name.source,
          symbol.test(Symbol::Flag::Implicit)
              ? "Implicit declaration of '%s'"_en_US
              : "Declaration of '%s'"_en_US,
          name.source});
  context().SetError(symbol, msg.isFatal());
}

void ScopeHandler::SayLocalMustBeVariable(
    const parser::Name &name, Symbol &symbol) {
  SayWithDecl(name, symbol,
      "The name '%s' must be a variable to appear"
      " in a locality-spec"_err_en_US);
}

void ScopeHandler::SayDerivedType(
    const SourceName &name, MessageFixedText &&msg, const Scope &type) {
  const Symbol &typeSymbol{DEREF(type.GetSymbol())};
  Say(name, std::move(msg), name, typeSymbol.name())
      .Attach(typeSymbol.name(), "Declaration of derived type '%s'"_en_US,
          typeSymbol.name());
}
void ScopeHandler::Say2(const SourceName &name1, MessageFixedText &&msg1,
    const SourceName &name2, MessageFixedText &&msg2) {
  Say(name1, std::move(msg1)).Attach(name2, std::move(msg2), name2);
}
void ScopeHandler::Say2(const SourceName &name, MessageFixedText &&msg1,
    Symbol &symbol, MessageFixedText &&msg2) {
  Say2(name, std::move(msg1), symbol.name(), std::move(msg2));
  context().SetError(symbol, msg1.isFatal());
}
void ScopeHandler::Say2(const parser::Name &name, MessageFixedText &&msg1,
    Symbol &symbol, MessageFixedText &&msg2) {
  Say2(name.source, std::move(msg1), symbol.name(), std::move(msg2));
  context().SetError(symbol, msg1.isFatal());
}

// This is essentially GetProgramUnitContaining(), but it can return
// a mutable Scope &, it ignores statement functions, and it fails
// gracefully for error recovery (returning the original Scope).
template <typename T> static T &GetInclusiveScope(T &scope) {
  for (T *s{&scope}; !s->IsGlobal(); s = &s->parent()) {
    switch (s->kind()) {
    case Scope::Kind::Module:
    case Scope::Kind::MainProgram:
    case Scope::Kind::Subprogram:
    case Scope::Kind::BlockData:
      if (!s->IsStmtFunction()) {
        return *s;
      }
      break;
    default:;
    }
  }
  return scope;
}

Scope &ScopeHandler::InclusiveScope() { return GetInclusiveScope(currScope()); }

Scope *ScopeHandler::GetHostProcedure() {
  Scope &parent{InclusiveScope().parent()};
  switch (parent.kind()) {
  case Scope::Kind::Subprogram:
    return &parent;
  case Scope::Kind::MainProgram:
    return &parent;
  default:
    return nullptr;
  }
}

Scope &ScopeHandler::NonDerivedTypeScope() {
  return currScope_->IsDerivedType() ? currScope_->parent() : *currScope_;
}

void ScopeHandler::PushScope(Scope::Kind kind, Symbol *symbol) {
  PushScope(currScope().MakeScope(kind, symbol));
}
void ScopeHandler::PushScope(Scope &scope) {
  currScope_ = &scope;
  auto kind{currScope_->kind()};
  if (kind != Scope::Kind::BlockConstruct &&
      kind != Scope::Kind::OtherConstruct) {
    BeginScope(scope);
  }
  // The name of a module or submodule cannot be "used" in its scope,
  // as we read 19.3.1(2), so we allow the name to be used as a local
  // identifier in the module or submodule too.  Same with programs
  // (14.1(3)) and BLOCK DATA.
  if (!currScope_->IsDerivedType() && kind != Scope::Kind::Module &&
      kind != Scope::Kind::MainProgram && kind != Scope::Kind::BlockData) {
    if (auto *symbol{scope.symbol()}) {
      // Create a dummy symbol so we can't create another one with the same
      // name. It might already be there if we previously pushed the scope.
      SourceName name{symbol->name()};
      if (!FindInScope(scope, name)) {
        auto &newSymbol{MakeSymbol(name)};
        if (kind == Scope::Kind::Subprogram) {
          // Allow for recursive references.  If this symbol is a function
          // without an explicit RESULT(), this new symbol will be discarded
          // and replaced with an object of the same name.
          newSymbol.set_details(HostAssocDetails{*symbol});
        } else {
          newSymbol.set_details(MiscDetails{MiscDetails::Kind::ScopeName});
        }
      }
    }
  }
}
void ScopeHandler::PopScope() {
  // Entities that are not yet classified as objects or procedures are now
  // assumed to be objects.
  // TODO: Statement functions
  for (auto &pair : currScope()) {
    ConvertToObjectEntity(*pair.second);
  }
  funcResultStack_.Pop();
  // If popping back into a global scope, pop back to the main global scope.
  SetScope(currScope_->parent().IsGlobal() ? context().globalScope()
                                           : currScope_->parent());
}
void ScopeHandler::SetScope(Scope &scope) {
  currScope_ = &scope;
  ImplicitRulesVisitor::SetScope(InclusiveScope());
}

Symbol *ScopeHandler::FindSymbol(const parser::Name &name) {
  return FindSymbol(currScope(), name);
}
Symbol *ScopeHandler::FindSymbol(const Scope &scope, const parser::Name &name) {
  if (scope.IsDerivedType()) {
    if (Symbol * symbol{scope.FindComponent(name.source)}) {
      if (!symbol->has<ProcBindingDetails>() &&
          !symbol->has<GenericDetails>() &&
          !symbol->test(Symbol::Flag::ParentComp)) {
        return Resolve(name, symbol);
      }
    }
    return FindSymbol(scope.parent(), name);
  } else {
    // In EQUIVALENCE statements only resolve names in the local scope, see
    // 19.5.1.4, paragraph 2, item (10)
    return Resolve(name,
        inEquivalenceStmt_ ? FindInScope(scope, name)
                           : scope.FindSymbol(name.source));
  }
}

Symbol &ScopeHandler::MakeSymbol(
    Scope &scope, const SourceName &name, Attrs attrs) {
  if (Symbol * symbol{FindInScope(scope, name)}) {
    symbol->attrs() |= attrs;
    return *symbol;
  } else {
    const auto pair{scope.try_emplace(name, attrs, UnknownDetails{})};
    CHECK(pair.second); // name was not found, so must be able to add
    return *pair.first->second;
  }
}
Symbol &ScopeHandler::MakeSymbol(const SourceName &name, Attrs attrs) {
  return MakeSymbol(currScope(), name, attrs);
}
Symbol &ScopeHandler::MakeSymbol(const parser::Name &name, Attrs attrs) {
  return Resolve(name, MakeSymbol(name.source, attrs));
}
Symbol &ScopeHandler::MakeHostAssocSymbol(
    const parser::Name &name, const Symbol &hostSymbol) {
  Symbol &symbol{*NonDerivedTypeScope()
                      .try_emplace(name.source, HostAssocDetails{hostSymbol})
                      .first->second};
  name.symbol = &symbol;
  symbol.attrs() = hostSymbol.attrs(); // TODO: except PRIVATE, PUBLIC?
  symbol.flags() = hostSymbol.flags();
  return symbol;
}
Symbol &ScopeHandler::CopySymbol(const SourceName &name, const Symbol &symbol) {
  CHECK(!FindInScope(name));
  return MakeSymbol(currScope(), name, symbol.attrs());
}

// Look for name only in scope, not in enclosing scopes.
Symbol *ScopeHandler::FindInScope(
    const Scope &scope, const parser::Name &name) {
  return Resolve(name, FindInScope(scope, name.source));
}
Symbol *ScopeHandler::FindInScope(const Scope &scope, const SourceName &name) {
  // all variants of names, e.g. "operator(.ne.)" for "operator(/=)"
  for (const std::string &n : GetAllNames(context(), name)) {
    auto it{scope.find(SourceName{n})};
    if (it != scope.end()) {
      return &*it->second;
    }
  }
  return nullptr;
}

// Find a component or type parameter by name in a derived type or its parents.
Symbol *ScopeHandler::FindInTypeOrParents(
    const Scope &scope, const parser::Name &name) {
  return Resolve(name, scope.FindComponent(name.source));
}
Symbol *ScopeHandler::FindInTypeOrParents(const parser::Name &name) {
  return FindInTypeOrParents(currScope(), name);
}

void ScopeHandler::EraseSymbol(const parser::Name &name) {
  currScope().erase(name.source);
  name.symbol = nullptr;
}

static bool NeedsType(const Symbol &symbol) {
  return !symbol.GetType() &&
      common::visit(common::visitors{
                        [](const EntityDetails &) { return true; },
                        [](const ObjectEntityDetails &) { return true; },
                        [](const AssocEntityDetails &) { return true; },
                        [&](const ProcEntityDetails &p) {
                          return symbol.test(Symbol::Flag::Function) &&
                              !symbol.attrs().test(Attr::INTRINSIC) &&
                              !p.interface().type() && !p.interface().symbol();
                        },
                        [](const auto &) { return false; },
                    },
          symbol.details());
}

void ScopeHandler::ApplyImplicitRules(
    Symbol &symbol, bool allowForwardReference) {
  funcResultStack_.CompleteTypeIfFunctionResult(symbol);
  if (context().HasError(symbol) || !NeedsType(symbol)) {
    return;
  }
  if (const DeclTypeSpec * type{GetImplicitType(symbol)}) {
    symbol.set(Symbol::Flag::Implicit);
    symbol.SetType(*type);
    return;
  }
  if (symbol.has<ProcEntityDetails>() && !symbol.attrs().test(Attr::EXTERNAL)) {
    std::optional<Symbol::Flag> functionOrSubroutineFlag;
    if (symbol.test(Symbol::Flag::Function)) {
      functionOrSubroutineFlag = Symbol::Flag::Function;
    } else if (symbol.test(Symbol::Flag::Subroutine)) {
      functionOrSubroutineFlag = Symbol::Flag::Subroutine;
    }
    if (IsIntrinsic(symbol.name(), functionOrSubroutineFlag)) {
      // type will be determined in expression semantics
      AcquireIntrinsicProcedureFlags(symbol);
      return;
    }
  }
  if (allowForwardReference && ImplicitlyTypeForwardRef(symbol)) {
    return;
  }
  if (!context().HasError(symbol)) {
    Say(symbol.name(), "No explicit type declared for '%s'"_err_en_US);
    context().SetError(symbol);
  }
}

// Extension: Allow forward references to scalar integer dummy arguments
// to appear in specification expressions under IMPLICIT NONE(TYPE) when
// what would otherwise have been their implicit type is default INTEGER.
bool ScopeHandler::ImplicitlyTypeForwardRef(Symbol &symbol) {
  if (!inSpecificationPart_ || context().HasError(symbol) || !IsDummy(symbol) ||
      symbol.Rank() != 0 ||
      !context().languageFeatures().IsEnabled(
          common::LanguageFeature::ForwardRefDummyImplicitNone)) {
    return false;
  }
  const DeclTypeSpec *type{
      GetImplicitType(symbol, false /*ignore IMPLICIT NONE*/)};
  if (!type || !type->IsNumeric(TypeCategory::Integer)) {
    return false;
  }
  auto kind{evaluate::ToInt64(type->numericTypeSpec().kind())};
  if (!kind || *kind != context().GetDefaultKind(TypeCategory::Integer)) {
    return false;
  }
  if (!ConvertToObjectEntity(symbol)) {
    return false;
  }
  // TODO: check no INTENT(OUT)?
  if (context().languageFeatures().ShouldWarn(
          common::LanguageFeature::ForwardRefDummyImplicitNone)) {
    Say(symbol.name(),
        "Dummy argument '%s' was used without being explicitly typed"_warn_en_US,
        symbol.name());
  }
  symbol.set(Symbol::Flag::Implicit);
  symbol.SetType(*type);
  return true;
}

// Ensure that the symbol for an intrinsic procedure is marked with
// the INTRINSIC attribute.  Also set PURE &/or ELEMENTAL as
// appropriate.
void ScopeHandler::AcquireIntrinsicProcedureFlags(Symbol &symbol) {
  symbol.attrs().set(Attr::INTRINSIC);
  switch (context().intrinsics().GetIntrinsicClass(symbol.name().ToString())) {
  case evaluate::IntrinsicClass::elementalFunction:
  case evaluate::IntrinsicClass::elementalSubroutine:
    symbol.attrs().set(Attr::ELEMENTAL);
    symbol.attrs().set(Attr::PURE);
    break;
  case evaluate::IntrinsicClass::impureSubroutine:
    break;
  default:
    symbol.attrs().set(Attr::PURE);
  }
}

const DeclTypeSpec *ScopeHandler::GetImplicitType(
    Symbol &symbol, bool respectImplicitNoneType) {
  const Scope *scope{&symbol.owner()};
  if (scope->IsGlobal()) {
    scope = &currScope();
  }
  scope = &GetInclusiveScope(*scope);
  const auto *type{implicitRulesMap_->at(scope).GetType(
      symbol.name(), respectImplicitNoneType)};
  if (type) {
    if (const DerivedTypeSpec * derived{type->AsDerived()}) {
      // Resolve any forward-referenced derived type; a quick no-op else.
      auto &instantiatable{*const_cast<DerivedTypeSpec *>(derived)};
      instantiatable.Instantiate(currScope());
    }
  }
  return type;
}

// Convert symbol to be a ObjectEntity or return false if it can't be.
bool ScopeHandler::ConvertToObjectEntity(Symbol &symbol) {
  if (symbol.has<ObjectEntityDetails>()) {
    // nothing to do
  } else if (symbol.has<UnknownDetails>()) {
    symbol.set_details(ObjectEntityDetails{});
  } else if (auto *details{symbol.detailsIf<EntityDetails>()}) {
    funcResultStack_.CompleteTypeIfFunctionResult(symbol);
    symbol.set_details(ObjectEntityDetails{std::move(*details)});
  } else if (auto *useDetails{symbol.detailsIf<UseDetails>()}) {
    return useDetails->symbol().has<ObjectEntityDetails>();
  } else if (auto *hostDetails{symbol.detailsIf<HostAssocDetails>()}) {
    return hostDetails->symbol().has<ObjectEntityDetails>();
  } else {
    return false;
  }
  return true;
}
// Convert symbol to be a ProcEntity or return false if it can't be.
bool ScopeHandler::ConvertToProcEntity(Symbol &symbol) {
  if (symbol.has<ProcEntityDetails>()) {
    // nothing to do
  } else if (symbol.has<UnknownDetails>()) {
    symbol.set_details(ProcEntityDetails{});
  } else if (auto *details{symbol.detailsIf<EntityDetails>()}) {
    if (IsFunctionResult(symbol) &&
        !(IsPointer(symbol) && symbol.attrs().test(Attr::EXTERNAL))) {
      // Don't turn function result into a procedure pointer unless both
      // POUNTER and EXTERNAL
      return false;
    }
    funcResultStack_.CompleteTypeIfFunctionResult(symbol);
    symbol.set_details(ProcEntityDetails{std::move(*details)});
    if (symbol.GetType() && !symbol.test(Symbol::Flag::Implicit)) {
      CHECK(!symbol.test(Symbol::Flag::Subroutine));
      symbol.set(Symbol::Flag::Function);
    }
  } else if (auto *useDetails{symbol.detailsIf<UseDetails>()}) {
    return useDetails->symbol().has<ProcEntityDetails>();
  } else if (auto *hostDetails{symbol.detailsIf<HostAssocDetails>()}) {
    return hostDetails->symbol().has<ProcEntityDetails>();
  } else {
    return false;
  }
  return true;
}

const DeclTypeSpec &ScopeHandler::MakeNumericType(
    TypeCategory category, const std::optional<parser::KindSelector> &kind) {
  KindExpr value{GetKindParamExpr(category, kind)};
  if (auto known{evaluate::ToInt64(value)}) {
    return context().MakeNumericType(category, static_cast<int>(*known));
  } else {
    return currScope_->MakeNumericType(category, std::move(value));
  }
}

const DeclTypeSpec &ScopeHandler::MakeLogicalType(
    const std::optional<parser::KindSelector> &kind) {
  KindExpr value{GetKindParamExpr(TypeCategory::Logical, kind)};
  if (auto known{evaluate::ToInt64(value)}) {
    return context().MakeLogicalType(static_cast<int>(*known));
  } else {
    return currScope_->MakeLogicalType(std::move(value));
  }
}

void ScopeHandler::NotePossibleBadForwardRef(const parser::Name &name) {
  if (inSpecificationPart_ && name.symbol) {
    auto kind{currScope().kind()};
    if ((kind == Scope::Kind::Subprogram && !currScope().IsStmtFunction()) ||
        kind == Scope::Kind::BlockConstruct) {
      bool isHostAssociated{&name.symbol->owner() == &currScope()
              ? name.symbol->has<HostAssocDetails>()
              : name.symbol->owner().Contains(currScope())};
      if (isHostAssociated) {
        specPartState_.forwardRefs.insert(name.source);
      }
    }
  }
}

std::optional<SourceName> ScopeHandler::HadForwardRef(
    const Symbol &symbol) const {
  auto iter{specPartState_.forwardRefs.find(symbol.name())};
  if (iter != specPartState_.forwardRefs.end()) {
    return *iter;
  }
  return std::nullopt;
}

bool ScopeHandler::CheckPossibleBadForwardRef(const Symbol &symbol) {
  if (!context().HasError(symbol)) {
    if (auto fwdRef{HadForwardRef(symbol)}) {
      const Symbol *outer{symbol.owner().FindSymbol(symbol.name())};
      if (outer && symbol.has<UseDetails>() &&
          &symbol.GetUltimate() == &outer->GetUltimate()) {
        // e.g. IMPORT of host's USE association
        return false;
      }
      Say(*fwdRef,
          "Forward reference to '%s' is not allowed in the same specification part"_err_en_US,
          *fwdRef)
          .Attach(symbol.name(), "Later declaration of '%s'"_en_US, *fwdRef);
      context().SetError(symbol);
      return true;
    }
    if (IsDummy(symbol) && isImplicitNoneType() &&
        symbol.test(Symbol::Flag::Implicit) && !context().HasError(symbol)) {
      // Dummy was implicitly typed despite IMPLICIT NONE(TYPE) in
      // ApplyImplicitRules() due to use in a specification expression,
      // and no explicit type declaration appeared later.
      Say(symbol.name(),
          "No explicit type declared for dummy argument '%s'"_err_en_US);
      context().SetError(symbol);
      return true;
    }
  }
  return false;
}

void ScopeHandler::MakeExternal(Symbol &symbol) {
  if (!symbol.attrs().test(Attr::EXTERNAL)) {
    symbol.attrs().set(Attr::EXTERNAL);
    if (symbol.attrs().test(Attr::INTRINSIC)) { // C840
      Say(symbol.name(),
          "Symbol '%s' cannot have both EXTERNAL and INTRINSIC attributes"_err_en_US,
          symbol.name());
    }
  }
}

// ModuleVisitor implementation

bool ModuleVisitor::Pre(const parser::Only &x) {
  common::visit(common::visitors{
                    [&](const Indirection<parser::GenericSpec> &generic) {
                      GenericSpecInfo genericSpecInfo{generic.value()};
                      AddUseOnly(genericSpecInfo.symbolName());
                      AddUse(genericSpecInfo);
                    },
                    [&](const parser::Name &name) {
                      AddUseOnly(name.source);
                      Resolve(name, AddUse(name.source, name.source).use);
                    },
                    [&](const parser::Rename &rename) { Walk(rename); },
                },
      x.u);
  return false;
}

bool ModuleVisitor::Pre(const parser::Rename::Names &x) {
  const auto &localName{std::get<0>(x.t)};
  const auto &useName{std::get<1>(x.t)};
  AddUseRename(useName.source);
  SymbolRename rename{AddUse(localName.source, useName.source)};
  if (rename.use) {
    EraseRenamedSymbol(*rename.use);
  }
  Resolve(useName, rename.use);
  Resolve(localName, rename.local);
  return false;
}
bool ModuleVisitor::Pre(const parser::Rename::Operators &x) {
  const parser::DefinedOpName &local{std::get<0>(x.t)};
  const parser::DefinedOpName &use{std::get<1>(x.t)};
  GenericSpecInfo localInfo{local};
  GenericSpecInfo useInfo{use};
  if (IsIntrinsicOperator(context(), local.v.source)) {
    Say(local.v,
        "Intrinsic operator '%s' may not be used as a defined operator"_err_en_US);
  } else if (IsLogicalConstant(context(), local.v.source)) {
    Say(local.v,
        "Logical constant '%s' may not be used as a defined operator"_err_en_US);
  } else {
    SymbolRename rename{AddUse(localInfo.symbolName(), useInfo.symbolName())};
    if (rename.use) {
      EraseRenamedSymbol(*rename.use);
    }
    useInfo.Resolve(rename.use);
    localInfo.Resolve(rename.local);
  }
  return false;
}

// Set useModuleScope_ to the Scope of the module being used.
bool ModuleVisitor::Pre(const parser::UseStmt &x) {
  std::optional<bool> isIntrinsic;
  if (x.nature) {
    isIntrinsic = *x.nature == parser::UseStmt::ModuleNature::Intrinsic;
    AddAndCheckExplicitIntrinsicUse(x.moduleName.source, *isIntrinsic);
  } else if (currScope().IsModule() && currScope().symbol() &&
      currScope().symbol()->attrs().test(Attr::INTRINSIC)) {
    // Intrinsic modules USE only other intrinsic modules
    isIntrinsic = true;
  }
  useModuleScope_ = FindModule(x.moduleName, isIntrinsic);
  if (!useModuleScope_) {
    return false;
  }
  // use the name from this source file
  useModuleScope_->symbol()->ReplaceName(x.moduleName.source);
  return true;
}

void ModuleVisitor::Post(const parser::UseStmt &x) {
  if (const auto *list{std::get_if<std::list<parser::Rename>>(&x.u)}) {
    // Not a use-only: collect the names that were used in renames,
    // then add a use for each public name that was not renamed.
    std::set<SourceName> useNames;
    for (const auto &rename : *list) {
      common::visit(common::visitors{
                        [&](const parser::Rename::Names &names) {
                          useNames.insert(std::get<1>(names.t).source);
                        },
                        [&](const parser::Rename::Operators &ops) {
                          useNames.insert(std::get<1>(ops.t).v.source);
                        },
                    },
          rename.u);
    }
    for (const auto &[name, symbol] : *useModuleScope_) {
      if (symbol->attrs().test(Attr::PUBLIC) && !IsUseRenamed(symbol->name()) &&
          (!symbol->attrs().test(Attr::INTRINSIC) ||
              symbol->has<UseDetails>()) &&
          !symbol->has<MiscDetails>() && useNames.count(name) == 0) {
        SourceName location{x.moduleName.source};
        if (auto *localSymbol{FindInScope(name)}) {
          DoAddUse(location, localSymbol->name(), *localSymbol, *symbol);
        } else {
          DoAddUse(location, location, CopySymbol(name, *symbol), *symbol);
        }
      }
    }
  }
  useModuleScope_ = nullptr;
}

ModuleVisitor::SymbolRename ModuleVisitor::AddUse(
    const SourceName &localName, const SourceName &useName) {
  return AddUse(localName, useName, FindInScope(*useModuleScope_, useName));
}

ModuleVisitor::SymbolRename ModuleVisitor::AddUse(
    const SourceName &localName, const SourceName &useName, Symbol *useSymbol) {
  if (!useModuleScope_) {
    return {}; // error occurred finding module
  }
  if (!useSymbol) {
    Say(useName, "'%s' not found in module '%s'"_err_en_US, MakeOpName(useName),
        useModuleScope_->GetName().value());
    return {};
  }
  if (useSymbol->attrs().test(Attr::PRIVATE) &&
      !FindModuleFileContaining(currScope())) {
    // Privacy is not enforced in module files so that generic interfaces
    // can be resolved to private specific procedures in specification
    // expressions.
    Say(useName, "'%s' is PRIVATE in '%s'"_err_en_US, MakeOpName(useName),
        useModuleScope_->GetName().value());
    return {};
  }
  auto &localSymbol{MakeSymbol(localName)};
  DoAddUse(useName, localName, localSymbol, *useSymbol);
  return {&localSymbol, useSymbol};
}

// symbol must be either a Use or a Generic formed by merging two uses.
// Convert it to a UseError with this additional location.
static bool ConvertToUseError(
    Symbol &symbol, const SourceName &location, const Scope &module) {
  const auto *useDetails{symbol.detailsIf<UseDetails>()};
  if (!useDetails) {
    if (auto *genericDetails{symbol.detailsIf<GenericDetails>()}) {
      if (!genericDetails->uses().empty()) {
        useDetails = &genericDetails->uses().at(0)->get<UseDetails>();
      }
    }
  }
  if (useDetails) {
    symbol.set_details(
        UseErrorDetails{*useDetails}.add_occurrence(location, module));
    return true;
  } else {
    return false;
  }
}

// If a symbol has previously been USE-associated and did not appear in a USE
// ONLY clause, erase it from the current scope.  This is needed when a name
// appears in a USE rename clause.
void ModuleVisitor::EraseRenamedSymbol(const Symbol &useSymbol) {
  const SourceName &name{useSymbol.name()};
  if (const Symbol * symbol{FindInScope(name)}) {
    if (auto *useDetails{symbol->detailsIf<UseDetails>()}) {
      const Symbol &moduleSymbol{useDetails->symbol()};
      if (moduleSymbol.name() == name &&
          moduleSymbol.owner() == useSymbol.owner() && IsUseRenamed(name) &&
          !IsUseOnly(name)) {
        EraseSymbol(*symbol);
      }
    }
  }
}

void ModuleVisitor::DoAddUse(SourceName location, SourceName localName,
    Symbol &localSymbol, const Symbol &useSymbol) {
  if (localName != useSymbol.name()) {
    EraseRenamedSymbol(useSymbol);
  }
  if (auto *details{localSymbol.detailsIf<UseErrorDetails>()}) {
    details->add_occurrence(location, *useModuleScope_);
    return;
  }

  if (localSymbol.has<UnknownDetails>()) {
    localSymbol.set_details(UseDetails{localName, useSymbol});
    localSymbol.attrs() =
        useSymbol.attrs() & ~Attrs{Attr::PUBLIC, Attr::PRIVATE};
    localSymbol.flags() = useSymbol.flags();
    return;
  }

  Symbol &localUltimate{localSymbol.GetUltimate()};
  const Symbol &useUltimate{useSymbol.GetUltimate()};
  if (&localUltimate == &useUltimate) {
    // use-associating the same symbol again -- ok
    return;
  }

  auto checkAmbiguousDerivedType{[this, location, localName](
                                     const Symbol *t1, const Symbol *t2) {
    if (!t1 || !t2) {
      return true;
    } else {
      t1 = &t1->GetUltimate();
      t2 = &t2->GetUltimate();
      if (&t1 != &t2) {
        Say(location,
            "Generic interface '%s' has ambiguous derived types from modules '%s' and '%s'"_err_en_US,
            localName, t1->owner().GetName().value(),
            t2->owner().GetName().value());
        return false;
      }
    }
  }};

  auto *localGeneric{localUltimate.detailsIf<GenericDetails>()};
  const auto *useGeneric{useUltimate.detailsIf<GenericDetails>()};
  auto combine{false};
  if (localGeneric) {
    if (useGeneric) {
      if (!checkAmbiguousDerivedType(
              localGeneric->derivedType(), useGeneric->derivedType())) {
        return;
      }
      combine = true;
    } else if (useUltimate.has<DerivedTypeDetails>()) {
      if (checkAmbiguousDerivedType(
              &useUltimate, localGeneric->derivedType())) {
        combine = true;
      } else {
        return;
      }
    } else if (&useUltimate == &BypassGeneric(localUltimate)) {
      return; // nothing to do; used subprogram is local's specific
    }
  } else if (useGeneric) {
    if (localUltimate.has<DerivedTypeDetails>()) {
      if (checkAmbiguousDerivedType(
              &localUltimate, useGeneric->derivedType())) {
        combine = true;
      } else {
        return;
      }
    } else if (&localUltimate == &BypassGeneric(useUltimate).GetUltimate()) {
      // Local is the specific of the used generic; replace it.
      EraseSymbol(localSymbol);
      Symbol &newSymbol{MakeSymbol(localName,
          useUltimate.attrs() & ~Attrs{Attr::PUBLIC, Attr::PRIVATE},
          UseDetails{localName, useUltimate})};
      newSymbol.flags() = useSymbol.flags();
      return;
    }
  } else {
    auto localClass{ClassifyProcedure(localUltimate)};
    auto useClass{ClassifyProcedure(useUltimate)};
    if (localClass == useClass &&
        (localClass == ProcedureDefinitionClass::Intrinsic ||
            localClass == ProcedureDefinitionClass::External) &&
        localUltimate.name() == useUltimate.name()) {
      auto localChars{evaluate::characteristics::Procedure::Characterize(
          localUltimate, GetFoldingContext())};
      auto useChars{evaluate::characteristics::Procedure::Characterize(
          useUltimate, GetFoldingContext())};
      if (localChars && useChars) {
        if (*localChars == *useChars) {
          // Same intrinsic or external procedure defined identically in two
          // modules
          return;
        }
      }
    }
  }
  if (!combine) {
    if (!ConvertToUseError(localSymbol, location, *useModuleScope_)) {
      Say(location,
          "Cannot use-associate '%s'; it is already declared in this scope"_err_en_US,
          localName)
          .Attach(localSymbol.name(), "Previous declaration of '%s'"_en_US,
              localName);
    }
    return;
  }

  // Two items are being use-associated from different modules
  // to the same local name.  At least one of them must be a generic,
  // and the other one can be a generic or a derived type.
  // (It could also have been the specific of the generic, but those
  // cases are handled above without needing to make a local copy of the
  // generic.)

  if (localGeneric) {
    if (localSymbol.has<UseDetails>()) {
      // Create a local copy of a previously use-associated generic so that
      // it can be locally extended without corrupting the original.
      GenericDetails generic;
      generic.CopyFrom(*localGeneric);
      if (localGeneric->specific()) {
        generic.set_specific(*localGeneric->specific());
      }
      EraseSymbol(localSymbol);
      Symbol &newSymbol{MakeSymbol(
          localSymbol.name(), localSymbol.attrs(), std::move(generic))};
      newSymbol.flags() = localSymbol.flags();
      localGeneric = &newSymbol.get<GenericDetails>();
      localGeneric->AddUse(localSymbol);
    }
    if (useGeneric) {
      // Combine two use-associated generics
      localSymbol.attrs() =
          useSymbol.attrs() & ~Attrs{Attr::PUBLIC, Attr::PRIVATE};
      localSymbol.flags() = useSymbol.flags();
      AddGenericUse(*localGeneric, localName, useUltimate);
      localGeneric->CopyFrom(*useGeneric);
      if (useGeneric->specific()) {
        if (!localGeneric->specific()) {
          localGeneric->set_specific(
              *const_cast<Symbol *>(useGeneric->specific()));
        } else if (&localGeneric->specific()->GetUltimate() !=
            &useGeneric->specific()->GetUltimate()) {
          Say(location,
              "Cannot use-associate generic interface '%s' with specific procedure of the same name when another such generic is in scope"_err_en_US,
              localName)
              .Attach(
                  localSymbol.name(), "Previous USE of '%s'"_en_US, localName);
        }
      }
    } else {
      CHECK(useUltimate.has<DerivedTypeDetails>());
      localGeneric->set_derivedType(
          AddGenericUse(*localGeneric, localName, useUltimate));
    }
  } else {
    CHECK(useGeneric && localUltimate.has<DerivedTypeDetails>());
    CHECK(localSymbol.has<UseDetails>());
    // Create a local copy of the use-associated generic, then extend it
    // with the local derived type.
    GenericDetails generic;
    generic.CopyFrom(*useGeneric);
    if (useGeneric->specific()) {
      generic.set_specific(*const_cast<Symbol *>(useGeneric->specific()));
    }
    EraseSymbol(localSymbol);
    Symbol &newSymbol{MakeSymbol(localName,
        useUltimate.attrs() & ~Attrs{Attr::PUBLIC, Attr::PRIVATE},
        std::move(generic))};
    newSymbol.flags() = useUltimate.flags();
    auto &newUseGeneric{newSymbol.get<GenericDetails>()};
    AddGenericUse(newUseGeneric, localName, useUltimate);
    newUseGeneric.AddUse(localSymbol);
    newUseGeneric.set_derivedType(localSymbol);
  }
}

void ModuleVisitor::AddUse(const GenericSpecInfo &info) {
  if (useModuleScope_) {
    const auto &name{info.symbolName()};
    auto rename{AddUse(name, name, FindInScope(*useModuleScope_, name))};
    info.Resolve(rename.use);
  }
}

// Create a UseDetails symbol for this USE and add it to generic
Symbol &ModuleVisitor::AddGenericUse(
    GenericDetails &generic, const SourceName &name, const Symbol &useSymbol) {
  Symbol &newSymbol{
      currScope().MakeSymbol(name, {}, UseDetails{name, useSymbol})};
  generic.AddUse(newSymbol);
  return newSymbol;
}

// Enforce C1406
void ModuleVisitor::AddAndCheckExplicitIntrinsicUse(
    SourceName name, bool isIntrinsic) {
  if (isIntrinsic) {
    if (auto iter{explicitNonIntrinsicUses_.find(name)};
        iter != explicitNonIntrinsicUses_.end()) {
      Say(name,
          "Cannot USE,INTRINSIC module '%s' in the same scope as USE,NON_INTRINSIC"_err_en_US,
          name)
          .Attach(*iter, "Previous USE of '%s'"_en_US, *iter);
    }
    explicitIntrinsicUses_.insert(name);
  } else {
    if (auto iter{explicitIntrinsicUses_.find(name)};
        iter != explicitIntrinsicUses_.end()) {
      Say(name,
          "Cannot USE,NON_INTRINSIC module '%s' in the same scope as USE,INTRINSIC"_err_en_US,
          name)
          .Attach(*iter, "Previous USE of '%s'"_en_US, *iter);
    }
    explicitNonIntrinsicUses_.insert(name);
  }
}

bool ModuleVisitor::BeginSubmodule(
    const parser::Name &name, const parser::ParentIdentifier &parentId) {
  const auto &ancestorName{std::get<parser::Name>(parentId.t)};
  Scope *parentScope{nullptr};
  Scope *ancestor{FindModule(ancestorName, false /*not intrinsic*/)};
  if (ancestor) {
    if (const auto &parentName{
            std::get<std::optional<parser::Name>>(parentId.t)}) {
      parentScope = FindModule(*parentName, false /*not intrinsic*/, ancestor);
    } else {
      parentScope = ancestor;
    }
  }
  if (parentScope) {
    PushScope(*parentScope);
  } else {
    // Error recovery: there's no ancestor scope, so create a dummy one to
    // hold the submodule's scope.
    SourceName dummyName{context().GetTempName(currScope())};
    Symbol &dummySymbol{MakeSymbol(dummyName, Attrs{}, ModuleDetails{false})};
    PushScope(Scope::Kind::Module, &dummySymbol);
    parentScope = &currScope();
  }
  BeginModule(name, true);
  if (ancestor && !ancestor->AddSubmodule(name.source, currScope())) {
    Say(name, "Module '%s' already has a submodule named '%s'"_err_en_US,
        ancestorName.source, name.source);
  }
  return true;
}

void ModuleVisitor::BeginModule(const parser::Name &name, bool isSubmodule) {
  auto &symbol{MakeSymbol(name, ModuleDetails{isSubmodule})};
  auto &details{symbol.get<ModuleDetails>()};
  PushScope(Scope::Kind::Module, &symbol);
  details.set_scope(&currScope());
  defaultAccess_ = Attr::PUBLIC;
  prevAccessStmt_ = std::nullopt;
}

// Find a module or submodule by name and return its scope.
// If ancestor is present, look for a submodule of that ancestor module.
// May have to read a .mod file to find it.
// If an error occurs, report it and return nullptr.
Scope *ModuleVisitor::FindModule(const parser::Name &name,
    std::optional<bool> isIntrinsic, Scope *ancestor) {
  ModFileReader reader{context()};
  Scope *scope{reader.Read(name.source, isIntrinsic, ancestor)};
  if (!scope) {
    return nullptr;
  }
  if (DoesScopeContain(scope, currScope())) { // 14.2.2(1)
    Say(name, "Module '%s' cannot USE itself"_err_en_US);
  }
  Resolve(name, scope->symbol());
  return scope;
}

void ModuleVisitor::ApplyDefaultAccess() {
  for (auto &pair : currScope()) {
    Symbol &symbol = *pair.second;
    if (!symbol.attrs().HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
      symbol.attrs().set(defaultAccess_);
    }
  }
}

// InterfaceVistor implementation

bool InterfaceVisitor::Pre(const parser::InterfaceStmt &x) {
  bool isAbstract{std::holds_alternative<parser::Abstract>(x.u)};
  genericInfo_.emplace(/*isInterface*/ true, isAbstract);
  return BeginAttrs();
}

void InterfaceVisitor::Post(const parser::InterfaceStmt &) { EndAttrs(); }

void InterfaceVisitor::Post(const parser::EndInterfaceStmt &) {
  genericInfo_.pop();
}

// Create a symbol in genericSymbol_ for this GenericSpec.
bool InterfaceVisitor::Pre(const parser::GenericSpec &x) {
  if (auto *symbol{FindInScope(GenericSpecInfo{x}.symbolName())}) {
    SetGenericSymbol(*symbol);
  }
  return false;
}

bool InterfaceVisitor::Pre(const parser::ProcedureStmt &x) {
  if (!isGeneric()) {
    Say("A PROCEDURE statement is only allowed in a generic interface block"_err_en_US);
    return false;
  }
  auto kind{std::get<parser::ProcedureStmt::Kind>(x.t)};
  const auto &names{std::get<std::list<parser::Name>>(x.t)};
  AddSpecificProcs(names, kind);
  return false;
}

bool InterfaceVisitor::Pre(const parser::GenericStmt &) {
  genericInfo_.emplace(/*isInterface*/ false);
  return true;
}
void InterfaceVisitor::Post(const parser::GenericStmt &x) {
  if (auto &accessSpec{std::get<std::optional<parser::AccessSpec>>(x.t)}) {
    GetGenericInfo().symbol->attrs().set(AccessSpecToAttr(*accessSpec));
  }
  const auto &names{std::get<std::list<parser::Name>>(x.t)};
  AddSpecificProcs(names, ProcedureKind::Procedure);
  genericInfo_.pop();
}

bool InterfaceVisitor::inInterfaceBlock() const {
  return !genericInfo_.empty() && GetGenericInfo().isInterface;
}
bool InterfaceVisitor::isGeneric() const {
  return !genericInfo_.empty() && GetGenericInfo().symbol;
}
bool InterfaceVisitor::isAbstract() const {
  return !genericInfo_.empty() && GetGenericInfo().isAbstract;
}

void InterfaceVisitor::AddSpecificProcs(
    const std::list<parser::Name> &names, ProcedureKind kind) {
  for (const auto &name : names) {
    specificProcs_.emplace(
        GetGenericInfo().symbol, std::make_pair(&name, kind));
  }
}

// By now we should have seen all specific procedures referenced by name in
// this generic interface. Resolve those names to symbols.
void InterfaceVisitor::ResolveSpecificsInGeneric(Symbol &generic) {
  auto &details{generic.get<GenericDetails>()};
  UnorderedSymbolSet symbolsSeen;
  for (const Symbol &symbol : details.specificProcs()) {
    symbolsSeen.insert(symbol.GetUltimate());
  }
  auto range{specificProcs_.equal_range(&generic)};
  for (auto it{range.first}; it != range.second; ++it) {
    const parser::Name *name{it->second.first};
    auto kind{it->second.second};
    const auto *symbol{FindSymbol(*name)};
    if (!symbol) {
      Say(*name, "Procedure '%s' not found"_err_en_US);
      continue;
    }
    // Subtlety: when *symbol is a use- or host-association, the specific
    // procedure that is recorded in the GenericDetails below must be *symbol,
    // not the specific procedure shadowed by a generic, because that specific
    // procedure may be a symbol from another module and its name unavailable to
    // emit to a module file.
    const Symbol &bypassed{BypassGeneric(*symbol)};
    const Symbol &specific{
        symbol == &symbol->GetUltimate() ? bypassed : *symbol};
    const Symbol &ultimate{bypassed.GetUltimate()};
    ProcedureDefinitionClass defClass{ClassifyProcedure(ultimate)};
    if (defClass == ProcedureDefinitionClass::Module) {
      // ok
    } else if (kind == ProcedureKind::ModuleProcedure) {
      Say(*name, "'%s' is not a module procedure"_err_en_US);
      continue;
    } else {
      switch (defClass) {
      case ProcedureDefinitionClass::Intrinsic:
      case ProcedureDefinitionClass::External:
      case ProcedureDefinitionClass::Internal:
        break;
      case ProcedureDefinitionClass::None:
        Say(*name, "'%s' is not a procedure"_err_en_US);
        continue;
      default:
        Say(*name,
            "'%s' is not a procedure that can appear in a generic interface"_err_en_US);
        continue;
      }
    }
    if (symbolsSeen.insert(ultimate).second /*true if added*/) {
      // When a specific procedure is a USE association, that association
      // is saved in the generic's specifics, not its ultimate symbol,
      // so that module file output of interfaces can distinguish them.
      details.AddSpecificProc(specific, name->source);
    } else if (&specific == &ultimate) {
      Say(name->source,
          "Procedure '%s' is already specified in generic '%s'"_err_en_US,
          name->source, MakeOpName(generic.name()));
    } else {
      Say(name->source,
          "Procedure '%s' from module '%s' is already specified in generic '%s'"_err_en_US,
          ultimate.name(), ultimate.owner().GetName().value(),
          MakeOpName(generic.name()));
    }
  }
  specificProcs_.erase(range.first, range.second);
}

// Check that the specific procedures are all functions or all subroutines.
// If there is a derived type with the same name they must be functions.
// Set the corresponding flag on generic.
void InterfaceVisitor::CheckGenericProcedures(Symbol &generic) {
  ResolveSpecificsInGeneric(generic);
  auto &details{generic.get<GenericDetails>()};
  if (auto *proc{details.CheckSpecific()}) {
    auto msg{
        "'%s' may not be the name of both a generic interface and a"
        " procedure unless it is a specific procedure of the generic"_err_en_US};
    if (proc->name().begin() > generic.name().begin()) {
      Say(proc->name(), std::move(msg));
    } else {
      Say(generic.name(), std::move(msg));
    }
  }
  auto &specifics{details.specificProcs()};
  if (specifics.empty()) {
    if (details.derivedType()) {
      generic.set(Symbol::Flag::Function);
    }
    return;
  }
  const Symbol &firstSpecific{specifics.front()};
  bool isFunction{firstSpecific.test(Symbol::Flag::Function)};
  for (const Symbol &specific : specifics) {
    if (isFunction != specific.test(Symbol::Flag::Function)) { // C1514
      auto &msg{Say(generic.name(),
          "Generic interface '%s' has both a function and a subroutine"_err_en_US)};
      if (isFunction) {
        msg.Attach(firstSpecific.name(), "Function declaration"_en_US);
        msg.Attach(specific.name(), "Subroutine declaration"_en_US);
      } else {
        msg.Attach(firstSpecific.name(), "Subroutine declaration"_en_US);
        msg.Attach(specific.name(), "Function declaration"_en_US);
      }
    }
  }
  if (!isFunction && details.derivedType()) {
    SayDerivedType(generic.name(),
        "Generic interface '%s' may only contain functions due to derived type"
        " with same name"_err_en_US,
        *details.derivedType()->scope());
  }
  generic.set(isFunction ? Symbol::Flag::Function : Symbol::Flag::Subroutine);
}

// SubprogramVisitor implementation

// Return false if it is actually an assignment statement.
bool SubprogramVisitor::HandleStmtFunction(const parser::StmtFunctionStmt &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  const DeclTypeSpec *resultType{nullptr};
  // Look up name: provides return type or tells us if it's an array
  if (auto *symbol{FindSymbol(name)}) {
    auto *details{symbol->detailsIf<EntityDetails>()};
    if (!details) {
      badStmtFuncFound_ = true;
      return false;
    }
    // TODO: check that attrs are compatible with stmt func
    resultType = details->type();
    symbol->details() = UnknownDetails{}; // will be replaced below
  }
  if (badStmtFuncFound_) {
    Say(name, "'%s' has not been declared as an array"_err_en_US);
    return true;
  }
  auto &symbol{PushSubprogramScope(name, Symbol::Flag::Function)};
  symbol.set(Symbol::Flag::StmtFunction);
  EraseSymbol(symbol); // removes symbol added by PushSubprogramScope
  auto &details{symbol.get<SubprogramDetails>()};
  for (const auto &dummyName : std::get<std::list<parser::Name>>(x.t)) {
    ObjectEntityDetails dummyDetails{true};
    if (auto *dummySymbol{FindInScope(currScope().parent(), dummyName)}) {
      if (auto *d{dummySymbol->detailsIf<EntityDetails>()}) {
        if (d->type()) {
          dummyDetails.set_type(*d->type());
        }
      }
    }
    Symbol &dummy{MakeSymbol(dummyName, std::move(dummyDetails))};
    ApplyImplicitRules(dummy);
    details.add_dummyArg(dummy);
  }
  ObjectEntityDetails resultDetails;
  if (resultType) {
    resultDetails.set_type(*resultType);
  }
  resultDetails.set_funcResult(true);
  Symbol &result{MakeSymbol(name, std::move(resultDetails))};
  ApplyImplicitRules(result);
  details.set_result(result);
  const auto &parsedExpr{std::get<parser::Scalar<parser::Expr>>(x.t)};
  Walk(parsedExpr);
  // The analysis of the expression that constitutes the body of the
  // statement function is deferred to FinishSpecificationPart() so that
  // all declarations and implicit typing are complete.
  PopScope();
  return true;
}

bool SubprogramVisitor::Pre(const parser::Suffix &suffix) {
  if (suffix.resultName) {
    if (IsFunction(currScope())) {
      if (FuncResultStack::FuncInfo * info{funcResultStack().Top()}) {
        if (info->inFunctionStmt) {
          info->resultName = &suffix.resultName.value();
        } else {
          // will check the result name in Post(EntryStmt)
        }
      }
    } else {
      Message &msg{Say(*suffix.resultName,
          "RESULT(%s) may appear only in a function"_err_en_US)};
      if (const Symbol * subprogram{InclusiveScope().symbol()}) {
        msg.Attach(subprogram->name(), "Containing subprogram"_en_US);
      }
    }
  }
  // LanguageBindingSpec deferred to Post(EntryStmt) or, for FunctionStmt,
  // all the way to EndSubprogram().
  return false;
}

bool SubprogramVisitor::Pre(const parser::PrefixSpec &x) {
  // Save this to process after UseStmt and ImplicitPart
  if (const auto *parsedType{std::get_if<parser::DeclarationTypeSpec>(&x.u)}) {
    FuncResultStack::FuncInfo &info{DEREF(funcResultStack().Top())};
    if (info.parsedType) { // C1543
      Say(currStmtSource().value(),
          "FUNCTION prefix cannot specify the type more than once"_err_en_US);
      return false;
    } else {
      info.parsedType = parsedType;
      info.source = currStmtSource();
      return false;
    }
  } else {
    return true;
  }
}

bool SubprogramVisitor::Pre(const parser::InterfaceBody::Subroutine &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement.t)};
  return BeginSubprogram(name, Symbol::Flag::Subroutine);
}
void SubprogramVisitor::Post(const parser::InterfaceBody::Subroutine &x) {
  const auto &stmt{std::get<parser::Statement<parser::SubroutineStmt>>(x.t)};
  EndSubprogram(stmt.source,
      &std::get<std::optional<parser::LanguageBindingSpec>>(stmt.statement.t));
}
bool SubprogramVisitor::Pre(const parser::InterfaceBody::Function &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement.t)};
  return BeginSubprogram(name, Symbol::Flag::Function);
}
void SubprogramVisitor::Post(const parser::InterfaceBody::Function &x) {
  const auto &stmt{std::get<parser::Statement<parser::FunctionStmt>>(x.t)};
  const auto &maybeSuffix{
      std::get<std::optional<parser::Suffix>>(stmt.statement.t)};
  EndSubprogram(stmt.source, maybeSuffix ? &maybeSuffix->binding : nullptr);
}

bool SubprogramVisitor::Pre(const parser::SubroutineStmt &stmt) {
  BeginAttrs();
  Walk(std::get<std::list<parser::PrefixSpec>>(stmt.t));
  Walk(std::get<parser::Name>(stmt.t));
  Walk(std::get<std::list<parser::DummyArg>>(stmt.t));
  // Don't traverse the LanguageBindingSpec now; it's deferred to EndSubprogram.
  const auto &name{std::get<parser::Name>(stmt.t)};
  auto &details{PostSubprogramStmt(name)};
  for (const auto &dummyArg : std::get<std::list<parser::DummyArg>>(stmt.t)) {
    if (const auto *dummyName{std::get_if<parser::Name>(&dummyArg.u)}) {
      Symbol &dummy{MakeSymbol(*dummyName, EntityDetails{true})};
      details.add_dummyArg(dummy);
    } else {
      details.add_alternateReturn();
    }
  }
  return false;
}
bool SubprogramVisitor::Pre(const parser::FunctionStmt &) {
  FuncResultStack::FuncInfo &info{DEREF(funcResultStack().Top())};
  CHECK(!info.inFunctionStmt);
  info.inFunctionStmt = true;
  return BeginAttrs();
}
bool SubprogramVisitor::Pre(const parser::EntryStmt &) { return BeginAttrs(); }

void SubprogramVisitor::Post(const parser::FunctionStmt &stmt) {
  const auto &name{std::get<parser::Name>(stmt.t)};
  auto &details{PostSubprogramStmt(name)};
  for (const auto &dummyName : std::get<std::list<parser::Name>>(stmt.t)) {
    Symbol &dummy{MakeSymbol(dummyName, EntityDetails{true})};
    details.add_dummyArg(dummy);
  }
  const parser::Name *funcResultName;
  FuncResultStack::FuncInfo &info{DEREF(funcResultStack().Top())};
  CHECK(info.inFunctionStmt);
  info.inFunctionStmt = false;
  bool distinctResultName{
      info.resultName && info.resultName->source != name.source};
  if (distinctResultName) {
    // Note that RESULT is ignored if it has the same name as the function.
    // The symbol created by PushScope() is retained as a place-holder
    // for error detection.
    funcResultName = info.resultName;
  } else {
    EraseSymbol(name); // was added by PushScope()
    funcResultName = &name;
  }
  if (details.isFunction()) {
    CHECK(context().HasError(currScope().symbol()));
  } else {
    // RESULT(x) can be the same explicitly-named RESULT(x) as an ENTRY
    // statement.
    Symbol *result{nullptr};
    if (distinctResultName) {
      if (auto iter{currScope().find(funcResultName->source)};
          iter != currScope().end()) {
        Symbol &entryResult{*iter->second};
        if (IsFunctionResult(entryResult)) {
          result = &entryResult;
        }
      }
    }
    if (result) {
      Resolve(*funcResultName, *result);
    } else {
      // add function result to function scope
      EntityDetails funcResultDetails;
      funcResultDetails.set_funcResult(true);
      result = &MakeSymbol(*funcResultName, std::move(funcResultDetails));
    }
    info.resultSymbol = result;
    details.set_result(*result);
  }
  // C1560.
  if (info.resultName && !distinctResultName) {
    Say(info.resultName->source,
        "The function name should not appear in RESULT, references to '%s' "
        "inside the function will be considered as references to the "
        "result only"_warn_en_US,
        name.source);
    // RESULT name was ignored above, the only side effect from doing so will be
    // the inability to make recursive calls. The related parser::Name is still
    // resolved to the created function result symbol because every parser::Name
    // should be resolved to avoid internal errors.
    Resolve(*info.resultName, info.resultSymbol);
  }
  name.symbol = currScope().symbol(); // must not be function result symbol
  // Clear the RESULT() name now in case an ENTRY statement in the implicit-part
  // has a RESULT() suffix.
  info.resultName = nullptr;
}

SubprogramDetails &SubprogramVisitor::PostSubprogramStmt(
    const parser::Name &name) {
  Symbol &symbol{*currScope().symbol()};
  symbol.attrs() |= EndAttrs();
  if (symbol.attrs().test(Attr::MODULE)) {
    symbol.attrs().set(Attr::EXTERNAL, false);
  }
  return symbol.get<SubprogramDetails>();
}

void SubprogramVisitor::Post(const parser::EntryStmt &stmt) {
  if (const auto &suffix{std::get<std::optional<parser::Suffix>>(stmt.t)}) {
    Walk(suffix->binding);
  }
  PostEntryStmt(stmt);
  EndAttrs();
}

void SubprogramVisitor::CreateEntry(
    const parser::EntryStmt &stmt, Symbol &subprogram) {
  const auto &entryName{std::get<parser::Name>(stmt.t)};
  Scope &outer{currScope().parent()};
  Symbol::Flag subpFlag{subprogram.test(Symbol::Flag::Function)
          ? Symbol::Flag::Function
          : Symbol::Flag::Subroutine};
  Attrs attrs;
  const auto &suffix{std::get<std::optional<parser::Suffix>>(stmt.t)};
  bool hasGlobalBindingName{outer.IsGlobal() && suffix && suffix->binding &&
      suffix->binding->v.has_value()};
  if (!hasGlobalBindingName) {
    if (Symbol * extant{FindSymbol(outer, entryName)}) {
      if (!HandlePreviousCalls(entryName, *extant, subpFlag)) {
        if (outer.IsTopLevel()) {
          Say2(entryName,
              "'%s' is already defined as a global identifier"_err_en_US,
              *extant, "Previous definition of '%s'"_en_US);
        } else {
          SayAlreadyDeclared(entryName, *extant);
        }
        return;
      }
      attrs = extant->attrs();
    }
  }
  bool badResultName{false};
  std::optional<SourceName> distinctResultName;
  if (suffix && suffix->resultName &&
      suffix->resultName->source != entryName.source) {
    distinctResultName = suffix->resultName->source;
    const parser::Name &resultName{*suffix->resultName};
    if (resultName.source == subprogram.name()) { // C1574
      Say2(resultName.source,
          "RESULT(%s) may not have the same name as the function"_err_en_US,
          subprogram, "Containing function"_en_US);
      badResultName = true;
    } else if (const Symbol * extant{FindSymbol(outer, resultName)}) { // C1574
      if (const auto *details{extant->detailsIf<SubprogramDetails>()}) {
        if (details->entryScope() == &currScope()) {
          Say2(resultName.source,
              "RESULT(%s) may not have the same name as an ENTRY in the function"_err_en_US,
              extant->name(), "Conflicting ENTRY"_en_US);
          badResultName = true;
        }
      }
    }
  }
  if (outer.IsModule() && !attrs.test(Attr::PRIVATE)) {
    attrs.set(Attr::PUBLIC);
  }
  Symbol *entrySymbol{nullptr};
  if (hasGlobalBindingName) {
    // Hide the entry's symbol in a new anonymous global scope so
    // that its name doesn't clash with anything.
    Symbol &symbol{MakeSymbol(outer, context().GetTempName(outer), Attrs{})};
    symbol.set_details(MiscDetails{MiscDetails::Kind::ScopeName});
    Scope &hidden{outer.MakeScope(Scope::Kind::Global, &symbol)};
    entrySymbol = &MakeSymbol(hidden, entryName.source, attrs);
  } else {
    entrySymbol = FindInScope(outer, entryName.source);
    if (entrySymbol) {
      if (auto *generic{entrySymbol->detailsIf<GenericDetails>()}) {
        if (auto *specific{generic->specific()}) {
          // Forward reference to ENTRY from a generic interface
          entrySymbol = specific;
          entrySymbol->attrs() |= attrs;
        }
      }
    } else {
      entrySymbol = &MakeSymbol(outer, entryName.source, attrs);
    }
  }
  SubprogramDetails entryDetails;
  entryDetails.set_entryScope(currScope());
  entrySymbol->set(subpFlag);
  if (subpFlag == Symbol::Flag::Function) {
    Symbol *result{nullptr};
    EntityDetails resultDetails;
    resultDetails.set_funcResult(true);
    if (distinctResultName) {
      if (!badResultName) {
        // RESULT(x) can be the same explicitly-named RESULT(x) as
        // the enclosing function or another ENTRY.
        if (auto iter{currScope().find(suffix->resultName->source)};
            iter != currScope().end()) {
          result = &*iter->second;
        }
        if (!result) {
          result = &MakeSymbol(
              *distinctResultName, Attrs{}, std::move(resultDetails));
        }
        Resolve(*suffix->resultName, *result);
      }
    } else {
      result = &MakeSymbol(entryName.source, Attrs{}, std::move(resultDetails));
    }
    if (result) {
      entryDetails.set_result(*result);
    }
  }
  if (subpFlag == Symbol::Flag::Subroutine ||
      (distinctResultName && !badResultName)) {
    Symbol &assoc{MakeSymbol(entryName.source)};
    assoc.set_details(HostAssocDetails{*entrySymbol});
    assoc.set(Symbol::Flag::Subroutine);
  }
  Resolve(entryName, *entrySymbol);
  Details details{std::move(entryDetails)};
  entrySymbol->set_details(std::move(entryDetails));
}

void SubprogramVisitor::PostEntryStmt(const parser::EntryStmt &stmt) {
  // The entry symbol should have already been created and resolved
  // in CreateEntry(), called by BeginSubprogram(), with one exception (below).
  const auto &name{std::get<parser::Name>(stmt.t)};
  Scope &inclusiveScope{InclusiveScope()};
  if (!name.symbol) {
    if (inclusiveScope.kind() != Scope::Kind::Subprogram) {
      Say(name.source,
          "ENTRY '%s' may appear only in a subroutine or function"_err_en_US,
          name.source);
    } else if (FindSeparateModuleSubprogramInterface(inclusiveScope.symbol())) {
      Say(name.source,
          "ENTRY '%s' may not appear in a separate module procedure"_err_en_US,
          name.source);
    } else {
      // C1571 - entry is nested, so was not put into the program tree; error
      // is emitted from MiscChecker in semantics.cpp.
    }
    return;
  }
  Symbol &entrySymbol{*name.symbol};
  if (context().HasError(entrySymbol)) {
    return;
  }
  if (!entrySymbol.has<SubprogramDetails>()) {
    SayAlreadyDeclared(name, entrySymbol);
    return;
  }
  SubprogramDetails &entryDetails{entrySymbol.get<SubprogramDetails>()};
  CHECK(entryDetails.entryScope() == &inclusiveScope);
  entrySymbol.attrs() |= GetAttrs();
  SetBindNameOn(entrySymbol);
  for (const auto &dummyArg : std::get<std::list<parser::DummyArg>>(stmt.t)) {
    if (const auto *dummyName{std::get_if<parser::Name>(&dummyArg.u)}) {
      Symbol *dummy{FindSymbol(*dummyName)};
      if (dummy) {
        common::visit(
            common::visitors{[](EntityDetails &x) { x.set_isDummy(); },
                [](ObjectEntityDetails &x) { x.set_isDummy(); },
                [](ProcEntityDetails &x) { x.set_isDummy(); },
                [](SubprogramDetails &x) { x.set_isDummy(); },
                [&](const auto &) {
                  Say2(dummyName->source,
                      "ENTRY dummy argument '%s' is previously declared as an item that may not be used as a dummy argument"_err_en_US,
                      dummy->name(), "Previous declaration of '%s'"_en_US);
                }},
            dummy->details());
      } else {
        dummy = &MakeSymbol(*dummyName, EntityDetails{true});
        if (!inSpecificationPart_) {
          ApplyImplicitRules(*dummy);
        }
      }
      entryDetails.add_dummyArg(*dummy);
    } else {
      if (entrySymbol.test(Symbol::Flag::Function)) { // C1573
        Say(name,
            "ENTRY in a function may not have an alternate return dummy argument"_err_en_US);
        break;
      }
      entryDetails.add_alternateReturn();
    }
  }
}

Symbol *ScopeHandler::FindSeparateModuleProcedureInterface(
    const parser::Name &name) {
  auto *symbol{FindSymbol(name)};
  if (symbol && symbol->has<SubprogramNameDetails>()) {
    symbol = FindSymbol(currScope().parent(), name);
  }
  if (symbol) {
    if (auto *generic{symbol->detailsIf<GenericDetails>()}) {
      symbol = generic->specific();
    }
  }
  if (!IsSeparateModuleProcedureInterface(symbol)) {
    Say(name, "'%s' was not declared a separate module procedure"_err_en_US);
    symbol = nullptr;
  }
  return symbol;
}

// A subprogram declared with MODULE PROCEDURE
bool SubprogramVisitor::BeginMpSubprogram(const parser::Name &name) {
  Symbol *symbol{FindSeparateModuleProcedureInterface(name)};
  if (!symbol) {
    return false;
  }
  if (symbol->owner() == currScope() && symbol->scope()) {
    // This is a MODULE PROCEDURE whose interface appears in its host.
    // Convert the module procedure's interface into a subprogram.
    SetScope(DEREF(symbol->scope()));
    symbol->get<SubprogramDetails>().set_isInterface(false);
  } else {
    // Copy the interface into a new subprogram scope.
    Symbol &newSymbol{MakeSymbol(name, SubprogramDetails{})};
    PushScope(Scope::Kind::Subprogram, &newSymbol);
    const auto &details{symbol->get<SubprogramDetails>()};
    auto &newDetails{newSymbol.get<SubprogramDetails>()};
    newDetails.set_moduleInterface(*symbol);
    for (const Symbol *dummyArg : details.dummyArgs()) {
      if (!dummyArg) {
        newDetails.add_alternateReturn();
      } else if (Symbol * copy{currScope().CopySymbol(*dummyArg)}) {
        newDetails.add_dummyArg(*copy);
      }
    }
    if (details.isFunction()) {
      currScope().erase(symbol->name());
      newDetails.set_result(*currScope().CopySymbol(details.result()));
    }
    newSymbol.attrs() |= symbol->attrs();
  }
  return true;
}

// A subprogram or interface declared with SUBROUTINE or FUNCTION
bool SubprogramVisitor::BeginSubprogram(const parser::Name &name,
    Symbol::Flag subpFlag, bool hasModulePrefix,
    const parser::LanguageBindingSpec *bindingSpec,
    const ProgramTree::EntryStmtList *entryStmts) {
  if (hasModulePrefix && currScope().IsGlobal()) { // C1547
    Say(name,
        "'%s' is a MODULE procedure which must be declared within a "
        "MODULE or SUBMODULE"_err_en_US);
    return false;
  }
  Symbol *moduleInterface{nullptr};
  if (hasModulePrefix && !inInterfaceBlock()) {
    moduleInterface = FindSeparateModuleProcedureInterface(name);
    if (moduleInterface && &moduleInterface->owner() == &currScope()) {
      // Subprogram is MODULE FUNCTION or MODULE SUBROUTINE with an interface
      // previously defined in the same scope.
      currScope().erase(moduleInterface->name());
    }
  }
  Symbol &newSymbol{PushSubprogramScope(name, subpFlag, bindingSpec)};
  if (moduleInterface) {
    newSymbol.get<SubprogramDetails>().set_moduleInterface(*moduleInterface);
    if (moduleInterface->attrs().test(Attr::PRIVATE)) {
      newSymbol.attrs().set(Attr::PRIVATE);
    } else if (moduleInterface->attrs().test(Attr::PUBLIC)) {
      newSymbol.attrs().set(Attr::PUBLIC);
    }
  }
  if (entryStmts) {
    for (const auto &ref : *entryStmts) {
      CreateEntry(*ref, newSymbol);
    }
  }
  return true;
}

void SubprogramVisitor::HandleLanguageBinding(Symbol *symbol,
    std::optional<parser::CharBlock> stmtSource,
    const std::optional<parser::LanguageBindingSpec> *binding) {
  if (binding && *binding && symbol) {
    // Finally process the BIND(C,NAME=name) now that symbols in the name
    // expression will resolve to local names if needed.
    auto flagRestorer{common::ScopedSet(inSpecificationPart_, false)};
    auto originalStmtSource{messageHandler().currStmtSource()};
    messageHandler().set_currStmtSource(stmtSource);
    BeginAttrs();
    Walk(**binding);
    SetBindNameOn(*symbol);
    symbol->attrs() |= EndAttrs();
    messageHandler().set_currStmtSource(originalStmtSource);
  }
}

void SubprogramVisitor::EndSubprogram(
    std::optional<parser::CharBlock> stmtSource,
    const std::optional<parser::LanguageBindingSpec> *binding,
    const ProgramTree::EntryStmtList *entryStmts) {
  HandleLanguageBinding(currScope().symbol(), stmtSource, binding);
  if (entryStmts) {
    for (const auto &ref : *entryStmts) {
      const parser::EntryStmt &entryStmt{*ref};
      if (const auto &suffix{
              std::get<std::optional<parser::Suffix>>(entryStmt.t)}) {
        const auto &name{std::get<parser::Name>(entryStmt.t)};
        HandleLanguageBinding(name.symbol, name.source, &suffix->binding);
      }
    }
  }
  PopScope();
}

bool SubprogramVisitor::HandlePreviousCalls(
    const parser::Name &name, Symbol &symbol, Symbol::Flag subpFlag) {
  // If the extant symbol is a generic, check its homonymous specific
  // procedure instead if it has one.
  if (auto *generic{symbol.detailsIf<GenericDetails>()}) {
    return generic->specific() &&
        HandlePreviousCalls(name, *generic->specific(), subpFlag);
  } else if (const auto *proc{symbol.detailsIf<ProcEntityDetails>()}; proc &&
             !proc->isDummy() &&
             !symbol.attrs().HasAny(Attrs{Attr::INTRINSIC, Attr::POINTER})) {
    // There's a symbol created for previous calls to this subprogram or
    // ENTRY's name.  We have to replace that symbol in situ to avoid the
    // obligation to rewrite symbol pointers in the parse tree.
    if (!symbol.test(subpFlag)) {
      Say2(name,
          subpFlag == Symbol::Flag::Function
              ? "'%s' was previously called as a subroutine"_err_en_US
              : "'%s' was previously called as a function"_err_en_US,
          symbol, "Previous call of '%s'"_en_US);
    }
    EntityDetails entity;
    if (proc->type()) {
      entity.set_type(*proc->type());
    }
    symbol.details() = std::move(entity);
    return true;
  } else {
    return symbol.has<UnknownDetails>() || symbol.has<SubprogramNameDetails>();
  }
}

void SubprogramVisitor::CheckExtantProc(
    const parser::Name &name, Symbol::Flag subpFlag) {
  if (auto *prev{FindSymbol(name)}) {
    if (IsDummy(*prev)) {
    } else if (auto *entity{prev->detailsIf<EntityDetails>()};
               IsPointer(*prev) && !entity->type()) {
      // POINTER attribute set before interface
    } else if (inInterfaceBlock() && currScope() != prev->owner()) {
      // Procedures in an INTERFACE block do not resolve to symbols
      // in scopes between the global scope and the current scope.
    } else if (!HandlePreviousCalls(name, *prev, subpFlag)) {
      SayAlreadyDeclared(name, *prev);
    }
  }
}

Symbol &SubprogramVisitor::PushSubprogramScope(const parser::Name &name,
    Symbol::Flag subpFlag, const parser::LanguageBindingSpec *bindingSpec) {
  Symbol *symbol{GetSpecificFromGeneric(name)};
  if (!symbol) {
    if (bindingSpec && currScope().IsGlobal() && bindingSpec->v) {
      // Create this new top-level subprogram with a binding label
      // in a new global scope, so that its symbol's name won't clash
      // with another symbol that has a distinct binding label.
      PushScope(Scope::Kind::Global,
          &MakeSymbol(context().GetTempName(currScope()), Attrs{},
              MiscDetails{MiscDetails::Kind::ScopeName}));
    }
    CheckExtantProc(name, subpFlag);
    symbol = &MakeSymbol(name, SubprogramDetails{});
  }
  symbol->ReplaceName(name.source);
  symbol->set(subpFlag);
  PushScope(Scope::Kind::Subprogram, symbol);
  if (subpFlag == Symbol::Flag::Function) {
    funcResultStack().Push(currScope());
  }
  if (inInterfaceBlock()) {
    auto &details{symbol->get<SubprogramDetails>()};
    details.set_isInterface();
    if (isAbstract()) {
      symbol->attrs().set(Attr::ABSTRACT);
    } else {
      MakeExternal(*symbol);
    }
    if (isGeneric()) {
      Symbol &genericSymbol{GetGenericSymbol()};
      if (genericSymbol.has<GenericDetails>()) {
        genericSymbol.get<GenericDetails>().AddSpecificProc(
            *symbol, name.source);
      } else {
        CHECK(context().HasError(genericSymbol));
      }
    }
    set_inheritFromParent(false);
  }
  FindSymbol(name)->set(subpFlag); // PushScope() created symbol
  return *symbol;
}

void SubprogramVisitor::PushBlockDataScope(const parser::Name &name) {
  if (auto *prev{FindSymbol(name)}) {
    if (prev->attrs().test(Attr::EXTERNAL) && prev->has<ProcEntityDetails>()) {
      if (prev->test(Symbol::Flag::Subroutine) ||
          prev->test(Symbol::Flag::Function)) {
        Say2(name, "BLOCK DATA '%s' has been called"_err_en_US, *prev,
            "Previous call of '%s'"_en_US);
      }
      EraseSymbol(name);
    }
  }
  if (name.source.empty()) {
    // Don't let unnamed BLOCK DATA conflict with unnamed PROGRAM
    PushScope(Scope::Kind::BlockData, nullptr);
  } else {
    PushScope(Scope::Kind::BlockData, &MakeSymbol(name, SubprogramDetails{}));
  }
}

// If name is a generic, return specific subprogram with the same name.
Symbol *SubprogramVisitor::GetSpecificFromGeneric(const parser::Name &name) {
  // Search for the name but don't resolve it
  if (auto *symbol{currScope().FindSymbol(name.source)}) {
    if (symbol->has<SubprogramNameDetails>()) {
      if (inInterfaceBlock()) {
        // Subtle: clear any MODULE flag so that the new interface
        // symbol doesn't inherit it and ruin the ability to check it.
        symbol->attrs().reset(Attr::MODULE);
      }
    } else if (auto *details{symbol->detailsIf<GenericDetails>()}) {
      // found generic, want specific procedure
      auto *specific{details->specific()};
      if (specific && inInterfaceBlock() &&
          specific->has<SubprogramNameDetails>() &&
          specific->attrs().test(Attr::MODULE)) {
        // The shadowed procedure is a separate module procedure that is
        // actually defined later in this (sub)module.
        // Define its interface now as a new symbol.
        details->clear_specific();
        specific = nullptr;
      }
      if (!specific) {
        specific =
            &currScope().MakeSymbol(name.source, Attrs{}, SubprogramDetails{});
        if (details->derivedType()) {
          // A specific procedure with the same name as a derived type
          SayAlreadyDeclared(name, *details->derivedType());
        } else {
          details->set_specific(Resolve(name, *specific));
        }
      } else if (isGeneric()) {
        SayAlreadyDeclared(name, *specific);
      }
      if (specific->has<SubprogramNameDetails>()) {
        specific->set_details(Details{SubprogramDetails{}});
      }
      return specific;
    }
  }
  return nullptr;
}

// DeclarationVisitor implementation

bool DeclarationVisitor::BeginDecl() {
  BeginDeclTypeSpec();
  BeginArraySpec();
  return BeginAttrs();
}
void DeclarationVisitor::EndDecl() {
  EndDeclTypeSpec();
  EndArraySpec();
  EndAttrs();
}

bool DeclarationVisitor::CheckUseError(const parser::Name &name) {
  const auto *details{
      name.symbol ? name.symbol->detailsIf<UseErrorDetails>() : nullptr};
  if (!details) {
    return false;
  }
  Message &msg{Say(name, "Reference to '%s' is ambiguous"_err_en_US)};
  for (const auto &[location, module] : details->occurrences()) {
    msg.Attach(location, "'%s' was use-associated from module '%s'"_en_US,
        name.source, module->GetName().value());
  }
  context().SetError(*name.symbol);
  return true;
}

// Report error if accessibility of symbol doesn't match isPrivate.
void DeclarationVisitor::CheckAccessibility(
    const SourceName &name, bool isPrivate, Symbol &symbol) {
  if (symbol.attrs().test(Attr::PRIVATE) != isPrivate) {
    Say2(name,
        "'%s' does not have the same accessibility as its previous declaration"_err_en_US,
        symbol, "Previous declaration of '%s'"_en_US);
  }
}

void DeclarationVisitor::Post(const parser::TypeDeclarationStmt &) {
  if (!GetAttrs().HasAny({Attr::POINTER, Attr::ALLOCATABLE})) { // C702
    if (const auto *typeSpec{GetDeclTypeSpec()}) {
      if (typeSpec->category() == DeclTypeSpec::Character) {
        if (typeSpec->characterTypeSpec().length().isDeferred()) {
          Say("The type parameter LEN cannot be deferred without"
              " the POINTER or ALLOCATABLE attribute"_err_en_US);
        }
      } else if (const DerivedTypeSpec * derivedSpec{typeSpec->AsDerived()}) {
        for (const auto &pair : derivedSpec->parameters()) {
          if (pair.second.isDeferred()) {
            Say(currStmtSource().value(),
                "The value of type parameter '%s' cannot be deferred"
                " without the POINTER or ALLOCATABLE attribute"_err_en_US,
                pair.first);
          }
        }
      }
    }
  }
  EndDecl();
}

void DeclarationVisitor::Post(const parser::DimensionStmt::Declaration &x) {
  DeclareObjectEntity(std::get<parser::Name>(x.t));
}
void DeclarationVisitor::Post(const parser::CodimensionDecl &x) {
  DeclareObjectEntity(std::get<parser::Name>(x.t));
}

bool DeclarationVisitor::Pre(const parser::Initialization &) {
  // Defer inspection of initializers to Initialization() so that the
  // symbol being initialized will be available within the initialization
  // expression.
  return false;
}

void DeclarationVisitor::Post(const parser::EntityDecl &x) {
  const auto &name{std::get<parser::ObjectName>(x.t)};
  Attrs attrs{attrs_ ? HandleSaveName(name.source, *attrs_) : Attrs{}};
  Symbol &symbol{DeclareUnknownEntity(name, attrs)};
  symbol.ReplaceName(name.source);
  if (const auto &init{std::get<std::optional<parser::Initialization>>(x.t)}) {
    ConvertToObjectEntity(symbol) || ConvertToProcEntity(symbol);
    Initialization(name, *init, false);
  } else if (attrs.test(Attr::PARAMETER)) { // C882, C883
    Say(name, "Missing initialization for parameter '%s'"_err_en_US);
  }
}

void DeclarationVisitor::Post(const parser::PointerDecl &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  if (const auto &deferredShapeSpecs{
          std::get<std::optional<parser::DeferredShapeSpecList>>(x.t)}) {
    CHECK(arraySpec().empty());
    BeginArraySpec();
    set_arraySpec(AnalyzeDeferredShapeSpecList(context(), *deferredShapeSpecs));
    Symbol &symbol{DeclareObjectEntity(name, Attrs{Attr::POINTER})};
    symbol.ReplaceName(name.source);
    EndArraySpec();
  } else {
    HandleAttributeStmt(Attr::POINTER, std::get<parser::Name>(x.t));
  }
}

bool DeclarationVisitor::Pre(const parser::BindEntity &x) {
  auto kind{std::get<parser::BindEntity::Kind>(x.t)};
  auto &name{std::get<parser::Name>(x.t)};
  Symbol *symbol;
  if (kind == parser::BindEntity::Kind::Object) {
    symbol = &HandleAttributeStmt(Attr::BIND_C, name);
  } else {
    symbol = &MakeCommonBlockSymbol(name);
    symbol->attrs().set(Attr::BIND_C);
  }
  // 8.6.4(1)
  // Some entities such as named constant or module name need to checked
  // elsewhere. This is to skip the ICE caused by setting Bind name for non-name
  // things such as data type and also checks for procedures.
  if (symbol->has<CommonBlockDetails>() || symbol->has<ObjectEntityDetails>() ||
      symbol->has<EntityDetails>()) {
    SetBindNameOn(*symbol);
  } else {
    Say(name,
        "Only variable and named common block can be in BIND statement"_err_en_US);
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::OldParameterStmt &x) {
  inOldStyleParameterStmt_ = true;
  Walk(x.v);
  inOldStyleParameterStmt_ = false;
  return false;
}
bool DeclarationVisitor::Pre(const parser::NamedConstantDef &x) {
  auto &name{std::get<parser::NamedConstant>(x.t).v};
  auto &symbol{HandleAttributeStmt(Attr::PARAMETER, name)};
  if (!ConvertToObjectEntity(symbol) ||
      symbol.test(Symbol::Flag::CrayPointer) ||
      symbol.test(Symbol::Flag::CrayPointee)) {
    SayWithDecl(
        name, symbol, "PARAMETER attribute not allowed on '%s'"_err_en_US);
    return false;
  }
  const auto &expr{std::get<parser::ConstantExpr>(x.t)};
  auto &details{symbol.get<ObjectEntityDetails>()};
  if (inOldStyleParameterStmt_) {
    // non-standard extension PARAMETER statement (no parentheses)
    Walk(expr);
    auto folded{EvaluateExpr(expr)};
    if (details.type()) {
      SayWithDecl(name, symbol,
          "Alternative style PARAMETER '%s' must not already have an explicit type"_err_en_US);
    } else if (folded) {
      auto at{expr.thing.value().source};
      if (evaluate::IsActuallyConstant(*folded)) {
        if (const auto *type{currScope().GetType(*folded)}) {
          if (type->IsPolymorphic()) {
            Say(at, "The expression must not be polymorphic"_err_en_US);
          } else if (auto shape{ToArraySpec(
                         GetFoldingContext(), evaluate::GetShape(*folded))}) {
            // The type of the named constant is assumed from the expression.
            details.set_type(*type);
            details.set_init(std::move(*folded));
            details.set_shape(std::move(*shape));
          } else {
            Say(at, "The expression must have constant shape"_err_en_US);
          }
        } else {
          Say(at, "The expression must have a known type"_err_en_US);
        }
      } else {
        Say(at, "The expression must be a constant of known type"_err_en_US);
      }
    }
  } else {
    // standard-conforming PARAMETER statement (with parentheses)
    ApplyImplicitRules(symbol);
    Walk(expr);
    if (auto converted{EvaluateNonPointerInitializer(
            symbol, expr, expr.thing.value().source)}) {
      details.set_init(std::move(*converted));
    }
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::NamedConstant &x) {
  const parser::Name &name{x.v};
  if (!FindSymbol(name)) {
    Say(name, "Named constant '%s' not found"_err_en_US);
  } else {
    CheckUseError(name);
  }
  return false;
}

bool DeclarationVisitor::Pre(const parser::Enumerator &enumerator) {
  const parser::Name &name{std::get<parser::NamedConstant>(enumerator.t).v};
  Symbol *symbol{FindSymbol(name)};
  if (symbol && !symbol->has<UnknownDetails>()) {
    // Contrary to named constants appearing in a PARAMETER statement,
    // enumerator names should not have their type, dimension or any other
    // attributes defined before they are declared in the enumerator statement,
    // with the exception of accessibility.
    // This is not explicitly forbidden by the standard, but they are scalars
    // which type is left for the compiler to chose, so do not let users try to
    // tamper with that.
    SayAlreadyDeclared(name, *symbol);
    symbol = nullptr;
  } else {
    // Enumerators are treated as PARAMETER (section 7.6 paragraph (4))
    symbol = &MakeSymbol(name, Attrs{Attr::PARAMETER}, ObjectEntityDetails{});
    symbol->SetType(context().MakeNumericType(
        TypeCategory::Integer, evaluate::CInteger::kind));
  }

  if (auto &init{std::get<std::optional<parser::ScalarIntConstantExpr>>(
          enumerator.t)}) {
    Walk(*init); // Resolve names in expression before evaluation.
    if (auto value{EvaluateInt64(context(), *init)}) {
      // Cast all init expressions to C_INT so that they can then be
      // safely incremented (see 7.6 Note 2).
      enumerationState_.value = static_cast<int>(*value);
    } else {
      Say(name,
          "Enumerator value could not be computed "
          "from the given expression"_err_en_US);
      // Prevent resolution of next enumerators value
      enumerationState_.value = std::nullopt;
    }
  }

  if (symbol) {
    if (enumerationState_.value) {
      symbol->get<ObjectEntityDetails>().set_init(SomeExpr{
          evaluate::Expr<evaluate::CInteger>{*enumerationState_.value}});
    } else {
      context().SetError(*symbol);
    }
  }

  if (enumerationState_.value) {
    (*enumerationState_.value)++;
  }
  return false;
}

void DeclarationVisitor::Post(const parser::EnumDef &) {
  enumerationState_ = EnumeratorState{};
}

bool DeclarationVisitor::Pre(const parser::AccessSpec &x) {
  Attr attr{AccessSpecToAttr(x)};
  if (!NonDerivedTypeScope().IsModule()) { // C817
    Say(currStmtSource().value(),
        "%s attribute may only appear in the specification part of a module"_err_en_US,
        EnumToString(attr));
  }
  CheckAndSet(attr);
  return false;
}

bool DeclarationVisitor::Pre(const parser::AsynchronousStmt &x) {
  return HandleAttributeStmt(Attr::ASYNCHRONOUS, x.v);
}
bool DeclarationVisitor::Pre(const parser::ContiguousStmt &x) {
  return HandleAttributeStmt(Attr::CONTIGUOUS, x.v);
}
bool DeclarationVisitor::Pre(const parser::ExternalStmt &x) {
  HandleAttributeStmt(Attr::EXTERNAL, x.v);
  for (const auto &name : x.v) {
    auto *symbol{FindSymbol(name)};
    if (!ConvertToProcEntity(DEREF(symbol))) {
      SayWithDecl(
          name, *symbol, "EXTERNAL attribute not allowed on '%s'"_err_en_US);
    } else if (symbol->attrs().test(Attr::INTRINSIC)) { // C840
      Say(symbol->name(),
          "Symbol '%s' cannot have both INTRINSIC and EXTERNAL attributes"_err_en_US,
          symbol->name());
    }
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::IntentStmt &x) {
  auto &intentSpec{std::get<parser::IntentSpec>(x.t)};
  auto &names{std::get<std::list<parser::Name>>(x.t)};
  return CheckNotInBlock("INTENT") && // C1107
      HandleAttributeStmt(IntentSpecToAttr(intentSpec), names);
}
bool DeclarationVisitor::Pre(const parser::IntrinsicStmt &x) {
  HandleAttributeStmt(Attr::INTRINSIC, x.v);
  for (const auto &name : x.v) {
    if (!IsIntrinsic(name.source, std::nullopt)) {
      Say(name.source, "'%s' is not a known intrinsic procedure"_err_en_US);
    }
    auto &symbol{DEREF(FindSymbol(name))};
    if (symbol.has<GenericDetails>()) {
      // Generic interface is extending intrinsic; ok
    } else if (!ConvertToProcEntity(symbol)) {
      SayWithDecl(
          name, symbol, "INTRINSIC attribute not allowed on '%s'"_err_en_US);
    } else if (symbol.attrs().test(Attr::EXTERNAL)) { // C840
      Say(symbol.name(),
          "Symbol '%s' cannot have both EXTERNAL and INTRINSIC attributes"_err_en_US,
          symbol.name());
    } else if (symbol.GetType()) {
      // These warnings are worded so that they should make sense in either
      // order.
      Say(symbol.name(),
          "Explicit type declaration ignored for intrinsic function '%s'"_warn_en_US,
          symbol.name())
          .Attach(name.source,
              "INTRINSIC statement for explicitly-typed '%s'"_en_US,
              name.source);
    }
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::OptionalStmt &x) {
  return CheckNotInBlock("OPTIONAL") && // C1107
      HandleAttributeStmt(Attr::OPTIONAL, x.v);
}
bool DeclarationVisitor::Pre(const parser::ProtectedStmt &x) {
  return HandleAttributeStmt(Attr::PROTECTED, x.v);
}
bool DeclarationVisitor::Pre(const parser::ValueStmt &x) {
  return CheckNotInBlock("VALUE") && // C1107
      HandleAttributeStmt(Attr::VALUE, x.v);
}
bool DeclarationVisitor::Pre(const parser::VolatileStmt &x) {
  return HandleAttributeStmt(Attr::VOLATILE, x.v);
}
// Handle a statement that sets an attribute on a list of names.
bool DeclarationVisitor::HandleAttributeStmt(
    Attr attr, const std::list<parser::Name> &names) {
  for (const auto &name : names) {
    HandleAttributeStmt(attr, name);
  }
  return false;
}
Symbol &DeclarationVisitor::HandleAttributeStmt(
    Attr attr, const parser::Name &name) {
  auto *symbol{FindInScope(name)};
  if (attr == Attr::ASYNCHRONOUS || attr == Attr::VOLATILE) {
    // these can be set on a symbol that is host-assoc or use-assoc
    if (!symbol &&
        (currScope().kind() == Scope::Kind::Subprogram ||
            currScope().kind() == Scope::Kind::BlockConstruct)) {
      if (auto *hostSymbol{FindSymbol(name)}) {
        symbol = &MakeHostAssocSymbol(name, *hostSymbol);
      }
    }
  } else if (symbol && symbol->has<UseDetails>()) {
    Say(currStmtSource().value(),
        "Cannot change %s attribute on use-associated '%s'"_err_en_US,
        EnumToString(attr), name.source);
    return *symbol;
  }
  if (!symbol) {
    symbol = &MakeSymbol(name, EntityDetails{});
  }
  symbol->attrs().set(attr);
  symbol->attrs() = HandleSaveName(name.source, symbol->attrs());
  return *symbol;
}
// C1107
bool DeclarationVisitor::CheckNotInBlock(const char *stmt) {
  if (currScope().kind() == Scope::Kind::BlockConstruct) {
    Say(MessageFormattedText{
        "%s statement is not allowed in a BLOCK construct"_err_en_US, stmt});
    return false;
  } else {
    return true;
  }
}

void DeclarationVisitor::Post(const parser::ObjectDecl &x) {
  CHECK(objectDeclAttr_);
  const auto &name{std::get<parser::ObjectName>(x.t)};
  DeclareObjectEntity(name, Attrs{*objectDeclAttr_});
}

// Declare an entity not yet known to be an object or proc.
Symbol &DeclarationVisitor::DeclareUnknownEntity(
    const parser::Name &name, Attrs attrs) {
  if (!arraySpec().empty() || !coarraySpec().empty()) {
    return DeclareObjectEntity(name, attrs);
  } else {
    Symbol &symbol{DeclareEntity<EntityDetails>(name, attrs)};
    if (auto *type{GetDeclTypeSpec()}) {
      SetType(name, *type);
    }
    charInfo_.length.reset();
    if (symbol.attrs().test(Attr::EXTERNAL)) {
      ConvertToProcEntity(symbol);
    }
    SetBindNameOn(symbol);
    return symbol;
  }
}

bool DeclarationVisitor::HasCycle(
    const Symbol &procSymbol, const ProcInterface &interface) {
  SourceOrderedSymbolSet procsInCycle;
  procsInCycle.insert(procSymbol);
  const ProcInterface *thisInterface{&interface};
  bool haveInterface{true};
  while (haveInterface) {
    haveInterface = false;
    if (const Symbol * interfaceSymbol{thisInterface->symbol()}) {
      if (procsInCycle.count(*interfaceSymbol) > 0) {
        for (const auto &procInCycle : procsInCycle) {
          Say(procInCycle->name(),
              "The interface for procedure '%s' is recursively "
              "defined"_err_en_US,
              procInCycle->name());
          context().SetError(*procInCycle);
        }
        return true;
      } else if (const auto *procDetails{
                     interfaceSymbol->detailsIf<ProcEntityDetails>()}) {
        haveInterface = true;
        thisInterface = &procDetails->interface();
        procsInCycle.insert(*interfaceSymbol);
      }
    }
  }
  return false;
}

Symbol &DeclarationVisitor::DeclareProcEntity(
    const parser::Name &name, Attrs attrs, const ProcInterface &interface) {
  Symbol &symbol{DeclareEntity<ProcEntityDetails>(name, attrs)};
  if (auto *details{symbol.detailsIf<ProcEntityDetails>()}) {
    if (details->IsInterfaceSet()) {
      SayWithDecl(name, symbol,
          "The interface for procedure '%s' has already been "
          "declared"_err_en_US);
      context().SetError(symbol);
    } else if (HasCycle(symbol, interface)) {
      return symbol;
    } else if (interface.type()) {
      symbol.set(Symbol::Flag::Function);
    } else if (interface.symbol()) {
      if (interface.symbol()->test(Symbol::Flag::Function)) {
        symbol.set(Symbol::Flag::Function);
      } else if (interface.symbol()->test(Symbol::Flag::Subroutine)) {
        symbol.set(Symbol::Flag::Subroutine);
      }
    }
    details->set_interface(interface);
    SetBindNameOn(symbol);
    SetPassNameOn(symbol);
  }
  return symbol;
}

Symbol &DeclarationVisitor::DeclareObjectEntity(
    const parser::Name &name, Attrs attrs) {
  Symbol &symbol{DeclareEntity<ObjectEntityDetails>(name, attrs)};
  if (auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (auto *type{GetDeclTypeSpec()}) {
      SetType(name, *type);
    }
    if (!arraySpec().empty()) {
      if (details->IsArray()) {
        if (!context().HasError(symbol)) {
          Say(name,
              "The dimensions of '%s' have already been declared"_err_en_US);
          context().SetError(symbol);
        }
      } else {
        details->set_shape(arraySpec());
      }
    }
    if (!coarraySpec().empty()) {
      if (details->IsCoarray()) {
        if (!context().HasError(symbol)) {
          Say(name,
              "The codimensions of '%s' have already been declared"_err_en_US);
          context().SetError(symbol);
        }
      } else {
        details->set_coshape(coarraySpec());
      }
    }
    SetBindNameOn(symbol);
  }
  ClearArraySpec();
  ClearCoarraySpec();
  charInfo_.length.reset();
  return symbol;
}

void DeclarationVisitor::Post(const parser::IntegerTypeSpec &x) {
  SetDeclTypeSpec(MakeNumericType(TypeCategory::Integer, x.v));
}
void DeclarationVisitor::Post(const parser::IntrinsicTypeSpec::Real &x) {
  SetDeclTypeSpec(MakeNumericType(TypeCategory::Real, x.kind));
}
void DeclarationVisitor::Post(const parser::IntrinsicTypeSpec::Complex &x) {
  SetDeclTypeSpec(MakeNumericType(TypeCategory::Complex, x.kind));
}
void DeclarationVisitor::Post(const parser::IntrinsicTypeSpec::Logical &x) {
  SetDeclTypeSpec(MakeLogicalType(x.kind));
}
void DeclarationVisitor::Post(const parser::IntrinsicTypeSpec::Character &) {
  if (!charInfo_.length) {
    charInfo_.length = ParamValue{1, common::TypeParamAttr::Len};
  }
  if (!charInfo_.kind) {
    charInfo_.kind =
        KindExpr{context().GetDefaultKind(TypeCategory::Character)};
  }
  SetDeclTypeSpec(currScope().MakeCharacterType(
      std::move(*charInfo_.length), std::move(*charInfo_.kind)));
  charInfo_ = {};
}
void DeclarationVisitor::Post(const parser::CharSelector::LengthAndKind &x) {
  charInfo_.kind = EvaluateSubscriptIntExpr(x.kind);
  std::optional<std::int64_t> intKind{ToInt64(charInfo_.kind)};
  if (intKind &&
      !context().targetCharacteristics().IsTypeEnabled(
          TypeCategory::Character, *intKind)) { // C715, C719
    Say(currStmtSource().value(),
        "KIND value (%jd) not valid for CHARACTER"_err_en_US, *intKind);
    charInfo_.kind = std::nullopt; // prevent further errors
  }
  if (x.length) {
    charInfo_.length = GetParamValue(*x.length, common::TypeParamAttr::Len);
  }
}
void DeclarationVisitor::Post(const parser::CharLength &x) {
  if (const auto *length{std::get_if<std::uint64_t>(&x.u)}) {
    charInfo_.length = ParamValue{
        static_cast<ConstantSubscript>(*length), common::TypeParamAttr::Len};
  } else {
    charInfo_.length = GetParamValue(
        std::get<parser::TypeParamValue>(x.u), common::TypeParamAttr::Len);
  }
}
void DeclarationVisitor::Post(const parser::LengthSelector &x) {
  if (const auto *param{std::get_if<parser::TypeParamValue>(&x.u)}) {
    charInfo_.length = GetParamValue(*param, common::TypeParamAttr::Len);
  }
}

bool DeclarationVisitor::Pre(const parser::KindParam &x) {
  if (const auto *kind{std::get_if<
          parser::Scalar<parser::Integer<parser::Constant<parser::Name>>>>(
          &x.u)}) {
    const parser::Name &name{kind->thing.thing.thing};
    if (!FindSymbol(name)) {
      Say(name, "Parameter '%s' not found"_err_en_US);
    }
  }
  return false;
}

bool DeclarationVisitor::Pre(const parser::DeclarationTypeSpec::Type &) {
  CHECK(GetDeclTypeSpecCategory() == DeclTypeSpec::Category::TypeDerived);
  return true;
}

void DeclarationVisitor::Post(const parser::DeclarationTypeSpec::Type &type) {
  const parser::Name &derivedName{std::get<parser::Name>(type.derived.t)};
  if (const Symbol * derivedSymbol{derivedName.symbol}) {
    CheckForAbstractType(*derivedSymbol); // C706
  }
}

bool DeclarationVisitor::Pre(const parser::DeclarationTypeSpec::Class &) {
  SetDeclTypeSpecCategory(DeclTypeSpec::Category::ClassDerived);
  return true;
}

void DeclarationVisitor::Post(
    const parser::DeclarationTypeSpec::Class &parsedClass) {
  const auto &typeName{std::get<parser::Name>(parsedClass.derived.t)};
  if (auto spec{ResolveDerivedType(typeName)};
      spec && !IsExtensibleType(&*spec)) { // C705
    SayWithDecl(typeName, *typeName.symbol,
        "Non-extensible derived type '%s' may not be used with CLASS"
        " keyword"_err_en_US);
  }
}

void DeclarationVisitor::Post(const parser::DerivedTypeSpec &x) {
  const auto &typeName{std::get<parser::Name>(x.t)};
  auto spec{ResolveDerivedType(typeName)};
  if (!spec) {
    return;
  }
  bool seenAnyName{false};
  for (const auto &typeParamSpec :
      std::get<std::list<parser::TypeParamSpec>>(x.t)) {
    const auto &optKeyword{
        std::get<std::optional<parser::Keyword>>(typeParamSpec.t)};
    std::optional<SourceName> name;
    if (optKeyword) {
      seenAnyName = true;
      name = optKeyword->v.source;
    } else if (seenAnyName) {
      Say(typeName.source, "Type parameter value must have a name"_err_en_US);
      continue;
    }
    const auto &value{std::get<parser::TypeParamValue>(typeParamSpec.t)};
    // The expressions in a derived type specifier whose values define
    // non-defaulted type parameters are evaluated (folded) in the enclosing
    // scope.  The KIND/LEN distinction is resolved later in
    // DerivedTypeSpec::CookParameters().
    ParamValue param{GetParamValue(value, common::TypeParamAttr::Kind)};
    if (!param.isExplicit() || param.GetExplicit()) {
      spec->AddRawParamValue(optKeyword, std::move(param));
    }
  }
  // The DerivedTypeSpec *spec is used initially as a search key.
  // If it turns out to have the same name and actual parameter
  // value expressions as another DerivedTypeSpec in the current
  // scope does, then we'll use that extant spec; otherwise, when this
  // spec is distinct from all derived types previously instantiated
  // in the current scope, this spec will be moved into that collection.
  const auto &dtDetails{spec->typeSymbol().get<DerivedTypeDetails>()};
  auto category{GetDeclTypeSpecCategory()};
  if (dtDetails.isForwardReferenced()) {
    DeclTypeSpec &type{currScope().MakeDerivedType(category, std::move(*spec))};
    SetDeclTypeSpec(type);
    return;
  }
  // Normalize parameters to produce a better search key.
  spec->CookParameters(GetFoldingContext());
  if (!spec->MightBeParameterized()) {
    spec->EvaluateParameters(context());
  }
  if (const DeclTypeSpec *
      extant{currScope().FindInstantiatedDerivedType(*spec, category)}) {
    // This derived type and parameter expressions (if any) are already present
    // in this scope.
    SetDeclTypeSpec(*extant);
  } else {
    DeclTypeSpec &type{currScope().MakeDerivedType(category, std::move(*spec))};
    DerivedTypeSpec &derived{type.derivedTypeSpec()};
    if (derived.MightBeParameterized() &&
        currScope().IsParameterizedDerivedType()) {
      // Defer instantiation; use the derived type's definition's scope.
      derived.set_scope(DEREF(spec->typeSymbol().scope()));
    } else if (&currScope() == spec->typeSymbol().scope()) {
      // Direct recursive use of a type in the definition of one of its
      // components: defer instantiation
    } else {
      auto restorer{
          GetFoldingContext().messages().SetLocation(currStmtSource().value())};
      derived.Instantiate(currScope());
    }
    SetDeclTypeSpec(type);
  }
  // Capture the DerivedTypeSpec in the parse tree for use in building
  // structure constructor expressions.
  x.derivedTypeSpec = &GetDeclTypeSpec()->derivedTypeSpec();
}

void DeclarationVisitor::Post(const parser::DeclarationTypeSpec::Record &rec) {
  const auto &typeName{rec.v};
  if (auto spec{ResolveDerivedType(typeName)}) {
    spec->CookParameters(GetFoldingContext());
    spec->EvaluateParameters(context());
    if (const DeclTypeSpec *
        extant{currScope().FindInstantiatedDerivedType(
            *spec, DeclTypeSpec::TypeDerived)}) {
      SetDeclTypeSpec(*extant);
    } else {
      Say(typeName.source, "%s is not a known STRUCTURE"_err_en_US,
          typeName.source);
    }
  }
}

// The descendents of DerivedTypeDef in the parse tree are visited directly
// in this Pre() routine so that recursive use of the derived type can be
// supported in the components.
bool DeclarationVisitor::Pre(const parser::DerivedTypeDef &x) {
  auto &stmt{std::get<parser::Statement<parser::DerivedTypeStmt>>(x.t)};
  Walk(stmt);
  Walk(std::get<std::list<parser::Statement<parser::TypeParamDefStmt>>>(x.t));
  auto &scope{currScope()};
  CHECK(scope.symbol());
  CHECK(scope.symbol()->scope() == &scope);
  auto &details{scope.symbol()->get<DerivedTypeDetails>()};
  details.set_isForwardReferenced(false);
  std::set<SourceName> paramNames;
  for (auto &paramName : std::get<std::list<parser::Name>>(stmt.statement.t)) {
    details.add_paramName(paramName.source);
    auto *symbol{FindInScope(scope, paramName)};
    if (!symbol) {
      Say(paramName,
          "No definition found for type parameter '%s'"_err_en_US); // C742
      // No symbol for a type param.  Create one and mark it as containing an
      // error to improve subsequent semantic processing
      BeginAttrs();
      Symbol *typeParam{MakeTypeSymbol(
          paramName, TypeParamDetails{common::TypeParamAttr::Len})};
      context().SetError(*typeParam);
      EndAttrs();
    } else if (!symbol->has<TypeParamDetails>()) {
      Say2(paramName, "'%s' is not defined as a type parameter"_err_en_US,
          *symbol, "Definition of '%s'"_en_US); // C741
    }
    if (!paramNames.insert(paramName.source).second) {
      Say(paramName,
          "Duplicate type parameter name: '%s'"_err_en_US); // C731
    }
  }
  for (const auto &[name, symbol] : currScope()) {
    if (symbol->has<TypeParamDetails>() && !paramNames.count(name)) {
      SayDerivedType(name,
          "'%s' is not a type parameter of this derived type"_err_en_US,
          currScope()); // C741
    }
  }
  Walk(std::get<std::list<parser::Statement<parser::PrivateOrSequence>>>(x.t));
  const auto &componentDefs{
      std::get<std::list<parser::Statement<parser::ComponentDefStmt>>>(x.t)};
  Walk(componentDefs);
  if (derivedTypeInfo_.sequence) {
    details.set_sequence(true);
    if (componentDefs.empty()) { // C740
      Say(stmt.source,
          "A sequence type must have at least one component"_err_en_US);
    }
    if (!details.paramNames().empty()) { // C740
      Say(stmt.source,
          "A sequence type may not have type parameters"_err_en_US);
    }
    if (derivedTypeInfo_.extends) { // C735
      Say(stmt.source,
          "A sequence type may not have the EXTENDS attribute"_err_en_US);
    }
  }
  Walk(std::get<std::optional<parser::TypeBoundProcedurePart>>(x.t));
  Walk(std::get<parser::Statement<parser::EndTypeStmt>>(x.t));
  derivedTypeInfo_ = {};
  PopScope();
  return false;
}

bool DeclarationVisitor::Pre(const parser::DerivedTypeStmt &) {
  return BeginAttrs();
}
void DeclarationVisitor::Post(const parser::DerivedTypeStmt &x) {
  auto &name{std::get<parser::Name>(x.t)};
  // Resolve the EXTENDS() clause before creating the derived
  // type's symbol to foil attempts to recursively extend a type.
  auto *extendsName{derivedTypeInfo_.extends};
  std::optional<DerivedTypeSpec> extendsType{
      ResolveExtendsType(name, extendsName)};
  auto &symbol{MakeSymbol(name, GetAttrs(), DerivedTypeDetails{})};
  symbol.ReplaceName(name.source);
  derivedTypeInfo_.type = &symbol;
  PushScope(Scope::Kind::DerivedType, &symbol);
  if (extendsType) {
    // Declare the "parent component"; private if the type is.
    // Any symbol stored in the EXTENDS() clause is temporarily
    // hidden so that a new symbol can be created for the parent
    // component without producing spurious errors about already
    // existing.
    const Symbol &extendsSymbol{extendsType->typeSymbol()};
    auto restorer{common::ScopedSet(extendsName->symbol, nullptr)};
    if (OkToAddComponent(*extendsName, &extendsSymbol)) {
      auto &comp{DeclareEntity<ObjectEntityDetails>(*extendsName, Attrs{})};
      comp.attrs().set(
          Attr::PRIVATE, extendsSymbol.attrs().test(Attr::PRIVATE));
      comp.set(Symbol::Flag::ParentComp);
      DeclTypeSpec &type{currScope().MakeDerivedType(
          DeclTypeSpec::TypeDerived, std::move(*extendsType))};
      type.derivedTypeSpec().set_scope(*extendsSymbol.scope());
      comp.SetType(type);
      DerivedTypeDetails &details{symbol.get<DerivedTypeDetails>()};
      details.add_component(comp);
    }
  }
  EndAttrs();
}

void DeclarationVisitor::Post(const parser::TypeParamDefStmt &x) {
  auto *type{GetDeclTypeSpec()};
  auto attr{std::get<common::TypeParamAttr>(x.t)};
  for (auto &decl : std::get<std::list<parser::TypeParamDecl>>(x.t)) {
    auto &name{std::get<parser::Name>(decl.t)};
    if (Symbol * symbol{MakeTypeSymbol(name, TypeParamDetails{attr})}) {
      SetType(name, *type);
      if (auto &init{
              std::get<std::optional<parser::ScalarIntConstantExpr>>(decl.t)}) {
        if (auto maybeExpr{EvaluateNonPointerInitializer(
                *symbol, *init, init->thing.thing.thing.value().source)}) {
          if (auto *intExpr{std::get_if<SomeIntExpr>(&maybeExpr->u)}) {
            symbol->get<TypeParamDetails>().set_init(std::move(*intExpr));
          }
        }
      }
    }
  }
  EndDecl();
}
bool DeclarationVisitor::Pre(const parser::TypeAttrSpec::Extends &x) {
  if (derivedTypeInfo_.extends) {
    Say(currStmtSource().value(),
        "Attribute 'EXTENDS' cannot be used more than once"_err_en_US);
  } else {
    derivedTypeInfo_.extends = &x.v;
  }
  return false;
}

bool DeclarationVisitor::Pre(const parser::PrivateStmt &) {
  if (!currScope().parent().IsModule()) {
    Say("PRIVATE is only allowed in a derived type that is"
        " in a module"_err_en_US); // C766
  } else if (derivedTypeInfo_.sawContains) {
    derivedTypeInfo_.privateBindings = true;
  } else if (!derivedTypeInfo_.privateComps) {
    derivedTypeInfo_.privateComps = true;
  } else {
    Say("PRIVATE may not appear more than once in"
        " derived type components"_warn_en_US); // C738
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::SequenceStmt &) {
  if (derivedTypeInfo_.sequence) {
    Say("SEQUENCE may not appear more than once in"
        " derived type components"_warn_en_US); // C738
  }
  derivedTypeInfo_.sequence = true;
  return false;
}
void DeclarationVisitor::Post(const parser::ComponentDecl &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  auto attrs{GetAttrs()};
  if (derivedTypeInfo_.privateComps &&
      !attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
    attrs.set(Attr::PRIVATE);
  }
  if (const auto *declType{GetDeclTypeSpec()}) {
    if (const auto *derived{declType->AsDerived()}) {
      if (!attrs.HasAny({Attr::POINTER, Attr::ALLOCATABLE})) {
        if (derivedTypeInfo_.type == &derived->typeSymbol()) { // C744
          Say("Recursive use of the derived type requires "
              "POINTER or ALLOCATABLE"_err_en_US);
        }
      }
      // TODO: This would be more appropriate in CheckDerivedType()
      if (auto it{FindCoarrayUltimateComponent(*derived)}) { // C748
        std::string ultimateName{it.BuildResultDesignatorName()};
        // Strip off the leading "%"
        if (ultimateName.length() > 1) {
          ultimateName.erase(0, 1);
          if (attrs.HasAny({Attr::POINTER, Attr::ALLOCATABLE})) {
            evaluate::AttachDeclaration(
                Say(name.source,
                    "A component with a POINTER or ALLOCATABLE attribute may "
                    "not "
                    "be of a type with a coarray ultimate component (named "
                    "'%s')"_err_en_US,
                    ultimateName),
                derived->typeSymbol());
          }
          if (!arraySpec().empty() || !coarraySpec().empty()) {
            evaluate::AttachDeclaration(
                Say(name.source,
                    "An array or coarray component may not be of a type with a "
                    "coarray ultimate component (named '%s')"_err_en_US,
                    ultimateName),
                derived->typeSymbol());
          }
        }
      }
    }
  }
  if (OkToAddComponent(name)) {
    auto &symbol{DeclareObjectEntity(name, attrs)};
    if (symbol.has<ObjectEntityDetails>()) {
      if (auto &init{std::get<std::optional<parser::Initialization>>(x.t)}) {
        Initialization(name, *init, true);
      }
    }
    currScope().symbol()->get<DerivedTypeDetails>().add_component(symbol);
  }
  ClearArraySpec();
  ClearCoarraySpec();
}
void DeclarationVisitor::Post(const parser::FillDecl &x) {
  // Replace "%FILL" with a distinct generated name
  const auto &name{std::get<parser::Name>(x.t)};
  const_cast<SourceName &>(name.source) = context().GetTempName(currScope());
  if (OkToAddComponent(name)) {
    auto &symbol{DeclareObjectEntity(name, GetAttrs())};
    currScope().symbol()->get<DerivedTypeDetails>().add_component(symbol);
  }
  ClearArraySpec();
}
bool DeclarationVisitor::Pre(const parser::ProcedureDeclarationStmt &x) {
  CHECK(!interfaceName_);
  const auto &procAttrSpec{std::get<std::list<parser::ProcAttrSpec>>(x.t)};
  for (const parser::ProcAttrSpec &procAttr : procAttrSpec) {
    if (auto *bindC{std::get_if<parser::LanguageBindingSpec>(&procAttr.u)}) {
      if (bindC->v.has_value()) {
        hasBindCName_ = true;
        break;
      }
    }
  }
  return BeginDecl();
}
void DeclarationVisitor::Post(const parser::ProcedureDeclarationStmt &) {
  interfaceName_ = nullptr;
  hasBindCName_ = false;
  EndDecl();
}
bool DeclarationVisitor::Pre(const parser::DataComponentDefStmt &x) {
  // Overrides parse tree traversal so as to handle attributes first,
  // so POINTER & ALLOCATABLE enable forward references to derived types.
  Walk(std::get<std::list<parser::ComponentAttrSpec>>(x.t));
  set_allowForwardReferenceToDerivedType(
      GetAttrs().HasAny({Attr::POINTER, Attr::ALLOCATABLE}));
  Walk(std::get<parser::DeclarationTypeSpec>(x.t));
  set_allowForwardReferenceToDerivedType(false);
  if (derivedTypeInfo_.sequence) { // C740
    if (const auto *declType{GetDeclTypeSpec()}) {
      if (!declType->AsIntrinsic() && !declType->IsSequenceType() &&
          !InModuleFile()) {
        if (GetAttrs().test(Attr::POINTER) &&
            context().IsEnabled(common::LanguageFeature::PointerInSeqType)) {
          if (context().ShouldWarn(common::LanguageFeature::PointerInSeqType)) {
            Say("A sequence type data component that is a pointer to a non-sequence type is not standard"_port_en_US);
          }
        } else {
          Say("A sequence type data component must either be of an intrinsic type or a derived sequence type"_err_en_US);
        }
      }
    }
  }
  Walk(std::get<std::list<parser::ComponentOrFill>>(x.t));
  return false;
}
bool DeclarationVisitor::Pre(const parser::ProcComponentDefStmt &) {
  CHECK(!interfaceName_);
  return true;
}
void DeclarationVisitor::Post(const parser::ProcComponentDefStmt &) {
  interfaceName_ = nullptr;
}
bool DeclarationVisitor::Pre(const parser::ProcPointerInit &x) {
  if (auto *name{std::get_if<parser::Name>(&x.u)}) {
    return !NameIsKnownOrIntrinsic(*name) && !CheckUseError(*name);
  }
  return true;
}
void DeclarationVisitor::Post(const parser::ProcInterface &x) {
  if (auto *name{std::get_if<parser::Name>(&x.u)}) {
    interfaceName_ = name;
    NoteInterfaceName(*name);
  }
}
void DeclarationVisitor::Post(const parser::ProcDecl &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  ProcInterface interface;
  if (interfaceName_) {
    interface.set_symbol(*interfaceName_->symbol);
  } else if (auto *type{GetDeclTypeSpec()}) {
    interface.set_type(*type);
  }
  auto attrs{HandleSaveName(name.source, GetAttrs())};
  DerivedTypeDetails *dtDetails{nullptr};
  if (Symbol * symbol{currScope().symbol()}) {
    dtDetails = symbol->detailsIf<DerivedTypeDetails>();
  }
  if (!dtDetails) {
    attrs.set(Attr::EXTERNAL);
  }
  Symbol &symbol{DeclareProcEntity(name, attrs, interface)};
  symbol.ReplaceName(name.source);
  if (dtDetails) {
    dtDetails->add_component(symbol);
  }
  if (hasBindCName_ && (IsPointer(symbol) || IsDummy(symbol))) {
    Say(symbol.name(),
        "BIND(C) procedure with NAME= specified can neither have POINTER attribute nor be a dummy procedure"_err_en_US);
  }
}

bool DeclarationVisitor::Pre(const parser::TypeBoundProcedurePart &) {
  derivedTypeInfo_.sawContains = true;
  return true;
}

// Resolve binding names from type-bound generics, saved in genericBindings_.
void DeclarationVisitor::Post(const parser::TypeBoundProcedurePart &) {
  // track specifics seen for the current generic to detect duplicates:
  const Symbol *currGeneric{nullptr};
  std::set<SourceName> specifics;
  for (const auto &[generic, bindingName] : genericBindings_) {
    if (generic != currGeneric) {
      currGeneric = generic;
      specifics.clear();
    }
    auto [it, inserted]{specifics.insert(bindingName->source)};
    if (!inserted) {
      Say(*bindingName, // C773
          "Binding name '%s' was already specified for generic '%s'"_err_en_US,
          bindingName->source, generic->name())
          .Attach(*it, "Previous specification of '%s'"_en_US, *it);
      continue;
    }
    auto *symbol{FindInTypeOrParents(*bindingName)};
    if (!symbol) {
      Say(*bindingName, // C772
          "Binding name '%s' not found in this derived type"_err_en_US);
    } else if (!symbol->has<ProcBindingDetails>()) {
      SayWithDecl(*bindingName, *symbol, // C772
          "'%s' is not the name of a specific binding of this type"_err_en_US);
    } else {
      generic->get<GenericDetails>().AddSpecificProc(
          *symbol, bindingName->source);
    }
  }
  genericBindings_.clear();
}

void DeclarationVisitor::Post(const parser::ContainsStmt &) {
  if (derivedTypeInfo_.sequence) {
    Say("A sequence type may not have a CONTAINS statement"_err_en_US); // C740
  }
}

void DeclarationVisitor::Post(
    const parser::TypeBoundProcedureStmt::WithoutInterface &x) {
  if (GetAttrs().test(Attr::DEFERRED)) { // C783
    Say("DEFERRED is only allowed when an interface-name is provided"_err_en_US);
  }
  for (auto &declaration : x.declarations) {
    auto &bindingName{std::get<parser::Name>(declaration.t)};
    auto &optName{std::get<std::optional<parser::Name>>(declaration.t)};
    const parser::Name &procedureName{optName ? *optName : bindingName};
    Symbol *procedure{FindSymbol(procedureName)};
    if (!procedure) {
      procedure = NoteInterfaceName(procedureName);
    }
    if (procedure) {
      if (auto *s{
              MakeTypeSymbol(bindingName, ProcBindingDetails{*procedure})}) {
        SetPassNameOn(*s);
        if (GetAttrs().test(Attr::DEFERRED)) {
          context().SetError(*s);
        }
      }
    }
  }
}

void DeclarationVisitor::CheckBindings(
    const parser::TypeBoundProcedureStmt::WithoutInterface &tbps) {
  CHECK(currScope().IsDerivedType());
  for (auto &declaration : tbps.declarations) {
    auto &bindingName{std::get<parser::Name>(declaration.t)};
    if (Symbol * binding{FindInScope(bindingName)}) {
      if (auto *details{binding->detailsIf<ProcBindingDetails>()}) {
        const Symbol *procedure{FindSubprogram(details->symbol())};
        if (!CanBeTypeBoundProc(procedure)) {
          if (details->symbol().name() != binding->name()) {
            Say(binding->name(),
                "The binding of '%s' ('%s') must be either an accessible "
                "module procedure or an external procedure with "
                "an explicit interface"_err_en_US,
                binding->name(), details->symbol().name());
          } else {
            Say(binding->name(),
                "'%s' must be either an accessible module procedure "
                "or an external procedure with an explicit interface"_err_en_US,
                binding->name());
          }
          context().SetError(*binding);
        }
      }
    }
  }
}

void DeclarationVisitor::Post(
    const parser::TypeBoundProcedureStmt::WithInterface &x) {
  if (!GetAttrs().test(Attr::DEFERRED)) { // C783
    Say("DEFERRED is required when an interface-name is provided"_err_en_US);
  }
  if (Symbol * interface{NoteInterfaceName(x.interfaceName)}) {
    for (auto &bindingName : x.bindingNames) {
      if (auto *s{
              MakeTypeSymbol(bindingName, ProcBindingDetails{*interface})}) {
        SetPassNameOn(*s);
        if (!GetAttrs().test(Attr::DEFERRED)) {
          context().SetError(*s);
        }
      }
    }
  }
}

void DeclarationVisitor::Post(const parser::FinalProcedureStmt &x) {
  if (currScope().IsDerivedType() && currScope().symbol()) {
    if (auto *details{currScope().symbol()->detailsIf<DerivedTypeDetails>()}) {
      for (const auto &subrName : x.v) {
        if (const auto *name{ResolveName(subrName)}) {
          auto pair{
              details->finals().emplace(name->source, DEREF(name->symbol))};
          if (!pair.second) { // C787
            Say(name->source,
                "FINAL subroutine '%s' already appeared in this derived type"_err_en_US,
                name->source)
                .Attach(pair.first->first,
                    "earlier appearance of this FINAL subroutine"_en_US);
          }
        }
      }
    }
  }
}

bool DeclarationVisitor::Pre(const parser::TypeBoundGenericStmt &x) {
  const auto &accessSpec{std::get<std::optional<parser::AccessSpec>>(x.t)};
  const auto &genericSpec{std::get<Indirection<parser::GenericSpec>>(x.t)};
  const auto &bindingNames{std::get<std::list<parser::Name>>(x.t)};
  auto info{GenericSpecInfo{genericSpec.value()}};
  SourceName symbolName{info.symbolName()};
  bool isPrivate{accessSpec ? accessSpec->v == parser::AccessSpec::Kind::Private
                            : derivedTypeInfo_.privateBindings};
  auto *genericSymbol{FindInScope(symbolName)};
  if (genericSymbol) {
    if (!genericSymbol->has<GenericDetails>()) {
      genericSymbol = nullptr; // MakeTypeSymbol will report the error below
    }
  } else {
    // look in parent types:
    Symbol *inheritedSymbol{nullptr};
    for (const auto &name : GetAllNames(context(), symbolName)) {
      inheritedSymbol = currScope().FindComponent(SourceName{name});
      if (inheritedSymbol) {
        break;
      }
    }
    if (inheritedSymbol && inheritedSymbol->has<GenericDetails>()) {
      CheckAccessibility(symbolName, isPrivate, *inheritedSymbol); // C771
    }
  }
  if (genericSymbol) {
    CheckAccessibility(symbolName, isPrivate, *genericSymbol); // C771
  } else {
    genericSymbol = MakeTypeSymbol(symbolName, GenericDetails{});
    if (!genericSymbol) {
      return false;
    }
    if (isPrivate) {
      genericSymbol->attrs().set(Attr::PRIVATE);
    }
  }
  for (const parser::Name &bindingName : bindingNames) {
    genericBindings_.emplace(genericSymbol, &bindingName);
  }
  info.Resolve(genericSymbol);
  return false;
}

// DEC STRUCTUREs are handled thus to allow for nested definitions.
bool DeclarationVisitor::Pre(const parser::StructureDef &def) {
  const auto &structureStatement{
      std::get<parser::Statement<parser::StructureStmt>>(def.t)};
  auto saveDerivedTypeInfo{derivedTypeInfo_};
  derivedTypeInfo_ = {};
  derivedTypeInfo_.isStructure = true;
  derivedTypeInfo_.sequence = true;
  Scope *previousStructure{nullptr};
  if (saveDerivedTypeInfo.isStructure) {
    previousStructure = &currScope();
    PopScope();
  }
  const parser::StructureStmt &structStmt{structureStatement.statement};
  const auto &name{std::get<std::optional<parser::Name>>(structStmt.t)};
  if (!name) {
    // Construct a distinct generated name for an anonymous structure
    auto &mutableName{const_cast<std::optional<parser::Name> &>(name)};
    mutableName.emplace(
        parser::Name{context().GetTempName(currScope()), nullptr});
  }
  auto &symbol{MakeSymbol(*name, DerivedTypeDetails{})};
  symbol.ReplaceName(name->source);
  symbol.get<DerivedTypeDetails>().set_sequence(true);
  symbol.get<DerivedTypeDetails>().set_isDECStructure(true);
  derivedTypeInfo_.type = &symbol;
  PushScope(Scope::Kind::DerivedType, &symbol);
  const auto &fields{std::get<std::list<parser::StructureField>>(def.t)};
  Walk(fields);
  PopScope();
  // Complete the definition
  DerivedTypeSpec derivedTypeSpec{symbol.name(), symbol};
  derivedTypeSpec.set_scope(DEREF(symbol.scope()));
  derivedTypeSpec.CookParameters(GetFoldingContext());
  derivedTypeSpec.EvaluateParameters(context());
  DeclTypeSpec &type{currScope().MakeDerivedType(
      DeclTypeSpec::TypeDerived, std::move(derivedTypeSpec))};
  type.derivedTypeSpec().Instantiate(currScope());
  // Restore previous structure definition context, if any
  derivedTypeInfo_ = saveDerivedTypeInfo;
  if (previousStructure) {
    PushScope(*previousStructure);
  }
  // Handle any entity declarations on the STRUCTURE statement
  const auto &decls{std::get<std::list<parser::EntityDecl>>(structStmt.t)};
  if (!decls.empty()) {
    BeginDecl();
    SetDeclTypeSpec(type);
    Walk(decls);
    EndDecl();
  }
  return false;
}

bool DeclarationVisitor::Pre(const parser::Union::UnionStmt &) {
  Say("support for UNION"_todo_en_US); // TODO
  return true;
}

bool DeclarationVisitor::Pre(const parser::StructureField &x) {
  if (std::holds_alternative<parser::Statement<parser::DataComponentDefStmt>>(
          x.u)) {
    BeginDecl();
  }
  return true;
}

void DeclarationVisitor::Post(const parser::StructureField &x) {
  if (std::holds_alternative<parser::Statement<parser::DataComponentDefStmt>>(
          x.u)) {
    EndDecl();
  }
}

bool DeclarationVisitor::Pre(const parser::AllocateStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclarationVisitor::Post(const parser::AllocateStmt &) {
  EndDeclTypeSpec();
}

bool DeclarationVisitor::Pre(const parser::StructureConstructor &x) {
  auto &parsedType{std::get<parser::DerivedTypeSpec>(x.t)};
  const DeclTypeSpec *type{ProcessTypeSpec(parsedType)};
  if (!type) {
    return false;
  }
  const DerivedTypeSpec *spec{type->AsDerived()};
  const Scope *typeScope{spec ? spec->scope() : nullptr};
  if (!typeScope) {
    return false;
  }

  // N.B C7102 is implicitly enforced by having inaccessible types not
  // being found in resolution.
  // More constraints are enforced in expression.cpp so that they
  // can apply to structure constructors that have been converted
  // from misparsed function references.
  for (const auto &component :
      std::get<std::list<parser::ComponentSpec>>(x.t)) {
    // Visit the component spec expression, but not the keyword, since
    // we need to resolve its symbol in the scope of the derived type.
    Walk(std::get<parser::ComponentDataSource>(component.t));
    if (const auto &kw{std::get<std::optional<parser::Keyword>>(component.t)}) {
      FindInTypeOrParents(*typeScope, kw->v);
    }
  }
  return false;
}

bool DeclarationVisitor::Pre(const parser::BasedPointerStmt &x) {
  for (const parser::BasedPointer &bp : x.v) {
    const parser::ObjectName &pointerName{std::get<0>(bp.t)};
    const parser::ObjectName &pointeeName{std::get<1>(bp.t)};
    auto *pointer{FindSymbol(pointerName)};
    if (!pointer) {
      pointer = &MakeSymbol(pointerName, ObjectEntityDetails{});
    } else if (!ConvertToObjectEntity(*pointer) || IsNamedConstant(*pointer)) {
      SayWithDecl(pointerName, *pointer, "'%s' is not a variable"_err_en_US);
    } else if (pointer->Rank() > 0) {
      SayWithDecl(pointerName, *pointer,
          "Cray pointer '%s' must be a scalar"_err_en_US);
    } else if (pointer->test(Symbol::Flag::CrayPointee)) {
      Say(pointerName,
          "'%s' cannot be a Cray pointer as it is already a Cray pointee"_err_en_US);
    }
    pointer->set(Symbol::Flag::CrayPointer);
    const DeclTypeSpec &pointerType{MakeNumericType(TypeCategory::Integer,
        context().defaultKinds().subscriptIntegerKind())};
    const auto *type{pointer->GetType()};
    if (!type) {
      pointer->SetType(pointerType);
    } else if (*type != pointerType) {
      Say(pointerName.source, "Cray pointer '%s' must have type %s"_err_en_US,
          pointerName.source, pointerType.AsFortran());
    }
    if (ResolveName(pointeeName)) {
      Symbol &pointee{*pointeeName.symbol};
      if (pointee.has<UseDetails>()) {
        Say(pointeeName,
            "'%s' cannot be a Cray pointee as it is use-associated"_err_en_US);
        continue;
      } else if (!ConvertToObjectEntity(pointee) || IsNamedConstant(pointee)) {
        Say(pointeeName, "'%s' is not a variable"_err_en_US);
        continue;
      } else if (pointee.test(Symbol::Flag::CrayPointer)) {
        Say(pointeeName,
            "'%s' cannot be a Cray pointee as it is already a Cray pointer"_err_en_US);
      } else if (pointee.test(Symbol::Flag::CrayPointee)) {
        Say(pointeeName,
            "'%s' was already declared as a Cray pointee"_err_en_US);
      } else {
        pointee.set(Symbol::Flag::CrayPointee);
      }
      if (const auto *pointeeType{pointee.GetType()}) {
        if (const auto *derived{pointeeType->AsDerived()}) {
          if (!derived->typeSymbol().get<DerivedTypeDetails>().sequence()) {
            Say(pointeeName,
                "Type of Cray pointee '%s' is a non-sequence derived type"_err_en_US);
          }
        }
      }
      // process the pointee array-spec, if present
      BeginArraySpec();
      Walk(std::get<std::optional<parser::ArraySpec>>(bp.t));
      const auto &spec{arraySpec()};
      if (!spec.empty()) {
        auto &details{pointee.get<ObjectEntityDetails>()};
        if (details.shape().empty()) {
          details.set_shape(spec);
        } else {
          SayWithDecl(pointeeName, pointee,
              "Array spec was already declared for '%s'"_err_en_US);
        }
      }
      ClearArraySpec();
      currScope().add_crayPointer(pointeeName.source, *pointer);
    }
  }
  return false;
}

bool DeclarationVisitor::Pre(const parser::NamelistStmt::Group &x) {
  if (!CheckNotInBlock("NAMELIST")) { // C1107
    return false;
  }
  const auto &groupName{std::get<parser::Name>(x.t)};
  auto *groupSymbol{FindInScope(groupName)};
  if (!groupSymbol || !groupSymbol->has<NamelistDetails>()) {
    groupSymbol = &MakeSymbol(groupName, NamelistDetails{});
    groupSymbol->ReplaceName(groupName.source);
  }
  // Name resolution of group items is deferred to FinishNamelists()
  // so that host association is handled correctly.
  GetDeferredDeclarationState(true)->namelistGroups.emplace_back(&x);
  return false;
}

void DeclarationVisitor::FinishNamelists() {
  if (auto *deferred{GetDeferredDeclarationState()}) {
    for (const parser::NamelistStmt::Group *group : deferred->namelistGroups) {
      if (auto *groupSymbol{FindInScope(std::get<parser::Name>(group->t))}) {
        if (auto *details{groupSymbol->detailsIf<NamelistDetails>()}) {
          for (const auto &name : std::get<std::list<parser::Name>>(group->t)) {
            auto *symbol{FindSymbol(name)};
            if (!symbol) {
              symbol = &MakeSymbol(name, ObjectEntityDetails{});
              ApplyImplicitRules(*symbol);
            } else if (!ConvertToObjectEntity(*symbol)) {
              SayWithDecl(name, *symbol, "'%s' is not a variable"_err_en_US);
            }
            symbol->GetUltimate().set(Symbol::Flag::InNamelist);
            details->add_object(*symbol);
          }
        }
      }
    }
    deferred->namelistGroups.clear();
  }
}

bool DeclarationVisitor::Pre(const parser::IoControlSpec &x) {
  if (const auto *name{std::get_if<parser::Name>(&x.u)}) {
    auto *symbol{FindSymbol(*name)};
    if (!symbol) {
      Say(*name, "Namelist group '%s' not found"_err_en_US);
    } else if (!symbol->GetUltimate().has<NamelistDetails>()) {
      SayWithDecl(
          *name, *symbol, "'%s' is not the name of a namelist group"_err_en_US);
    }
  }
  return true;
}

bool DeclarationVisitor::Pre(const parser::CommonStmt::Block &x) {
  CheckNotInBlock("COMMON"); // C1107
  return true;
}

bool DeclarationVisitor::Pre(const parser::CommonBlockObject &) {
  BeginArraySpec();
  return true;
}

void DeclarationVisitor::Post(const parser::CommonBlockObject &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  DeclareObjectEntity(name);
  auto pair{specPartState_.commonBlockObjects.insert(name.source)};
  if (!pair.second) {
    const SourceName &prev{*pair.first};
    Say2(name.source, "'%s' is already in a COMMON block"_err_en_US, prev,
        "Previous occurrence of '%s' in a COMMON block"_en_US);
  }
}

bool DeclarationVisitor::Pre(const parser::EquivalenceStmt &x) {
  // save equivalence sets to be processed after specification part
  if (CheckNotInBlock("EQUIVALENCE")) { // C1107
    for (const std::list<parser::EquivalenceObject> &set : x.v) {
      specPartState_.equivalenceSets.push_back(&set);
    }
  }
  return false; // don't implicitly declare names yet
}

void DeclarationVisitor::CheckEquivalenceSets() {
  EquivalenceSets equivSets{context()};
  inEquivalenceStmt_ = true;
  for (const auto *set : specPartState_.equivalenceSets) {
    const auto &source{set->front().v.value().source};
    if (set->size() <= 1) { // R871
      Say(source, "Equivalence set must have more than one object"_err_en_US);
    }
    for (const parser::EquivalenceObject &object : *set) {
      const auto &designator{object.v.value()};
      // The designator was not resolved when it was encountered so do it now.
      // AnalyzeExpr causes array sections to be changed to substrings as needed
      Walk(designator);
      if (AnalyzeExpr(context(), designator)) {
        equivSets.AddToSet(designator);
      }
    }
    equivSets.FinishSet(source);
  }
  inEquivalenceStmt_ = false;
  for (auto &set : equivSets.sets()) {
    if (!set.empty()) {
      currScope().add_equivalenceSet(std::move(set));
    }
  }
  specPartState_.equivalenceSets.clear();
}

bool DeclarationVisitor::Pre(const parser::SaveStmt &x) {
  if (x.v.empty()) {
    specPartState_.saveInfo.saveAll = currStmtSource();
    currScope().set_hasSAVE();
  } else {
    for (const parser::SavedEntity &y : x.v) {
      auto kind{std::get<parser::SavedEntity::Kind>(y.t)};
      const auto &name{std::get<parser::Name>(y.t)};
      if (kind == parser::SavedEntity::Kind::Common) {
        MakeCommonBlockSymbol(name);
        AddSaveName(specPartState_.saveInfo.commons, name.source);
      } else {
        HandleAttributeStmt(Attr::SAVE, name);
      }
    }
  }
  return false;
}

void DeclarationVisitor::CheckSaveStmts() {
  for (const SourceName &name : specPartState_.saveInfo.entities) {
    auto *symbol{FindInScope(name)};
    if (!symbol) {
      // error was reported
    } else if (specPartState_.saveInfo.saveAll) {
      // C889 - note that pgi, ifort, xlf do not enforce this constraint
      Say2(name,
          "Explicit SAVE of '%s' is redundant due to global SAVE statement"_warn_en_US,
          *specPartState_.saveInfo.saveAll, "Global SAVE statement"_en_US);
    } else if (auto msg{CheckSaveAttr(*symbol)}) {
      Say(name, std::move(*msg));
      context().SetError(*symbol);
    } else {
      SetSaveAttr(*symbol);
    }
  }
  for (const SourceName &name : specPartState_.saveInfo.commons) {
    if (auto *symbol{currScope().FindCommonBlock(name)}) {
      auto &objects{symbol->get<CommonBlockDetails>().objects()};
      if (objects.empty()) {
        if (currScope().kind() != Scope::Kind::BlockConstruct) {
          Say(name,
              "'%s' appears as a COMMON block in a SAVE statement but not in"
              " a COMMON statement"_err_en_US);
        } else { // C1108
          Say(name,
              "SAVE statement in BLOCK construct may not contain a"
              " common block name '%s'"_err_en_US);
        }
      } else {
        for (auto &object : symbol->get<CommonBlockDetails>().objects()) {
          SetSaveAttr(*object);
        }
      }
    }
  }
  if (specPartState_.saveInfo.saveAll) {
    // Apply SAVE attribute to applicable symbols
    for (auto pair : currScope()) {
      auto &symbol{*pair.second};
      if (!CheckSaveAttr(symbol)) {
        SetSaveAttr(symbol);
      }
    }
  }
  specPartState_.saveInfo = {};
}

// If SAVE attribute can't be set on symbol, return error message.
std::optional<MessageFixedText> DeclarationVisitor::CheckSaveAttr(
    const Symbol &symbol) {
  if (IsDummy(symbol)) {
    return "SAVE attribute may not be applied to dummy argument '%s'"_err_en_US;
  } else if (symbol.IsFuncResult()) {
    return "SAVE attribute may not be applied to function result '%s'"_err_en_US;
  } else if (symbol.has<ProcEntityDetails>() &&
      !symbol.attrs().test(Attr::POINTER)) {
    return "Procedure '%s' with SAVE attribute must also have POINTER attribute"_err_en_US;
  } else if (IsAutomatic(symbol)) {
    return "SAVE attribute may not be applied to automatic data object '%s'"_err_en_US;
  } else {
    return std::nullopt;
  }
}

// Record SAVEd names in specPartState_.saveInfo.entities.
Attrs DeclarationVisitor::HandleSaveName(const SourceName &name, Attrs attrs) {
  if (attrs.test(Attr::SAVE)) {
    AddSaveName(specPartState_.saveInfo.entities, name);
  }
  return attrs;
}

// Record a name in a set of those to be saved.
void DeclarationVisitor::AddSaveName(
    std::set<SourceName> &set, const SourceName &name) {
  auto pair{set.insert(name)};
  if (!pair.second) {
    Say2(name, "SAVE attribute was already specified on '%s'"_warn_en_US,
        *pair.first, "Previous specification of SAVE attribute"_en_US);
  }
}

// Set the SAVE attribute on symbol unless it is implicitly saved anyway.
void DeclarationVisitor::SetSaveAttr(Symbol &symbol) {
  if (!IsSaved(symbol)) {
    symbol.attrs().set(Attr::SAVE);
  }
}

// Check types of common block objects, now that they are known.
void DeclarationVisitor::CheckCommonBlocks() {
  // check for empty common blocks
  for (const auto &pair : currScope().commonBlocks()) {
    const auto &symbol{*pair.second};
    if (symbol.get<CommonBlockDetails>().objects().empty() &&
        symbol.attrs().test(Attr::BIND_C)) {
      Say(symbol.name(),
          "'%s' appears as a COMMON block in a BIND statement but not in"
          " a COMMON statement"_err_en_US);
    }
  }
  // check objects in common blocks
  for (const auto &name : specPartState_.commonBlockObjects) {
    const auto *symbol{currScope().FindSymbol(name)};
    if (!symbol) {
      continue;
    }
    const auto &attrs{symbol->attrs()};
    if (attrs.test(Attr::ALLOCATABLE)) {
      Say(name,
          "ALLOCATABLE object '%s' may not appear in a COMMON block"_err_en_US);
    } else if (attrs.test(Attr::BIND_C)) {
      Say(name,
          "Variable '%s' with BIND attribute may not appear in a COMMON block"_err_en_US);
    } else if (IsNamedConstant(*symbol)) {
      Say(name,
          "A named constant '%s' may not appear in a COMMON block"_err_en_US);
    } else if (IsDummy(*symbol)) {
      Say(name,
          "Dummy argument '%s' may not appear in a COMMON block"_err_en_US);
    } else if (symbol->IsFuncResult()) {
      Say(name,
          "Function result '%s' may not appear in a COMMON block"_err_en_US);
    } else if (const DeclTypeSpec * type{symbol->GetType()}) {
      if (type->category() == DeclTypeSpec::ClassStar) {
        Say(name,
            "Unlimited polymorphic pointer '%s' may not appear in a COMMON block"_err_en_US);
      } else if (const auto *derived{type->AsDerived()}) {
        auto &typeSymbol{derived->typeSymbol()};
        if (!typeSymbol.attrs().test(Attr::BIND_C) &&
            !typeSymbol.get<DerivedTypeDetails>().sequence()) {
          Say(name,
              "Derived type '%s' in COMMON block must have the BIND or"
              " SEQUENCE attribute"_err_en_US);
        }
        CheckCommonBlockDerivedType(name, typeSymbol);
      }
    }
  }
  specPartState_.commonBlockObjects = {};
}

Symbol &DeclarationVisitor::MakeCommonBlockSymbol(const parser::Name &name) {
  return Resolve(name, currScope().MakeCommonBlock(name.source));
}
Symbol &DeclarationVisitor::MakeCommonBlockSymbol(
    const std::optional<parser::Name> &name) {
  if (name) {
    return MakeCommonBlockSymbol(*name);
  } else {
    return MakeCommonBlockSymbol(parser::Name{});
  }
}

bool DeclarationVisitor::NameIsKnownOrIntrinsic(const parser::Name &name) {
  return FindSymbol(name) || HandleUnrestrictedSpecificIntrinsicFunction(name);
}

// Check if this derived type can be in a COMMON block.
void DeclarationVisitor::CheckCommonBlockDerivedType(
    const SourceName &name, const Symbol &typeSymbol) {
  if (const auto *scope{typeSymbol.scope()}) {
    for (const auto &pair : *scope) {
      const Symbol &component{*pair.second};
      if (component.attrs().test(Attr::ALLOCATABLE)) {
        Say2(name,
            "Derived type variable '%s' may not appear in a COMMON block"
            " due to ALLOCATABLE component"_err_en_US,
            component.name(), "Component with ALLOCATABLE attribute"_en_US);
        return;
      }
      const auto *details{component.detailsIf<ObjectEntityDetails>()};
      if (component.test(Symbol::Flag::InDataStmt) ||
          (details && details->init())) {
        Say2(name,
            "Derived type variable '%s' may not appear in a COMMON block due to component with default initialization"_err_en_US,
            component.name(), "Component with default initialization"_en_US);
        return;
      }
      if (details) {
        if (const auto *type{details->type()}) {
          if (const auto *derived{type->AsDerived()}) {
            CheckCommonBlockDerivedType(name, derived->typeSymbol());
          }
        }
      }
    }
  }
}

bool DeclarationVisitor::HandleUnrestrictedSpecificIntrinsicFunction(
    const parser::Name &name) {
  if (auto interface{context().intrinsics().IsSpecificIntrinsicFunction(
          name.source.ToString())}) {
    // Unrestricted specific intrinsic function names (e.g., "cos")
    // are acceptable as procedure interfaces.  The presence of the
    // INTRINSIC flag will cause this symbol to have a complete interface
    // recreated for it later on demand, but capturing its result type here
    // will make GetType() return a correct result without having to
    // probe the intrinsics table again.
    Symbol &symbol{
        MakeSymbol(InclusiveScope(), name.source, Attrs{Attr::INTRINSIC})};
    CHECK(interface->functionResult.has_value());
    evaluate::DynamicType dyType{
        DEREF(interface->functionResult->GetTypeAndShape()).type()};
    CHECK(common::IsNumericTypeCategory(dyType.category()));
    const DeclTypeSpec &typeSpec{
        MakeNumericType(dyType.category(), dyType.kind())};
    ProcEntityDetails details;
    ProcInterface procInterface;
    procInterface.set_type(typeSpec);
    details.set_interface(procInterface);
    symbol.set_details(std::move(details));
    symbol.set(Symbol::Flag::Function);
    if (interface->IsElemental()) {
      symbol.attrs().set(Attr::ELEMENTAL);
    }
    if (interface->IsPure()) {
      symbol.attrs().set(Attr::PURE);
    }
    Resolve(name, symbol);
    return true;
  } else {
    return false;
  }
}

// Checks for all locality-specs: LOCAL, LOCAL_INIT, and SHARED
bool DeclarationVisitor::PassesSharedLocalityChecks(
    const parser::Name &name, Symbol &symbol) {
  if (!IsVariableName(symbol)) {
    SayLocalMustBeVariable(name, symbol); // C1124
    return false;
  }
  if (symbol.owner() == currScope()) { // C1125 and C1126
    SayAlreadyDeclared(name, symbol);
    return false;
  }
  return true;
}

// Checks for locality-specs LOCAL and LOCAL_INIT
bool DeclarationVisitor::PassesLocalityChecks(
    const parser::Name &name, Symbol &symbol) {
  if (IsAllocatable(symbol)) { // C1128
    SayWithDecl(name, symbol,
        "ALLOCATABLE variable '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (IsOptional(symbol)) { // C1128
    SayWithDecl(name, symbol,
        "OPTIONAL argument '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (IsIntentIn(symbol)) { // C1128
    SayWithDecl(name, symbol,
        "INTENT IN argument '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (IsFinalizable(symbol)) { // C1128
    SayWithDecl(name, symbol,
        "Finalizable variable '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (evaluate::IsCoarray(symbol)) { // C1128
    SayWithDecl(
        name, symbol, "Coarray '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    if (type->IsPolymorphic() && IsDummy(symbol) &&
        !IsPointer(symbol)) { // C1128
      SayWithDecl(name, symbol,
          "Nonpointer polymorphic argument '%s' not allowed in a "
          "locality-spec"_err_en_US);
      return false;
    }
  }
  if (IsAssumedSizeArray(symbol)) { // C1128
    SayWithDecl(name, symbol,
        "Assumed size array '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (std::optional<Message> msg{WhyNotModifiable(symbol, currScope())}) {
    SayWithReason(name, symbol,
        "'%s' may not appear in a locality-spec because it is not "
        "definable"_err_en_US,
        std::move(*msg));
    return false;
  }
  return PassesSharedLocalityChecks(name, symbol);
}

Symbol &DeclarationVisitor::FindOrDeclareEnclosingEntity(
    const parser::Name &name) {
  Symbol *prev{FindSymbol(name)};
  if (!prev) {
    // Declare the name as an object in the enclosing scope so that
    // the name can't be repurposed there later as something else.
    prev = &MakeSymbol(InclusiveScope(), name.source, Attrs{});
    ConvertToObjectEntity(*prev);
    ApplyImplicitRules(*prev);
  }
  return *prev;
}

Symbol *DeclarationVisitor::DeclareLocalEntity(const parser::Name &name) {
  Symbol &prev{FindOrDeclareEnclosingEntity(name)};
  if (!PassesLocalityChecks(name, prev)) {
    return nullptr;
  }
  return &MakeHostAssocSymbol(name, prev);
}

Symbol *DeclarationVisitor::DeclareStatementEntity(
    const parser::DoVariable &doVar,
    const std::optional<parser::IntegerTypeSpec> &type) {
  const parser::Name &name{doVar.thing.thing};
  const DeclTypeSpec *declTypeSpec{nullptr};
  if (auto *prev{FindSymbol(name)}) {
    if (prev->owner() == currScope()) {
      SayAlreadyDeclared(name, *prev);
      return nullptr;
    }
    name.symbol = nullptr;
    declTypeSpec = prev->GetType();
  }
  Symbol &symbol{DeclareEntity<ObjectEntityDetails>(name, {})};
  if (!symbol.has<ObjectEntityDetails>()) {
    return nullptr; // error was reported in DeclareEntity
  }
  if (type) {
    declTypeSpec = ProcessTypeSpec(*type);
  }
  if (declTypeSpec) {
    // Subtlety: Don't let a "*length" specifier (if any is pending) affect the
    // declaration of this implied DO loop control variable.
    auto restorer{
        common::ScopedSet(charInfo_.length, std::optional<ParamValue>{})};
    SetType(name, *declTypeSpec);
  } else {
    ApplyImplicitRules(symbol);
  }
  Symbol *result{Resolve(name, &symbol)};
  AnalyzeExpr(context(), doVar); // enforce INTEGER type
  return result;
}

// Set the type of an entity or report an error.
void DeclarationVisitor::SetType(
    const parser::Name &name, const DeclTypeSpec &type) {
  CHECK(name.symbol);
  auto &symbol{*name.symbol};
  if (charInfo_.length) { // Declaration has "*length" (R723)
    auto length{std::move(*charInfo_.length)};
    charInfo_.length.reset();
    if (type.category() == DeclTypeSpec::Character) {
      auto kind{type.characterTypeSpec().kind()};
      // Recurse with correct type.
      SetType(name,
          currScope().MakeCharacterType(std::move(length), std::move(kind)));
      return;
    } else { // C753
      Say(name,
          "A length specifier cannot be used to declare the non-character entity '%s'"_err_en_US);
    }
  }
  auto *prevType{symbol.GetType()};
  if (!prevType) {
    symbol.SetType(type);
  } else if (symbol.has<UseDetails>()) {
    // error recovery case, redeclaration of use-associated name
  } else if (HadForwardRef(symbol)) {
    // error recovery after use of host-associated name
  } else if (!symbol.test(Symbol::Flag::Implicit)) {
    SayWithDecl(
        name, symbol, "The type of '%s' has already been declared"_err_en_US);
    context().SetError(symbol);
  } else if (type != *prevType) {
    SayWithDecl(name, symbol,
        "The type of '%s' has already been implicitly declared"_err_en_US);
    context().SetError(symbol);
  } else {
    symbol.set(Symbol::Flag::Implicit, false);
  }
}

std::optional<DerivedTypeSpec> DeclarationVisitor::ResolveDerivedType(
    const parser::Name &name) {
  Scope &outer{NonDerivedTypeScope()};
  Symbol *symbol{FindSymbol(outer, name)};
  Symbol *ultimate{symbol ? &symbol->GetUltimate() : nullptr};
  auto *generic{ultimate ? ultimate->detailsIf<GenericDetails>() : nullptr};
  if (generic) {
    if (Symbol * genDT{generic->derivedType()}) {
      symbol = genDT;
      generic = nullptr;
    }
  }
  if (!symbol || symbol->has<UnknownDetails>() ||
      (generic && &ultimate->owner() == &outer)) {
    if (allowForwardReferenceToDerivedType()) {
      if (!symbol) {
        symbol = &MakeSymbol(outer, name.source, Attrs{});
        Resolve(name, *symbol);
      } else if (generic) {
        // forward ref to type with later homonymous generic
        symbol = &outer.MakeSymbol(name.source, Attrs{}, UnknownDetails{});
        generic->set_derivedType(*symbol);
        name.symbol = symbol;
      }
      DerivedTypeDetails details;
      details.set_isForwardReferenced(true);
      symbol->set_details(std::move(details));
    } else { // C732
      Say(name, "Derived type '%s' not found"_err_en_US);
      return std::nullopt;
    }
  }
  if (CheckUseError(name)) {
    return std::nullopt;
  }
  symbol = &symbol->GetUltimate();
  if (symbol->has<DerivedTypeDetails>()) {
    return DerivedTypeSpec{name.source, *symbol};
  } else {
    Say(name, "'%s' is not a derived type"_err_en_US);
    return std::nullopt;
  }
}

std::optional<DerivedTypeSpec> DeclarationVisitor::ResolveExtendsType(
    const parser::Name &typeName, const parser::Name *extendsName) {
  if (!extendsName) {
    return std::nullopt;
  } else if (typeName.source == extendsName->source) {
    Say(extendsName->source,
        "Derived type '%s' cannot extend itself"_err_en_US);
    return std::nullopt;
  } else {
    return ResolveDerivedType(*extendsName);
  }
}

Symbol *DeclarationVisitor::NoteInterfaceName(const parser::Name &name) {
  // The symbol is checked later by CheckExplicitInterface() and
  // CheckBindings().  It can be a forward reference.
  if (!NameIsKnownOrIntrinsic(name)) {
    Symbol &symbol{MakeSymbol(InclusiveScope(), name.source, Attrs{})};
    Resolve(name, symbol);
  }
  return name.symbol;
}

void DeclarationVisitor::CheckExplicitInterface(const parser::Name &name) {
  if (const Symbol * symbol{name.symbol}) {
    const Symbol &ultimate{symbol->GetUltimate()};
    if (!context().HasError(*symbol) && !context().HasError(ultimate) &&
        !ultimate.HasExplicitInterface()) {
      Say(name,
          "'%s' must be an abstract interface or a procedure with "
          "an explicit interface"_err_en_US,
          symbol->name());
    }
  }
}

// Create a symbol for a type parameter, component, or procedure binding in
// the current derived type scope. Return false on error.
Symbol *DeclarationVisitor::MakeTypeSymbol(
    const parser::Name &name, Details &&details) {
  return Resolve(name, MakeTypeSymbol(name.source, std::move(details)));
}
Symbol *DeclarationVisitor::MakeTypeSymbol(
    const SourceName &name, Details &&details) {
  Scope &derivedType{currScope()};
  CHECK(derivedType.IsDerivedType());
  if (auto *symbol{FindInScope(derivedType, name)}) { // C742
    Say2(name,
        "Type parameter, component, or procedure binding '%s'"
        " already defined in this type"_err_en_US,
        *symbol, "Previous definition of '%s'"_en_US);
    return nullptr;
  } else {
    auto attrs{GetAttrs()};
    // Apply binding-private-stmt if present and this is a procedure binding
    if (derivedTypeInfo_.privateBindings &&
        !attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE}) &&
        std::holds_alternative<ProcBindingDetails>(details)) {
      attrs.set(Attr::PRIVATE);
    }
    Symbol &result{MakeSymbol(name, attrs, std::move(details))};
    if (result.has<TypeParamDetails>()) {
      derivedType.symbol()->get<DerivedTypeDetails>().add_paramDecl(result);
    }
    return &result;
  }
}

// Return true if it is ok to declare this component in the current scope.
// Otherwise, emit an error and return false.
bool DeclarationVisitor::OkToAddComponent(
    const parser::Name &name, const Symbol *extends) {
  for (const Scope *scope{&currScope()}; scope;) {
    CHECK(scope->IsDerivedType());
    if (auto *prev{FindInScope(*scope, name.source)}) {
      std::optional<parser::MessageFixedText> msg;
      if (context().HasError(*prev)) { // don't pile on
      } else if (extends) {
        msg = "Type cannot be extended as it has a component named"
              " '%s'"_err_en_US;
      } else if (CheckAccessibleComponent(currScope(), *prev)) {
        // inaccessible component -- redeclaration is ok
        msg = "Component '%s' is inaccessibly declared in or as a "
              "parent of this derived type"_warn_en_US;
      } else if (prev->test(Symbol::Flag::ParentComp)) {
        msg = "'%s' is a parent type of this type and so cannot be"
              " a component"_err_en_US;
      } else if (scope == &currScope()) {
        msg = "Component '%s' is already declared in this"
              " derived type"_err_en_US;
      } else {
        msg = "Component '%s' is already declared in a parent of this"
              " derived type"_err_en_US;
      }
      if (msg) {
        Say2(
            name, std::move(*msg), *prev, "Previous declaration of '%s'"_en_US);
        if (msg->severity() == parser::Severity::Error) {
          Resolve(name, *prev);
          return false;
        }
      }
    }
    if (scope == &currScope() && extends) {
      // The parent component has not yet been added to the scope.
      scope = extends->scope();
    } else {
      scope = scope->GetDerivedTypeParent();
    }
  }
  return true;
}

ParamValue DeclarationVisitor::GetParamValue(
    const parser::TypeParamValue &x, common::TypeParamAttr attr) {
  return common::visit(
      common::visitors{
          [=](const parser::ScalarIntExpr &x) { // C704
            return ParamValue{EvaluateIntExpr(x), attr};
          },
          [=](const parser::Star &) { return ParamValue::Assumed(attr); },
          [=](const parser::TypeParamValue::Deferred &) {
            return ParamValue::Deferred(attr);
          },
      },
      x.u);
}

// ConstructVisitor implementation

void ConstructVisitor::ResolveIndexName(
    const parser::ConcurrentControl &control) {
  const parser::Name &name{std::get<parser::Name>(control.t)};
  auto *prev{FindSymbol(name)};
  if (prev) {
    if (prev->owner().kind() == Scope::Kind::Forall ||
        prev->owner() == currScope()) {
      SayAlreadyDeclared(name, *prev);
      return;
    }
    name.symbol = nullptr;
  }
  auto &symbol{DeclareObjectEntity(name)};
  if (symbol.GetType()) {
    // type came from explicit type-spec
  } else if (!prev) {
    ApplyImplicitRules(symbol);
  } else {
    const Symbol &prevRoot{prev->GetUltimate()};
    // prev could be host- use- or construct-associated with another symbol
    if (!prevRoot.has<ObjectEntityDetails>() &&
        !prevRoot.has<AssocEntityDetails>()) {
      Say2(name, "Index name '%s' conflicts with existing identifier"_err_en_US,
          *prev, "Previous declaration of '%s'"_en_US);
      context().SetError(symbol);
      return;
    } else {
      if (const auto *type{prevRoot.GetType()}) {
        symbol.SetType(*type);
      }
      if (prevRoot.IsObjectArray()) {
        SayWithDecl(name, *prev, "Index variable '%s' is not scalar"_err_en_US);
        return;
      }
    }
  }
  EvaluateExpr(parser::Scalar{parser::Integer{common::Clone(name)}});
}

// We need to make sure that all of the index-names get declared before the
// expressions in the loop control are evaluated so that references to the
// index-names in the expressions are correctly detected.
bool ConstructVisitor::Pre(const parser::ConcurrentHeader &header) {
  BeginDeclTypeSpec();
  Walk(std::get<std::optional<parser::IntegerTypeSpec>>(header.t));
  const auto &controls{
      std::get<std::list<parser::ConcurrentControl>>(header.t)};
  for (const auto &control : controls) {
    ResolveIndexName(control);
  }
  Walk(controls);
  Walk(std::get<std::optional<parser::ScalarLogicalExpr>>(header.t));
  EndDeclTypeSpec();
  return false;
}

bool ConstructVisitor::Pre(const parser::LocalitySpec::Local &x) {
  for (auto &name : x.v) {
    if (auto *symbol{DeclareLocalEntity(name)}) {
      symbol->set(Symbol::Flag::LocalityLocal);
    }
  }
  return false;
}

bool ConstructVisitor::Pre(const parser::LocalitySpec::LocalInit &x) {
  for (auto &name : x.v) {
    if (auto *symbol{DeclareLocalEntity(name)}) {
      symbol->set(Symbol::Flag::LocalityLocalInit);
    }
  }
  return false;
}

bool ConstructVisitor::Pre(const parser::LocalitySpec::Shared &x) {
  for (const auto &name : x.v) {
    if (!FindSymbol(name)) {
      Say(name,
          "Variable '%s' with SHARED locality implicitly declared"_warn_en_US);
    }
    Symbol &prev{FindOrDeclareEnclosingEntity(name)};
    if (PassesSharedLocalityChecks(name, prev)) {
      MakeHostAssocSymbol(name, prev).set(Symbol::Flag::LocalityShared);
    }
  }
  return false;
}

bool ConstructVisitor::Pre(const parser::AcSpec &x) {
  ProcessTypeSpec(x.type);
  Walk(x.values);
  return false;
}

// Section 19.4, paragraph 5 says that each ac-do-variable has the scope of the
// enclosing ac-implied-do
bool ConstructVisitor::Pre(const parser::AcImpliedDo &x) {
  auto &values{std::get<std::list<parser::AcValue>>(x.t)};
  auto &control{std::get<parser::AcImpliedDoControl>(x.t)};
  auto &type{std::get<std::optional<parser::IntegerTypeSpec>>(control.t)};
  auto &bounds{std::get<parser::AcImpliedDoControl::Bounds>(control.t)};
  // F'2018 has the scope of the implied DO variable covering the entire
  // implied DO production (19.4(5)), which seems wrong in cases where the name
  // of the implied DO variable appears in one of the bound expressions. Thus
  // this extension, which shrinks the scope of the variable to exclude the
  // expressions in the bounds.
  auto restore{BeginCheckOnIndexUseInOwnBounds(bounds.name)};
  Walk(bounds.lower);
  Walk(bounds.upper);
  Walk(bounds.step);
  EndCheckOnIndexUseInOwnBounds(restore);
  PushScope(Scope::Kind::ImpliedDos, nullptr);
  DeclareStatementEntity(bounds.name, type);
  Walk(values);
  PopScope();
  return false;
}

bool ConstructVisitor::Pre(const parser::DataImpliedDo &x) {
  auto &objects{std::get<std::list<parser::DataIDoObject>>(x.t)};
  auto &type{std::get<std::optional<parser::IntegerTypeSpec>>(x.t)};
  auto &bounds{std::get<parser::DataImpliedDo::Bounds>(x.t)};
  // See comment in Pre(AcImpliedDo) above.
  auto restore{BeginCheckOnIndexUseInOwnBounds(bounds.name)};
  Walk(bounds.lower);
  Walk(bounds.upper);
  Walk(bounds.step);
  EndCheckOnIndexUseInOwnBounds(restore);
  bool pushScope{currScope().kind() != Scope::Kind::ImpliedDos};
  if (pushScope) {
    PushScope(Scope::Kind::ImpliedDos, nullptr);
  }
  DeclareStatementEntity(bounds.name, type);
  Walk(objects);
  if (pushScope) {
    PopScope();
  }
  return false;
}

// Sets InDataStmt flag on a variable (or misidentified function) in a DATA
// statement so that the predicate IsInitialized() will be true
// during semantic analysis before the symbol's initializer is constructed.
bool ConstructVisitor::Pre(const parser::DataIDoObject &x) {
  common::visit(
      common::visitors{
          [&](const parser::Scalar<Indirection<parser::Designator>> &y) {
            Walk(y.thing.value());
            const parser::Name &first{parser::GetFirstName(y.thing.value())};
            if (first.symbol) {
              first.symbol->set(Symbol::Flag::InDataStmt);
            }
          },
          [&](const Indirection<parser::DataImpliedDo> &y) { Walk(y.value()); },
      },
      x.u);
  return false;
}

bool ConstructVisitor::Pre(const parser::DataStmtObject &x) {
  // Subtle: DATA statements may appear in both the specification and
  // execution parts, but should be treated as if in the execution part
  // for purposes of implicit variable declaration vs. host association.
  // When a name first appears as an object in a DATA statement, it should
  // be implicitly declared locally as if it had been assigned.
  auto flagRestorer{common::ScopedSet(inSpecificationPart_, false)};
  common::visit(common::visitors{
                    [&](const Indirection<parser::Variable> &y) {
                      Walk(y.value());
                      const parser::Name &first{
                          parser::GetFirstName(y.value())};
                      if (first.symbol) {
                        first.symbol->set(Symbol::Flag::InDataStmt);
                      }
                    },
                    [&](const parser::DataImpliedDo &y) {
                      PushScope(Scope::Kind::ImpliedDos, nullptr);
                      Walk(y);
                      PopScope();
                    },
                },
      x.u);
  return false;
}

bool ConstructVisitor::Pre(const parser::DataStmtValue &x) {
  const auto &data{std::get<parser::DataStmtConstant>(x.t)};
  auto &mutableData{const_cast<parser::DataStmtConstant &>(data)};
  if (auto *elem{parser::Unwrap<parser::ArrayElement>(mutableData)}) {
    if (const auto *name{std::get_if<parser::Name>(&elem->base.u)}) {
      if (const Symbol * symbol{FindSymbol(*name)}) {
        const Symbol &ultimate{symbol->GetUltimate()};
        if (ultimate.has<DerivedTypeDetails>()) {
          mutableData.u = elem->ConvertToStructureConstructor(
              DerivedTypeSpec{name->source, ultimate});
        }
      }
    }
  }
  return true;
}

bool ConstructVisitor::Pre(const parser::DoConstruct &x) {
  if (x.IsDoConcurrent()) {
    PushScope(Scope::Kind::OtherConstruct, nullptr);
  }
  return true;
}
void ConstructVisitor::Post(const parser::DoConstruct &x) {
  if (x.IsDoConcurrent()) {
    PopScope();
  }
}

bool ConstructVisitor::Pre(const parser::ForallConstruct &) {
  PushScope(Scope::Kind::Forall, nullptr);
  return true;
}
void ConstructVisitor::Post(const parser::ForallConstruct &) { PopScope(); }
bool ConstructVisitor::Pre(const parser::ForallStmt &) {
  PushScope(Scope::Kind::Forall, nullptr);
  return true;
}
void ConstructVisitor::Post(const parser::ForallStmt &) { PopScope(); }

bool ConstructVisitor::Pre(const parser::BlockStmt &x) {
  CheckDef(x.v);
  PushScope(Scope::Kind::BlockConstruct, nullptr);
  return false;
}
bool ConstructVisitor::Pre(const parser::EndBlockStmt &x) {
  PopScope();
  CheckRef(x.v);
  return false;
}

void ConstructVisitor::Post(const parser::Selector &x) {
  GetCurrentAssociation().selector = ResolveSelector(x);
}

void ConstructVisitor::Post(const parser::AssociateStmt &x) {
  CheckDef(x.t);
  PushScope(Scope::Kind::OtherConstruct, nullptr);
  const auto assocCount{std::get<std::list<parser::Association>>(x.t).size()};
  for (auto nthLastAssoc{assocCount}; nthLastAssoc > 0; --nthLastAssoc) {
    SetCurrentAssociation(nthLastAssoc);
    if (auto *symbol{MakeAssocEntity()}) {
      if (ExtractCoarrayRef(GetCurrentAssociation().selector.expr)) { // C1103
        Say("Selector must not be a coindexed object"_err_en_US);
      }
      SetTypeFromAssociation(*symbol);
      SetAttrsFromAssociation(*symbol);
    }
  }
  PopAssociation(assocCount);
}

void ConstructVisitor::Post(const parser::EndAssociateStmt &x) {
  PopScope();
  CheckRef(x.v);
}

bool ConstructVisitor::Pre(const parser::Association &x) {
  PushAssociation();
  const auto &name{std::get<parser::Name>(x.t)};
  GetCurrentAssociation().name = &name;
  return true;
}

bool ConstructVisitor::Pre(const parser::ChangeTeamStmt &x) {
  CheckDef(x.t);
  PushScope(Scope::Kind::OtherConstruct, nullptr);
  PushAssociation();
  return true;
}

void ConstructVisitor::Post(const parser::CoarrayAssociation &x) {
  const auto &decl{std::get<parser::CodimensionDecl>(x.t)};
  const auto &name{std::get<parser::Name>(decl.t)};
  if (auto *symbol{FindInScope(name)}) {
    const auto &selector{std::get<parser::Selector>(x.t)};
    if (auto sel{ResolveSelector(selector)}) {
      const Symbol *whole{UnwrapWholeSymbolDataRef(sel.expr)};
      if (!whole || whole->Corank() == 0) {
        Say(sel.source, // C1116
            "Selector in coarray association must name a coarray"_err_en_US);
      } else if (auto dynType{sel.expr->GetType()}) {
        if (!symbol->GetType()) {
          symbol->SetType(ToDeclTypeSpec(std::move(*dynType)));
        }
      }
    }
  }
}

void ConstructVisitor::Post(const parser::EndChangeTeamStmt &x) {
  PopAssociation();
  PopScope();
  CheckRef(x.t);
}

bool ConstructVisitor::Pre(const parser::SelectTypeConstruct &) {
  PushAssociation();
  return true;
}

void ConstructVisitor::Post(const parser::SelectTypeConstruct &) {
  PopAssociation();
}

void ConstructVisitor::Post(const parser::SelectTypeStmt &x) {
  auto &association{GetCurrentAssociation()};
  if (const std::optional<parser::Name> &name{std::get<1>(x.t)}) {
    // This isn't a name in the current scope, it is in each TypeGuardStmt
    MakePlaceholder(*name, MiscDetails::Kind::SelectTypeAssociateName);
    association.name = &*name;
    auto exprType{association.selector.expr->GetType()};
    if (ExtractCoarrayRef(association.selector.expr)) { // C1103
      Say("Selector must not be a coindexed object"_err_en_US);
    }
    if (exprType && !exprType->IsPolymorphic()) { // C1159
      Say(association.selector.source,
          "Selector '%s' in SELECT TYPE statement must be "
          "polymorphic"_err_en_US);
    }
  } else {
    if (const Symbol *
        whole{UnwrapWholeSymbolDataRef(association.selector.expr)}) {
      ConvertToObjectEntity(const_cast<Symbol &>(*whole));
      if (!IsVariableName(*whole)) {
        Say(association.selector.source, // C901
            "Selector is not a variable"_err_en_US);
        association = {};
      }
      if (const DeclTypeSpec * type{whole->GetType()}) {
        if (!type->IsPolymorphic()) { // C1159
          Say(association.selector.source,
              "Selector '%s' in SELECT TYPE statement must be "
              "polymorphic"_err_en_US);
        }
      }
    } else {
      Say(association.selector.source, // C1157
          "Selector is not a named variable: 'associate-name =>' is required"_err_en_US);
      association = {};
    }
  }
}

void ConstructVisitor::Post(const parser::SelectRankStmt &x) {
  auto &association{GetCurrentAssociation()};
  if (const std::optional<parser::Name> &name{std::get<1>(x.t)}) {
    // This isn't a name in the current scope, it is in each SelectRankCaseStmt
    MakePlaceholder(*name, MiscDetails::Kind::SelectRankAssociateName);
    association.name = &*name;
  }
}

bool ConstructVisitor::Pre(const parser::SelectTypeConstruct::TypeCase &) {
  PushScope(Scope::Kind::OtherConstruct, nullptr);
  return true;
}
void ConstructVisitor::Post(const parser::SelectTypeConstruct::TypeCase &) {
  PopScope();
}

bool ConstructVisitor::Pre(const parser::SelectRankConstruct::RankCase &) {
  PushScope(Scope::Kind::OtherConstruct, nullptr);
  return true;
}
void ConstructVisitor::Post(const parser::SelectRankConstruct::RankCase &) {
  PopScope();
}

void ConstructVisitor::Post(const parser::TypeGuardStmt::Guard &x) {
  if (auto *symbol{MakeAssocEntity()}) {
    if (std::holds_alternative<parser::Default>(x.u)) {
      SetTypeFromAssociation(*symbol);
    } else if (const auto *type{GetDeclTypeSpec()}) {
      symbol->SetType(*type);
    }
    SetAttrsFromAssociation(*symbol);
  }
}

void ConstructVisitor::Post(const parser::SelectRankCaseStmt::Rank &x) {
  if (auto *symbol{MakeAssocEntity()}) {
    SetTypeFromAssociation(*symbol);
    SetAttrsFromAssociation(*symbol);
    if (const auto *init{std::get_if<parser::ScalarIntConstantExpr>(&x.u)}) {
      if (auto val{EvaluateInt64(context(), *init)}) {
        auto &details{symbol->get<AssocEntityDetails>()};
        details.set_rank(*val);
      }
    }
  }
}

bool ConstructVisitor::Pre(const parser::SelectRankConstruct &) {
  PushAssociation();
  return true;
}

void ConstructVisitor::Post(const parser::SelectRankConstruct &) {
  PopAssociation();
}

bool ConstructVisitor::CheckDef(const std::optional<parser::Name> &x) {
  if (x) {
    MakeSymbol(*x, MiscDetails{MiscDetails::Kind::ConstructName});
  }
  return true;
}

void ConstructVisitor::CheckRef(const std::optional<parser::Name> &x) {
  if (x) {
    // Just add an occurrence of this name; checking is done in ValidateLabels
    FindSymbol(*x);
  }
}

// Make a symbol for the associating entity of the current association.
Symbol *ConstructVisitor::MakeAssocEntity() {
  Symbol *symbol{nullptr};
  auto &association{GetCurrentAssociation()};
  if (association.name) {
    symbol = &MakeSymbol(*association.name, UnknownDetails{});
    if (symbol->has<AssocEntityDetails>() && symbol->owner() == currScope()) {
      Say(*association.name, // C1102
          "The associate name '%s' is already used in this associate statement"_err_en_US);
      return nullptr;
    }
  } else if (const Symbol *
      whole{UnwrapWholeSymbolDataRef(association.selector.expr)}) {
    symbol = &MakeSymbol(whole->name());
  } else {
    return nullptr;
  }
  if (auto &expr{association.selector.expr}) {
    symbol->set_details(AssocEntityDetails{common::Clone(*expr)});
  } else {
    symbol->set_details(AssocEntityDetails{});
  }
  return symbol;
}

// Set the type of symbol based on the current association selector.
void ConstructVisitor::SetTypeFromAssociation(Symbol &symbol) {
  auto &details{symbol.get<AssocEntityDetails>()};
  const MaybeExpr *pexpr{&details.expr()};
  if (!*pexpr) {
    pexpr = &GetCurrentAssociation().selector.expr;
  }
  if (*pexpr) {
    const SomeExpr &expr{**pexpr};
    if (std::optional<evaluate::DynamicType> type{expr.GetType()}) {
      if (const auto *charExpr{
              evaluate::UnwrapExpr<evaluate::Expr<evaluate::SomeCharacter>>(
                  expr)}) {
        symbol.SetType(ToDeclTypeSpec(std::move(*type),
            FoldExpr(common::visit(
                [](const auto &kindChar) { return kindChar.LEN(); },
                charExpr->u))));
      } else {
        symbol.SetType(ToDeclTypeSpec(std::move(*type)));
      }
    } else {
      // BOZ literals, procedure designators, &c. are not acceptable
      Say(symbol.name(), "Associate name '%s' must have a type"_err_en_US);
    }
  }
}

// If current selector is a variable, set some of its attributes on symbol.
void ConstructVisitor::SetAttrsFromAssociation(Symbol &symbol) {
  Attrs attrs{evaluate::GetAttrs(GetCurrentAssociation().selector.expr)};
  symbol.attrs() |= attrs &
      Attrs{Attr::TARGET, Attr::ASYNCHRONOUS, Attr::VOLATILE, Attr::CONTIGUOUS};
  if (attrs.test(Attr::POINTER)) {
    symbol.attrs().set(Attr::TARGET);
  }
}

ConstructVisitor::Selector ConstructVisitor::ResolveSelector(
    const parser::Selector &x) {
  return common::visit(common::visitors{
                           [&](const parser::Expr &expr) {
                             return Selector{expr.source, EvaluateExpr(x)};
                           },
                           [&](const parser::Variable &var) {
                             return Selector{var.GetSource(), EvaluateExpr(x)};
                           },
                       },
      x.u);
}

// Set the current association to the nth to the last association on the
// association stack.  The top of the stack is at n = 1.  This allows access
// to the interior of a list of associations at the top of the stack.
void ConstructVisitor::SetCurrentAssociation(std::size_t n) {
  CHECK(n > 0 && n <= associationStack_.size());
  currentAssociation_ = &associationStack_[associationStack_.size() - n];
}

ConstructVisitor::Association &ConstructVisitor::GetCurrentAssociation() {
  CHECK(currentAssociation_);
  return *currentAssociation_;
}

void ConstructVisitor::PushAssociation() {
  associationStack_.emplace_back(Association{});
  currentAssociation_ = &associationStack_.back();
}

void ConstructVisitor::PopAssociation(std::size_t count) {
  CHECK(count > 0 && count <= associationStack_.size());
  associationStack_.resize(associationStack_.size() - count);
  currentAssociation_ =
      associationStack_.empty() ? nullptr : &associationStack_.back();
}

const DeclTypeSpec &ConstructVisitor::ToDeclTypeSpec(
    evaluate::DynamicType &&type) {
  switch (type.category()) {
    SWITCH_COVERS_ALL_CASES
  case common::TypeCategory::Integer:
  case common::TypeCategory::Real:
  case common::TypeCategory::Complex:
    return context().MakeNumericType(type.category(), type.kind());
  case common::TypeCategory::Logical:
    return context().MakeLogicalType(type.kind());
  case common::TypeCategory::Derived:
    if (type.IsAssumedType()) {
      return currScope().MakeTypeStarType();
    } else if (type.IsUnlimitedPolymorphic()) {
      return currScope().MakeClassStarType();
    } else {
      return currScope().MakeDerivedType(
          type.IsPolymorphic() ? DeclTypeSpec::ClassDerived
                               : DeclTypeSpec::TypeDerived,
          common::Clone(type.GetDerivedTypeSpec())

      );
    }
  case common::TypeCategory::Character:
    CRASH_NO_CASE;
  }
}

const DeclTypeSpec &ConstructVisitor::ToDeclTypeSpec(
    evaluate::DynamicType &&type, MaybeSubscriptIntExpr &&length) {
  CHECK(type.category() == common::TypeCategory::Character);
  if (length) {
    return currScope().MakeCharacterType(
        ParamValue{SomeIntExpr{*std::move(length)}, common::TypeParamAttr::Len},
        KindExpr{type.kind()});
  } else {
    return currScope().MakeCharacterType(
        ParamValue::Deferred(common::TypeParamAttr::Len),
        KindExpr{type.kind()});
  }
}

// ResolveNamesVisitor implementation

bool ResolveNamesVisitor::Pre(const parser::FunctionReference &x) {
  HandleCall(Symbol::Flag::Function, x.v);
  return false;
}
bool ResolveNamesVisitor::Pre(const parser::CallStmt &x) {
  HandleCall(Symbol::Flag::Subroutine, x.v);
  return false;
}

bool ResolveNamesVisitor::Pre(const parser::ImportStmt &x) {
  auto &scope{currScope()};
  // Check C896 and C899: where IMPORT statements are allowed
  switch (scope.kind()) {
  case Scope::Kind::Module:
    if (scope.IsModule()) {
      Say("IMPORT is not allowed in a module scoping unit"_err_en_US);
      return false;
    } else if (x.kind == common::ImportKind::None) {
      Say("IMPORT,NONE is not allowed in a submodule scoping unit"_err_en_US);
      return false;
    }
    break;
  case Scope::Kind::MainProgram:
    Say("IMPORT is not allowed in a main program scoping unit"_err_en_US);
    return false;
  case Scope::Kind::Subprogram:
    if (scope.parent().IsGlobal()) {
      Say("IMPORT is not allowed in an external subprogram scoping unit"_err_en_US);
      return false;
    }
    break;
  case Scope::Kind::BlockData: // C1415 (in part)
    Say("IMPORT is not allowed in a BLOCK DATA subprogram"_err_en_US);
    return false;
  default:;
  }
  if (auto error{scope.SetImportKind(x.kind)}) {
    Say(std::move(*error));
  }
  for (auto &name : x.names) {
    if (FindSymbol(scope.parent(), name)) {
      scope.add_importName(name.source);
    } else {
      Say(name, "'%s' not found in host scope"_err_en_US);
    }
  }
  prevImportStmt_ = currStmtSource();
  return false;
}

const parser::Name *DeclarationVisitor::ResolveStructureComponent(
    const parser::StructureComponent &x) {
  return FindComponent(ResolveDataRef(x.base), x.component);
}

const parser::Name *DeclarationVisitor::ResolveDesignator(
    const parser::Designator &x) {
  return common::visit(
      common::visitors{
          [&](const parser::DataRef &x) { return ResolveDataRef(x); },
          [&](const parser::Substring &x) {
            Walk(std::get<parser::SubstringRange>(x.t).t);
            return ResolveDataRef(std::get<parser::DataRef>(x.t));
          },
      },
      x.u);
}

const parser::Name *DeclarationVisitor::ResolveDataRef(
    const parser::DataRef &x) {
  return common::visit(
      common::visitors{
          [=](const parser::Name &y) { return ResolveName(y); },
          [=](const Indirection<parser::StructureComponent> &y) {
            return ResolveStructureComponent(y.value());
          },
          [&](const Indirection<parser::ArrayElement> &y) {
            Walk(y.value().subscripts);
            const parser::Name *name{ResolveDataRef(y.value().base)};
            if (name && name->symbol) {
              if (!IsProcedure(*name->symbol)) {
                ConvertToObjectEntity(*name->symbol);
              } else if (!context().HasError(*name->symbol)) {
                SayWithDecl(*name, *name->symbol,
                    "Cannot reference function '%s' as data"_err_en_US);
              }
            }
            return name;
          },
          [&](const Indirection<parser::CoindexedNamedObject> &y) {
            Walk(y.value().imageSelector);
            return ResolveDataRef(y.value().base);
          },
      },
      x.u);
}

// If implicit types are allowed, ensure name is in the symbol table.
// Otherwise, report an error if it hasn't been declared.
const parser::Name *DeclarationVisitor::ResolveName(const parser::Name &name) {
  FindSymbol(name);
  if (CheckForHostAssociatedImplicit(name)) {
    NotePossibleBadForwardRef(name);
    return &name;
  }
  if (Symbol * symbol{name.symbol}) {
    if (CheckUseError(name)) {
      return nullptr; // reported an error
    }
    NotePossibleBadForwardRef(name);
    symbol->set(Symbol::Flag::ImplicitOrError, false);
    if (IsUplevelReference(*symbol)) {
      MakeHostAssocSymbol(name, *symbol);
    } else if (IsDummy(*symbol) ||
        (!symbol->GetType() && FindCommonBlockContaining(*symbol))) {
      ConvertToObjectEntity(*symbol);
      ApplyImplicitRules(*symbol);
    }
    if (checkIndexUseInOwnBounds_ &&
        *checkIndexUseInOwnBounds_ == name.source && !InModuleFile()) {
      Say(name,
          "Implied DO index '%s' uses an object of the same name in its bounds expressions"_port_en_US,
          name.source);
    }
    return &name;
  }
  if (isImplicitNoneType()) {
    Say(name, "No explicit type declared for '%s'"_err_en_US);
    return nullptr;
  }
  // Create the symbol then ensure it is accessible
  if (checkIndexUseInOwnBounds_ && *checkIndexUseInOwnBounds_ == name.source) {
    Say(name,
        "Implied DO index '%s' uses itself in its own bounds expressions"_err_en_US,
        name.source);
  }
  MakeSymbol(InclusiveScope(), name.source, Attrs{});
  auto *symbol{FindSymbol(name)};
  if (!symbol) {
    Say(name,
        "'%s' from host scoping unit is not accessible due to IMPORT"_err_en_US);
    return nullptr;
  }
  ConvertToObjectEntity(*symbol);
  ApplyImplicitRules(*symbol);
  NotePossibleBadForwardRef(name);
  return &name;
}

// A specification expression may refer to a symbol in the host procedure that
// is implicitly typed. Because specification parts are processed before
// execution parts, this may be the first time we see the symbol. It can't be a
// local in the current scope (because it's in a specification expression) so
// either it is implicitly declared in the host procedure or it is an error.
// We create a symbol in the host assuming it is the former; if that proves to
// be wrong we report an error later in CheckDeclarations().
bool DeclarationVisitor::CheckForHostAssociatedImplicit(
    const parser::Name &name) {
  if (!inSpecificationPart_) {
    return false;
  }
  if (name.symbol) {
    ApplyImplicitRules(*name.symbol, true);
  }
  Symbol *hostSymbol;
  Scope *host{GetHostProcedure()};
  if (!host || isImplicitNoneType(*host)) {
    return false;
  }
  if (!name.symbol) {
    hostSymbol = &MakeSymbol(*host, name.source, Attrs{});
    ConvertToObjectEntity(*hostSymbol);
    ApplyImplicitRules(*hostSymbol);
    hostSymbol->set(Symbol::Flag::ImplicitOrError);
  } else if (name.symbol->test(Symbol::Flag::ImplicitOrError)) {
    hostSymbol = name.symbol;
  } else {
    return false;
  }
  Symbol &symbol{MakeHostAssocSymbol(name, *hostSymbol)};
  if (isImplicitNoneType()) {
    symbol.get<HostAssocDetails>().implicitOrExplicitTypeError = true;
  } else {
    symbol.get<HostAssocDetails>().implicitOrSpecExprError = true;
  }
  return true;
}

bool DeclarationVisitor::IsUplevelReference(const Symbol &symbol) {
  const Scope &symbolUnit{GetProgramUnitContaining(symbol)};
  if (symbolUnit == GetProgramUnitContaining(currScope())) {
    return false;
  } else {
    Scope::Kind kind{symbolUnit.kind()};
    return kind == Scope::Kind::Subprogram || kind == Scope::Kind::MainProgram;
  }
}

// base is a part-ref of a derived type; find the named component in its type.
// Also handles intrinsic type parameter inquiries (%kind, %len) and
// COMPLEX component references (%re, %im).
const parser::Name *DeclarationVisitor::FindComponent(
    const parser::Name *base, const parser::Name &component) {
  if (!base || !base->symbol) {
    return nullptr;
  }
  if (auto *misc{base->symbol->detailsIf<MiscDetails>()}) {
    if (component.source == "kind") {
      if (misc->kind() == MiscDetails::Kind::ComplexPartRe ||
          misc->kind() == MiscDetails::Kind::ComplexPartIm ||
          misc->kind() == MiscDetails::Kind::KindParamInquiry ||
          misc->kind() == MiscDetails::Kind::LenParamInquiry) {
        // x%{re,im,kind,len}%kind
        MakePlaceholder(component, MiscDetails::Kind::KindParamInquiry);
        return &component;
      }
    }
  }
  auto &symbol{base->symbol->GetUltimate()};
  if (!symbol.has<AssocEntityDetails>() && !ConvertToObjectEntity(symbol)) {
    SayWithDecl(*base, symbol,
        "'%s' is an invalid base for a component reference"_err_en_US);
    return nullptr;
  }
  auto *type{symbol.GetType()};
  if (!type) {
    return nullptr; // should have already reported error
  }
  if (const IntrinsicTypeSpec * intrinsic{type->AsIntrinsic()}) {
    auto category{intrinsic->category()};
    MiscDetails::Kind miscKind{MiscDetails::Kind::None};
    if (component.source == "kind") {
      miscKind = MiscDetails::Kind::KindParamInquiry;
    } else if (category == TypeCategory::Character) {
      if (component.source == "len") {
        miscKind = MiscDetails::Kind::LenParamInquiry;
      }
    } else if (category == TypeCategory::Complex) {
      if (component.source == "re") {
        miscKind = MiscDetails::Kind::ComplexPartRe;
      } else if (component.source == "im") {
        miscKind = MiscDetails::Kind::ComplexPartIm;
      }
    }
    if (miscKind != MiscDetails::Kind::None) {
      MakePlaceholder(component, miscKind);
      return &component;
    }
  } else if (DerivedTypeSpec * derived{type->AsDerived()}) {
    derived->Instantiate(currScope()); // in case of forward referenced type
    if (const Scope * scope{derived->scope()}) {
      if (Resolve(component, scope->FindComponent(component.source))) {
        if (auto msg{
                CheckAccessibleComponent(currScope(), *component.symbol)}) {
          context().Say(component.source, *msg);
        }
        return &component;
      } else {
        SayDerivedType(component.source,
            "Component '%s' not found in derived type '%s'"_err_en_US, *scope);
      }
    }
    return nullptr;
  }
  if (symbol.test(Symbol::Flag::Implicit)) {
    Say(*base,
        "'%s' is not an object of derived type; it is implicitly typed"_err_en_US);
  } else {
    SayWithDecl(
        *base, symbol, "'%s' is not an object of derived type"_err_en_US);
  }
  return nullptr;
}

void DeclarationVisitor::Initialization(const parser::Name &name,
    const parser::Initialization &init, bool inComponentDecl) {
  // Traversal of the initializer was deferred to here so that the
  // symbol being declared can be available for use in the expression, e.g.:
  //   real, parameter :: x = tiny(x)
  if (!name.symbol) {
    return;
  }
  Symbol &ultimate{name.symbol->GetUltimate()};
  if (IsAllocatable(ultimate)) {
    Say(name, "Allocatable object '%s' cannot be initialized"_err_en_US);
    return;
  }
  // TODO: check C762 - all bounds and type parameters of component
  // are colons or constant expressions if component is initialized
  common::visit(
      common::visitors{
          [&](const parser::ConstantExpr &expr) {
            NonPointerInitialization(name, expr);
          },
          [&](const parser::NullInit &null) { // => NULL()
            Walk(null);
            if (auto nullInit{EvaluateExpr(null)}) {
              if (!evaluate::IsNullPointer(*nullInit)) {
                Say(name,
                    "Pointer initializer must be intrinsic NULL()"_err_en_US); // C813
              } else if (IsPointer(ultimate)) {
                if (auto *object{ultimate.detailsIf<ObjectEntityDetails>()}) {
                  object->set_init(std::move(*nullInit));
                } else if (auto *procPtr{
                               ultimate.detailsIf<ProcEntityDetails>()}) {
                  procPtr->set_init(nullptr);
                }
              } else {
                Say(name,
                    "Non-pointer component '%s' initialized with null pointer"_err_en_US);
              }
            }
          },
          [&](const parser::InitialDataTarget &) {
            // Defer analysis to the end of the specification part
            // so that forward references and attribute checks like SAVE
            // work better.
            ultimate.set(Symbol::Flag::InDataStmt);
          },
          [&](const std::list<Indirection<parser::DataStmtValue>> &values) {
            // Handled later in data-to-inits conversion
            ultimate.set(Symbol::Flag::InDataStmt);
            Walk(values);
          },
      },
      init.u);
}

void DeclarationVisitor::PointerInitialization(
    const parser::Name &name, const parser::InitialDataTarget &target) {
  if (name.symbol) {
    Symbol &ultimate{name.symbol->GetUltimate()};
    if (!context().HasError(ultimate)) {
      if (IsPointer(ultimate)) {
        if (auto *details{ultimate.detailsIf<ObjectEntityDetails>()}) {
          CHECK(!details->init());
          Walk(target);
          if (MaybeExpr expr{EvaluateExpr(target)}) {
            // Validation is done in declaration checking.
            details->set_init(std::move(*expr));
          }
        }
      } else {
        Say(name,
            "'%s' is not a pointer but is initialized like one"_err_en_US);
        context().SetError(ultimate);
      }
    }
  }
}
void DeclarationVisitor::PointerInitialization(
    const parser::Name &name, const parser::ProcPointerInit &target) {
  if (name.symbol) {
    Symbol &ultimate{name.symbol->GetUltimate()};
    if (!context().HasError(ultimate)) {
      if (IsProcedurePointer(ultimate)) {
        auto &details{ultimate.get<ProcEntityDetails>()};
        CHECK(!details.init());
        Walk(target);
        if (const auto *targetName{std::get_if<parser::Name>(&target.u)}) {
          if (!CheckUseError(*targetName) && targetName->symbol) {
            // Validation is done in declaration checking.
            details.set_init(*targetName->symbol);
          }
        } else {
          details.set_init(nullptr); // explicit NULL()
        }
      } else {
        Say(name,
            "'%s' is not a procedure pointer but is initialized "
            "like one"_err_en_US);
        context().SetError(ultimate);
      }
    }
  }
}

void DeclarationVisitor::NonPointerInitialization(
    const parser::Name &name, const parser::ConstantExpr &expr) {
  if (name.symbol) {
    Symbol &ultimate{name.symbol->GetUltimate()};
    if (!context().HasError(ultimate) && !context().HasError(name.symbol)) {
      if (IsPointer(ultimate)) {
        Say(name,
            "'%s' is a pointer but is not initialized like one"_err_en_US);
      } else if (auto *details{ultimate.detailsIf<ObjectEntityDetails>()}) {
        CHECK(!details->init());
        Walk(expr);
        if (ultimate.owner().IsParameterizedDerivedType()) {
          // Save the expression for per-instantiation analysis.
          details->set_unanalyzedPDTComponentInit(&expr.thing.value());
        } else {
          if (MaybeExpr folded{EvaluateNonPointerInitializer(
                  ultimate, expr, expr.thing.value().source)}) {
            details->set_init(std::move(*folded));
          }
        }
      }
    }
  }
}

void ResolveNamesVisitor::HandleCall(
    Symbol::Flag procFlag, const parser::Call &call) {
  common::visit(
      common::visitors{
          [&](const parser::Name &x) { HandleProcedureName(procFlag, x); },
          [&](const parser::ProcComponentRef &x) { Walk(x); },
      },
      std::get<parser::ProcedureDesignator>(call.t).u);
  Walk(std::get<std::list<parser::ActualArgSpec>>(call.t));
}

void ResolveNamesVisitor::HandleProcedureName(
    Symbol::Flag flag, const parser::Name &name) {
  CHECK(flag == Symbol::Flag::Function || flag == Symbol::Flag::Subroutine);
  auto *symbol{FindSymbol(NonDerivedTypeScope(), name)};
  if (!symbol) {
    if (IsIntrinsic(name.source, flag)) {
      symbol =
          &MakeSymbol(InclusiveScope(), name.source, Attrs{Attr::INTRINSIC});
    } else {
      symbol = &MakeSymbol(context().globalScope(), name.source, Attrs{});
    }
    Resolve(name, *symbol);
    if (!symbol->attrs().test(Attr::INTRINSIC)) {
      if (CheckImplicitNoneExternal(name.source, *symbol)) {
        MakeExternal(*symbol);
      }
    }
    ConvertToProcEntity(*symbol);
    SetProcFlag(name, *symbol, flag);
  } else if (CheckUseError(name)) {
    // error was reported
  } else {
    auto &nonUltimateSymbol{*symbol};
    symbol = &Resolve(name, symbol)->GetUltimate();
    bool convertedToProcEntity{ConvertToProcEntity(*symbol)};
    if (convertedToProcEntity && !symbol->attrs().test(Attr::EXTERNAL) &&
        IsIntrinsic(symbol->name(), flag) && !IsDummy(*symbol)) {
      AcquireIntrinsicProcedureFlags(*symbol);
    }
    if (!SetProcFlag(name, *symbol, flag)) {
      return; // reported error
    }
    if (!symbol->has<GenericDetails>()) {
      CheckImplicitNoneExternal(name.source, *symbol);
    }
    if (symbol->has<SubprogramDetails>() &&
        symbol->attrs().test(Attr::ABSTRACT)) {
      Say(name, "Abstract interface '%s' may not be called"_err_en_US);
    } else if (IsProcedure(*symbol) || symbol->has<DerivedTypeDetails>() ||
        symbol->has<AssocEntityDetails>()) {
      // Symbols with DerivedTypeDetails and AssocEntityDetails are accepted
      // here as procedure-designators because this means the related
      // FunctionReference are mis-parsed structure constructors or array
      // references that will be fixed later when analyzing expressions.
    } else if (symbol->has<ObjectEntityDetails>()) {
      // Symbols with ObjectEntityDetails are also accepted because this can be
      // a mis-parsed array references that will be fixed later. Ensure that if
      // this is a symbol from a host procedure, a symbol with HostAssocDetails
      // is created for the current scope.
      // Operate on non ultimate symbol so that HostAssocDetails are also
      // created for symbols used associated in the host procedure.
      if (IsUplevelReference(nonUltimateSymbol)) {
        MakeHostAssocSymbol(name, nonUltimateSymbol);
      }
    } else if (symbol->test(Symbol::Flag::Implicit)) {
      Say(name,
          "Use of '%s' as a procedure conflicts with its implicit definition"_err_en_US);
    } else {
      SayWithDecl(name, *symbol,
          "Use of '%s' as a procedure conflicts with its declaration"_err_en_US);
    }
  }
}

bool ResolveNamesVisitor::CheckImplicitNoneExternal(
    const SourceName &name, const Symbol &symbol) {
  if (isImplicitNoneExternal() && !symbol.attrs().test(Attr::EXTERNAL) &&
      !symbol.attrs().test(Attr::INTRINSIC) && !symbol.HasExplicitInterface()) {
    Say(name,
        "'%s' is an external procedure without the EXTERNAL"
        " attribute in a scope with IMPLICIT NONE(EXTERNAL)"_err_en_US);
    return false;
  }
  return true;
}

// Variant of HandleProcedureName() for use while skimming the executable
// part of a subprogram to catch calls to dummy procedures that are part
// of the subprogram's interface, and to mark as procedures any symbols
// that might otherwise have been miscategorized as objects.
void ResolveNamesVisitor::NoteExecutablePartCall(
    Symbol::Flag flag, const parser::Call &call) {
  auto &designator{std::get<parser::ProcedureDesignator>(call.t)};
  if (const auto *name{std::get_if<parser::Name>(&designator.u)}) {
    // Subtlety: The symbol pointers in the parse tree are not set, because
    // they might end up resolving elsewhere (e.g., construct entities in
    // SELECT TYPE).
    if (Symbol * symbol{currScope().FindSymbol(name->source)}) {
      Symbol::Flag other{flag == Symbol::Flag::Subroutine
              ? Symbol::Flag::Function
              : Symbol::Flag::Subroutine};
      if (!symbol->test(other)) {
        ConvertToProcEntity(*symbol);
        if (symbol->has<ProcEntityDetails>()) {
          symbol->set(flag);
          if (IsDummy(*symbol)) {
            symbol->attrs().set(Attr::EXTERNAL);
          }
          ApplyImplicitRules(*symbol);
        }
      }
    }
  }
}

static bool IsLocallyImplicitGlobalSymbol(
    const Symbol &symbol, const parser::Name &localName) {
  return symbol.owner().IsGlobal() &&
      (!symbol.scope() ||
          !symbol.scope()->sourceRange().Contains(localName.source));
}

static bool TypesMismatchIfNonNull(
    const DeclTypeSpec *type1, const DeclTypeSpec *type2) {
  return type1 && type2 && *type1 != *type2;
}

// Check and set the Function or Subroutine flag on symbol; false on error.
bool ResolveNamesVisitor::SetProcFlag(
    const parser::Name &name, Symbol &symbol, Symbol::Flag flag) {
  if (symbol.test(Symbol::Flag::Function) && flag == Symbol::Flag::Subroutine) {
    SayWithDecl(
        name, symbol, "Cannot call function '%s' like a subroutine"_err_en_US);
    return false;
  } else if (symbol.test(Symbol::Flag::Subroutine) &&
      flag == Symbol::Flag::Function) {
    SayWithDecl(
        name, symbol, "Cannot call subroutine '%s' like a function"_err_en_US);
    return false;
  } else if (flag == Symbol::Flag::Function &&
      IsLocallyImplicitGlobalSymbol(symbol, name) &&
      TypesMismatchIfNonNull(symbol.GetType(), GetImplicitType(symbol))) {
    SayWithDecl(name, symbol,
        "Implicit declaration of function '%s' has a different result type than in previous declaration"_err_en_US);
    return false;
  } else if (symbol.has<ProcEntityDetails>()) {
    symbol.set(flag); // in case it hasn't been set yet
    if (flag == Symbol::Flag::Function) {
      ApplyImplicitRules(symbol);
    }
    if (symbol.attrs().test(Attr::INTRINSIC)) {
      AcquireIntrinsicProcedureFlags(symbol);
    }
  } else if (symbol.GetType() && flag == Symbol::Flag::Subroutine) {
    SayWithDecl(
        name, symbol, "Cannot call function '%s' like a subroutine"_err_en_US);
  } else if (symbol.attrs().test(Attr::INTRINSIC)) {
    AcquireIntrinsicProcedureFlags(symbol);
  }
  return true;
}

bool ModuleVisitor::Pre(const parser::AccessStmt &x) {
  Attr accessAttr{AccessSpecToAttr(std::get<parser::AccessSpec>(x.t))};
  if (!currScope().IsModule()) { // C869
    Say(currStmtSource().value(),
        "%s statement may only appear in the specification part of a module"_err_en_US,
        EnumToString(accessAttr));
    return false;
  }
  const auto &accessIds{std::get<std::list<parser::AccessId>>(x.t)};
  if (accessIds.empty()) {
    if (prevAccessStmt_) { // C869
      Say("The default accessibility of this module has already been declared"_err_en_US)
          .Attach(*prevAccessStmt_, "Previous declaration"_en_US);
    }
    prevAccessStmt_ = currStmtSource();
    defaultAccess_ = accessAttr;
  } else {
    for (const auto &accessId : accessIds) {
      common::visit(
          common::visitors{
              [=](const parser::Name &y) {
                Resolve(y, SetAccess(y.source, accessAttr));
              },
              [=](const Indirection<parser::GenericSpec> &y) {
                auto info{GenericSpecInfo{y.value()}};
                const auto &symbolName{info.symbolName()};
                if (auto *symbol{FindInScope(symbolName)}) {
                  info.Resolve(&SetAccess(symbolName, accessAttr, symbol));
                } else if (info.kind().IsName()) {
                  info.Resolve(&SetAccess(symbolName, accessAttr));
                } else {
                  Say(symbolName, "Generic spec '%s' not found"_err_en_US);
                }
              },
          },
          accessId.u);
    }
  }
  return false;
}

// Set the access specification for this symbol.
Symbol &ModuleVisitor::SetAccess(
    const SourceName &name, Attr attr, Symbol *symbol) {
  if (!symbol) {
    symbol = &MakeSymbol(name);
  }
  Attrs &attrs{symbol->attrs()};
  if (attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
    // PUBLIC/PRIVATE already set: make it a fatal error if it changed
    Attr prev = attrs.test(Attr::PUBLIC) ? Attr::PUBLIC : Attr::PRIVATE;
    Say(name,
        WithSeverity(
            "The accessibility of '%s' has already been specified as %s"_warn_en_US,
            attr != prev ? parser::Severity::Error : parser::Severity::Warning),
        MakeOpName(name), EnumToString(prev));
  } else {
    attrs.set(attr);
  }
  return *symbol;
}

static bool NeedsExplicitType(const Symbol &symbol) {
  if (symbol.has<UnknownDetails>()) {
    return true;
  } else if (const auto *details{symbol.detailsIf<EntityDetails>()}) {
    return !details->type();
  } else if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    return !details->type();
  } else if (const auto *details{symbol.detailsIf<ProcEntityDetails>()}) {
    return !details->interface().symbol() && !details->interface().type();
  } else {
    return false;
  }
}

bool ResolveNamesVisitor::Pre(const parser::SpecificationPart &x) {
  const auto &[accDecls, ompDecls, compilerDirectives, useStmts, importStmts,
      implicitPart, decls] = x.t;
  auto flagRestorer{common::ScopedSet(inSpecificationPart_, true)};
  auto stateRestorer{
      common::ScopedSet(specPartState_, SpecificationPartState{})};
  Walk(accDecls);
  Walk(ompDecls);
  Walk(compilerDirectives);
  Walk(useStmts);
  ClearUseRenames();
  ClearUseOnly();
  ClearExplicitIntrinsicUses();
  Walk(importStmts);
  Walk(implicitPart);
  for (const auto &decl : decls) {
    if (const auto *spec{
            std::get_if<parser::SpecificationConstruct>(&decl.u)}) {
      PreSpecificationConstruct(*spec);
    }
  }
  Walk(decls);
  FinishSpecificationPart(decls);
  return false;
}

// Initial processing on specification constructs, before visiting them.
void ResolveNamesVisitor::PreSpecificationConstruct(
    const parser::SpecificationConstruct &spec) {
  common::visit(
      common::visitors{
          [&](const parser::Statement<Indirection<parser::GenericStmt>> &y) {
            CreateGeneric(std::get<parser::GenericSpec>(y.statement.value().t));
          },
          [&](const Indirection<parser::InterfaceBlock> &y) {
            const auto &stmt{std::get<parser::Statement<parser::InterfaceStmt>>(
                y.value().t)};
            if (const auto *spec{parser::Unwrap<parser::GenericSpec>(stmt)}) {
              CreateGeneric(*spec);
            }
          },
          [&](const parser::Statement<parser::OtherSpecificationStmt> &y) {
            if (const auto *commonStmt{parser::Unwrap<parser::CommonStmt>(y)}) {
              CreateCommonBlockSymbols(*commonStmt);
            }
          },
          [&](const auto &) {},
      },
      spec.u);
}

void ResolveNamesVisitor::CreateCommonBlockSymbols(
    const parser::CommonStmt &commonStmt) {
  for (const parser::CommonStmt::Block &block : commonStmt.blocks) {
    const auto &[name, objects] = block.t;
    Symbol &commonBlock{MakeCommonBlockSymbol(name)};
    for (const auto &object : objects) {
      Symbol &obj{DeclareObjectEntity(std::get<parser::Name>(object.t))};
      if (auto *details{obj.detailsIf<ObjectEntityDetails>()}) {
        details->set_commonBlock(commonBlock);
        commonBlock.get<CommonBlockDetails>().add_object(obj);
      }
    }
  }
}

void ResolveNamesVisitor::CreateGeneric(const parser::GenericSpec &x) {
  auto info{GenericSpecInfo{x}};
  SourceName symbolName{info.symbolName()};
  if (IsLogicalConstant(context(), symbolName)) {
    Say(symbolName,
        "Logical constant '%s' may not be used as a defined operator"_err_en_US);
    return;
  }
  GenericDetails genericDetails;
  Symbol *existing{nullptr};
  // Check all variants of names, e.g. "operator(.ne.)" for "operator(/=)"
  for (const std::string &n : GetAllNames(context(), symbolName)) {
    if (auto iter{currScope().find(n)}; iter != currScope().end()) {
      existing = &*iter->second;
      break;
    }
  }
  if (existing) {
    Symbol &ultimate{existing->GetUltimate()};
    if (auto *existingGeneric{ultimate.detailsIf<GenericDetails>()}) {
      if (const auto *existingUse{existing->detailsIf<UseDetails>()}) {
        // Create a local copy of a use associated generic so that
        // it can be locally extended without corrupting the original.
        genericDetails.CopyFrom(*existingGeneric);
        if (existingGeneric->specific()) {
          genericDetails.set_specific(*existingGeneric->specific());
        }
        AddGenericUse(genericDetails, existing->name(), existingUse->symbol());
      } else if (existing == &ultimate) {
        // Extending an extant generic in the same scope
        info.Resolve(existing);
        return;
      } else {
        // Host association of a generic is handled in ResolveGeneric()
        CHECK(existing->has<HostAssocDetails>());
      }
    } else if (ultimate.has<SubprogramDetails>() ||
        ultimate.has<SubprogramNameDetails>()) {
      genericDetails.set_specific(ultimate);
    } else if (ultimate.has<DerivedTypeDetails>()) {
      genericDetails.set_derivedType(ultimate);
    } else {
      SayAlreadyDeclared(symbolName, *existing);
      return;
    }
    EraseSymbol(*existing);
  }
  info.Resolve(&MakeSymbol(symbolName, Attrs{}, std::move(genericDetails)));
}

void ResolveNamesVisitor::FinishSpecificationPart(
    const std::list<parser::DeclarationConstruct> &decls) {
  badStmtFuncFound_ = false;
  funcResultStack().CompleteFunctionResultType();
  CheckImports();
  bool inModule{currScope().kind() == Scope::Kind::Module};
  for (auto &pair : currScope()) {
    auto &symbol{*pair.second};
    if (NeedsExplicitType(symbol)) {
      ApplyImplicitRules(symbol);
    }
    if (IsDummy(symbol) && isImplicitNoneType() &&
        symbol.test(Symbol::Flag::Implicit) && !context().HasError(symbol)) {
      Say(symbol.name(),
          "No explicit type declared for dummy argument '%s'"_err_en_US);
      context().SetError(symbol);
    }
    if (symbol.has<GenericDetails>()) {
      CheckGenericProcedures(symbol);
    }
    if (inModule && symbol.attrs().test(Attr::EXTERNAL) &&
        !symbol.test(Symbol::Flag::Function) &&
        !symbol.test(Symbol::Flag::Subroutine)) {
      // in a module, external proc without return type is subroutine
      symbol.set(
          symbol.GetType() ? Symbol::Flag::Function : Symbol::Flag::Subroutine);
    }
    if (!symbol.has<HostAssocDetails>()) {
      CheckPossibleBadForwardRef(symbol);
    }
  }
  currScope().InstantiateDerivedTypes();
  for (const auto &decl : decls) {
    if (const auto *statement{std::get_if<
            parser::Statement<common::Indirection<parser::StmtFunctionStmt>>>(
            &decl.u)}) {
      AnalyzeStmtFunctionStmt(statement->statement.value());
    }
  }
  // TODO: what about instantiations in BLOCK?
  CheckSaveStmts();
  CheckCommonBlocks();
  if (!inInterfaceBlock()) {
    // TODO: warn for the case where the EQUIVALENCE statement is in a
    // procedure declaration in an interface block
    CheckEquivalenceSets();
  }
}

// Analyze the bodies of statement functions now that the symbols in this
// specification part have been fully declared and implicitly typed.
void ResolveNamesVisitor::AnalyzeStmtFunctionStmt(
    const parser::StmtFunctionStmt &stmtFunc) {
  Symbol *symbol{std::get<parser::Name>(stmtFunc.t).symbol};
  if (!symbol || !symbol->has<SubprogramDetails>()) {
    return;
  }
  auto &details{symbol->get<SubprogramDetails>()};
  auto expr{AnalyzeExpr(
      context(), std::get<parser::Scalar<parser::Expr>>(stmtFunc.t))};
  if (!expr) {
    context().SetError(*symbol);
    return;
  }
  if (auto type{evaluate::DynamicType::From(*symbol)}) {
    auto converted{ConvertToType(*type, std::move(*expr))};
    if (!converted) {
      context().SetError(*symbol);
      return;
    }
    details.set_stmtFunction(std::move(*converted));
  } else {
    details.set_stmtFunction(std::move(*expr));
  }
}

void ResolveNamesVisitor::CheckImports() {
  auto &scope{currScope()};
  switch (scope.GetImportKind()) {
  case common::ImportKind::None:
    break;
  case common::ImportKind::All:
    // C8102: all entities in host must not be hidden
    for (const auto &pair : scope.parent()) {
      auto &name{pair.first};
      std::optional<SourceName> scopeName{scope.GetName()};
      if (!scopeName || name != *scopeName) {
        CheckImport(prevImportStmt_.value(), name);
      }
    }
    break;
  case common::ImportKind::Default:
  case common::ImportKind::Only:
    // C8102: entities named in IMPORT must not be hidden
    for (auto &name : scope.importNames()) {
      CheckImport(name, name);
    }
    break;
  }
}

void ResolveNamesVisitor::CheckImport(
    const SourceName &location, const SourceName &name) {
  if (auto *symbol{FindInScope(name)}) {
    const Symbol &ultimate{symbol->GetUltimate()};
    if (&ultimate.owner() == &currScope()) {
      Say(location, "'%s' from host is not accessible"_err_en_US, name)
          .Attach(symbol->name(), "'%s' is hidden by this entity"_en_US,
              symbol->name());
    }
  }
}

bool ResolveNamesVisitor::Pre(const parser::ImplicitStmt &x) {
  return CheckNotInBlock("IMPLICIT") && // C1107
      ImplicitRulesVisitor::Pre(x);
}

void ResolveNamesVisitor::Post(const parser::PointerObject &x) {
  common::visit(common::visitors{
                    [&](const parser::Name &x) { ResolveName(x); },
                    [&](const parser::StructureComponent &x) {
                      ResolveStructureComponent(x);
                    },
                },
      x.u);
}
void ResolveNamesVisitor::Post(const parser::AllocateObject &x) {
  common::visit(common::visitors{
                    [&](const parser::Name &x) { ResolveName(x); },
                    [&](const parser::StructureComponent &x) {
                      ResolveStructureComponent(x);
                    },
                },
      x.u);
}

bool ResolveNamesVisitor::Pre(const parser::PointerAssignmentStmt &x) {
  const auto &dataRef{std::get<parser::DataRef>(x.t)};
  const auto &bounds{std::get<parser::PointerAssignmentStmt::Bounds>(x.t)};
  const auto &expr{std::get<parser::Expr>(x.t)};
  ResolveDataRef(dataRef);
  Walk(bounds);
  // Resolve unrestricted specific intrinsic procedures as in "p => cos".
  if (const parser::Name * name{parser::Unwrap<parser::Name>(expr)}) {
    if (NameIsKnownOrIntrinsic(*name)) {
      // If the name is known because it is an object entity from a host
      // procedure, create a host associated symbol.
      if (Symbol * symbol{name->symbol}; symbol &&
          symbol->GetUltimate().has<ObjectEntityDetails>() &&
          IsUplevelReference(*symbol)) {
        MakeHostAssocSymbol(*name, *symbol);
      }
      return false;
    }
  }
  Walk(expr);
  return false;
}
void ResolveNamesVisitor::Post(const parser::Designator &x) {
  ResolveDesignator(x);
}
void ResolveNamesVisitor::Post(const parser::SubstringInquiry &x) {
  Walk(std::get<parser::SubstringRange>(x.v.t).t);
  ResolveDataRef(std::get<parser::DataRef>(x.v.t));
}

void ResolveNamesVisitor::Post(const parser::ProcComponentRef &x) {
  ResolveStructureComponent(x.v.thing);
}
void ResolveNamesVisitor::Post(const parser::TypeGuardStmt &x) {
  DeclTypeSpecVisitor::Post(x);
  ConstructVisitor::Post(x);
}
bool ResolveNamesVisitor::Pre(const parser::StmtFunctionStmt &x) {
  CheckNotInBlock("STATEMENT FUNCTION"); // C1107
  if (HandleStmtFunction(x)) {
    return false;
  } else {
    // This is an array element assignment: resolve names of indices
    const auto &names{std::get<std::list<parser::Name>>(x.t)};
    for (auto &name : names) {
      ResolveName(name);
    }
    return true;
  }
}

bool ResolveNamesVisitor::Pre(const parser::DefinedOpName &x) {
  const parser::Name &name{x.v};
  if (FindSymbol(name)) {
    // OK
  } else if (IsLogicalConstant(context(), name.source)) {
    Say(name,
        "Logical constant '%s' may not be used as a defined operator"_err_en_US);
  } else {
    // Resolved later in expression semantics
    MakePlaceholder(name, MiscDetails::Kind::TypeBoundDefinedOp);
  }
  return false;
}

void ResolveNamesVisitor::Post(const parser::AssignStmt &x) {
  if (auto *name{ResolveName(std::get<parser::Name>(x.t))}) {
    ConvertToObjectEntity(DEREF(name->symbol));
  }
}
void ResolveNamesVisitor::Post(const parser::AssignedGotoStmt &x) {
  if (auto *name{ResolveName(std::get<parser::Name>(x.t))}) {
    ConvertToObjectEntity(DEREF(name->symbol));
  }
}

bool ResolveNamesVisitor::Pre(const parser::ProgramUnit &x) {
  if (std::holds_alternative<common::Indirection<parser::CompilerDirective>>(
          x.u)) {
    // TODO: global directives
    return true;
  }
  auto root{ProgramTree::Build(x)};
  SetScope(topScope_);
  ResolveSpecificationParts(root);
  FinishSpecificationParts(root);
  ResolveExecutionParts(root);
  ResolveAccParts(context(), x);
  ResolveOmpParts(context(), x);
  return false;
}

// References to procedures need to record that their symbols are known
// to be procedures, so that they don't get converted to objects by default.
class ExecutionPartSkimmer {
public:
  explicit ExecutionPartSkimmer(ResolveNamesVisitor &resolver)
      : resolver_{resolver} {}

  void Walk(const parser::ExecutionPart *exec) {
    if (exec) {
      parser::Walk(*exec, *this);
    }
  }

  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}
  void Post(const parser::FunctionReference &fr) {
    resolver_.NoteExecutablePartCall(Symbol::Flag::Function, fr.v);
  }
  void Post(const parser::CallStmt &cs) {
    resolver_.NoteExecutablePartCall(Symbol::Flag::Subroutine, cs.v);
  }

private:
  ResolveNamesVisitor &resolver_;
};

// Build the scope tree and resolve names in the specification parts of this
// node and its children
void ResolveNamesVisitor::ResolveSpecificationParts(ProgramTree &node) {
  if (node.isSpecificationPartResolved()) {
    return; // been here already
  }
  node.set_isSpecificationPartResolved();
  if (!BeginScopeForNode(node)) {
    return; // an error prevented scope from being created
  }
  Scope &scope{currScope()};
  node.set_scope(scope);
  AddSubpNames(node);
  common::visit(
      [&](const auto *x) {
        if (x) {
          Walk(*x);
        }
      },
      node.stmt());
  Walk(node.spec());
  // If this is a function, convert result to an object. This is to prevent the
  // result from being converted later to a function symbol if it is called
  // inside the function.
  // If the result is function pointer, then ConvertToObjectEntity will not
  // convert the result to an object, and calling the symbol inside the function
  // will result in calls to the result pointer.
  // A function cannot be called recursively if RESULT was not used to define a
  // distinct result name (15.6.2.2 point 4.).
  if (Symbol * symbol{scope.symbol()}) {
    if (auto *details{symbol->detailsIf<SubprogramDetails>()}) {
      if (details->isFunction()) {
        ConvertToObjectEntity(const_cast<Symbol &>(details->result()));
      }
    }
  }
  if (node.IsModule()) {
    ApplyDefaultAccess();
  }
  for (auto &child : node.children()) {
    ResolveSpecificationParts(child);
  }
  ExecutionPartSkimmer{*this}.Walk(node.exec());
  EndScopeForNode(node);
  // Ensure that every object entity has a type.
  for (auto &pair : *node.scope()) {
    ApplyImplicitRules(*pair.second);
  }
}

// Add SubprogramNameDetails symbols for module and internal subprograms and
// their ENTRY statements.
void ResolveNamesVisitor::AddSubpNames(ProgramTree &node) {
  auto kind{
      node.IsModule() ? SubprogramKind::Module : SubprogramKind::Internal};
  for (auto &child : node.children()) {
    auto &symbol{MakeSymbol(child.name(), SubprogramNameDetails{kind, child})};
    if (child.HasModulePrefix()) {
      symbol.attrs().set(Attr::MODULE);
    }
    auto childKind{child.GetKind()};
    if (childKind == ProgramTree::Kind::Function) {
      symbol.set(Symbol::Flag::Function);
    } else if (childKind == ProgramTree::Kind::Subroutine) {
      symbol.set(Symbol::Flag::Subroutine);
    } else {
      continue; // make ENTRY symbols only where valid
    }
    for (const auto &entryStmt : child.entryStmts()) {
      SubprogramNameDetails details{kind, child};
      auto &symbol{
          MakeSymbol(std::get<parser::Name>(entryStmt->t), std::move(details))};
      symbol.set(child.GetSubpFlag());
      if (child.HasModulePrefix()) {
        symbol.attrs().set(Attr::MODULE);
      }
    }
  }
  for (const auto &generic : node.genericSpecs()) {
    if (const auto *name{std::get_if<parser::Name>(&generic->u)}) {
      if (currScope().find(name->source) != currScope().end()) {
        // If this scope has both a generic interface and a contained
        // subprogram with the same name, create the generic's symbol
        // now so that any other generics of the same name that are pulled
        // into scope later via USE association will properly merge instead
        // of raising a bogus error due a conflict with the subprogram.
        CreateGeneric(*generic);
      }
    }
  }
}

// Push a new scope for this node or return false on error.
bool ResolveNamesVisitor::BeginScopeForNode(const ProgramTree &node) {
  switch (node.GetKind()) {
    SWITCH_COVERS_ALL_CASES
  case ProgramTree::Kind::Program:
    PushScope(Scope::Kind::MainProgram,
        &MakeSymbol(node.name(), MainProgramDetails{}));
    return true;
  case ProgramTree::Kind::Function:
  case ProgramTree::Kind::Subroutine:
    return BeginSubprogram(node.name(), node.GetSubpFlag(),
        node.HasModulePrefix(), node.bindingSpec(), &node.entryStmts());
  case ProgramTree::Kind::MpSubprogram:
    return BeginMpSubprogram(node.name());
  case ProgramTree::Kind::Module:
    BeginModule(node.name(), false);
    return true;
  case ProgramTree::Kind::Submodule:
    return BeginSubmodule(node.name(), node.GetParentId());
  case ProgramTree::Kind::BlockData:
    PushBlockDataScope(node.name());
    return true;
  }
}

void ResolveNamesVisitor::EndScopeForNode(const ProgramTree &node) {
  std::optional<parser::CharBlock> stmtSource;
  const std::optional<parser::LanguageBindingSpec> *binding{nullptr};
  common::visit(
      common::visitors{
          [&](const parser::Statement<parser::FunctionStmt> *stmt) {
            if (stmt) {
              stmtSource = stmt->source;
              if (const auto &maybeSuffix{
                      std::get<std::optional<parser::Suffix>>(
                          stmt->statement.t)}) {
                binding = &maybeSuffix->binding;
              }
            }
          },
          [&](const parser::Statement<parser::SubroutineStmt> *stmt) {
            if (stmt) {
              stmtSource = stmt->source;
              binding = &std::get<std::optional<parser::LanguageBindingSpec>>(
                  stmt->statement.t);
            }
          },
          [](const auto *) {},
      },
      node.stmt());
  EndSubprogram(stmtSource, binding, &node.entryStmts());
}

// Some analyses and checks, such as the processing of initializers of
// pointers, are deferred until all of the pertinent specification parts
// have been visited.  This deferred processing enables the use of forward
// references in these circumstances.
class DeferredCheckVisitor {
public:
  explicit DeferredCheckVisitor(ResolveNamesVisitor &resolver)
      : resolver_{resolver} {}

  template <typename A> void Walk(const A &x) { parser::Walk(x, *this); }

  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}

  void Post(const parser::DerivedTypeStmt &x) {
    const auto &name{std::get<parser::Name>(x.t)};
    if (Symbol * symbol{name.symbol}) {
      if (Scope * scope{symbol->scope()}) {
        if (scope->IsDerivedType()) {
          resolver_.PushScope(*scope);
          pushedScope_ = true;
        }
      }
    }
  }
  void Post(const parser::EndTypeStmt &) {
    if (pushedScope_) {
      resolver_.PopScope();
      pushedScope_ = false;
    }
  }

  void Post(const parser::ProcInterface &pi) {
    if (const auto *name{std::get_if<parser::Name>(&pi.u)}) {
      resolver_.CheckExplicitInterface(*name);
    }
  }
  bool Pre(const parser::EntityDecl &decl) {
    Init(std::get<parser::Name>(decl.t),
        std::get<std::optional<parser::Initialization>>(decl.t));
    return false;
  }
  bool Pre(const parser::ComponentDecl &decl) {
    Init(std::get<parser::Name>(decl.t),
        std::get<std::optional<parser::Initialization>>(decl.t));
    return false;
  }
  bool Pre(const parser::ProcDecl &decl) {
    if (const auto &init{
            std::get<std::optional<parser::ProcPointerInit>>(decl.t)}) {
      resolver_.PointerInitialization(std::get<parser::Name>(decl.t), *init);
    }
    return false;
  }
  void Post(const parser::TypeBoundProcedureStmt::WithInterface &tbps) {
    resolver_.CheckExplicitInterface(tbps.interfaceName);
  }
  void Post(const parser::TypeBoundProcedureStmt::WithoutInterface &tbps) {
    if (pushedScope_) {
      resolver_.CheckBindings(tbps);
    }
  }

private:
  void Init(const parser::Name &name,
      const std::optional<parser::Initialization> &init) {
    if (init) {
      if (const auto *target{
              std::get_if<parser::InitialDataTarget>(&init->u)}) {
        resolver_.PointerInitialization(name, *target);
      }
    }
  }

  ResolveNamesVisitor &resolver_;
  bool pushedScope_{false};
};

// Perform checks and completions that need to happen after all of
// the specification parts but before any of the execution parts.
void ResolveNamesVisitor::FinishSpecificationParts(const ProgramTree &node) {
  if (!node.scope()) {
    return; // error occurred creating scope
  }
  SetScope(*node.scope());
  // The initializers of pointers, the default initializers of pointer
  // components, and non-deferred type-bound procedure bindings have not
  // yet been traversed.
  // We do that now, when any (formerly) forward references that appear
  // in those initializers will resolve to the right symbols without
  // incurring spurious errors with IMPLICIT NONE.
  DeferredCheckVisitor{*this}.Walk(node.spec());
  DeferredCheckVisitor{*this}.Walk(node.exec()); // for BLOCK
  for (Scope &childScope : currScope().children()) {
    if (childScope.IsParameterizedDerivedTypeInstantiation()) {
      FinishDerivedTypeInstantiation(childScope);
    }
  }
  for (const auto &child : node.children()) {
    FinishSpecificationParts(child);
  }
}

// Duplicate and fold component object pointer default initializer designators
// using the actual type parameter values of each particular instantiation.
// Validation is done later in declaration checking.
void ResolveNamesVisitor::FinishDerivedTypeInstantiation(Scope &scope) {
  CHECK(scope.IsDerivedType() && !scope.symbol());
  if (DerivedTypeSpec * spec{scope.derivedTypeSpec()}) {
    spec->Instantiate(currScope());
    const Symbol &origTypeSymbol{spec->typeSymbol()};
    if (const Scope * origTypeScope{origTypeSymbol.scope()}) {
      CHECK(origTypeScope->IsDerivedType() &&
          origTypeScope->symbol() == &origTypeSymbol);
      auto &foldingContext{GetFoldingContext()};
      auto restorer{foldingContext.WithPDTInstance(*spec)};
      for (auto &pair : scope) {
        Symbol &comp{*pair.second};
        const Symbol &origComp{DEREF(FindInScope(*origTypeScope, comp.name()))};
        if (IsPointer(comp)) {
          if (auto *details{comp.detailsIf<ObjectEntityDetails>()}) {
            auto origDetails{origComp.get<ObjectEntityDetails>()};
            if (const MaybeExpr & init{origDetails.init()}) {
              SomeExpr newInit{*init};
              MaybeExpr folded{
                  evaluate::Fold(foldingContext, std::move(newInit))};
              details->set_init(std::move(folded));
            }
          }
        }
      }
    }
  }
}

// Resolve names in the execution part of this node and its children
void ResolveNamesVisitor::ResolveExecutionParts(const ProgramTree &node) {
  if (!node.scope()) {
    return; // error occurred creating scope
  }
  SetScope(*node.scope());
  if (const auto *exec{node.exec()}) {
    Walk(*exec);
  }
  FinishNamelists();
  PopScope(); // converts unclassified entities into objects
  for (const auto &child : node.children()) {
    ResolveExecutionParts(child);
  }
}

void ResolveNamesVisitor::Post(const parser::Program &) {
  // ensure that all temps were deallocated
  CHECK(!attrs_);
  CHECK(!GetDeclTypeSpec());
}

// A singleton instance of the scope -> IMPLICIT rules mapping is
// shared by all instances of ResolveNamesVisitor and accessed by this
// pointer when the visitors (other than the top-level original) are
// constructed.
static ImplicitRulesMap *sharedImplicitRulesMap{nullptr};

bool ResolveNames(
    SemanticsContext &context, const parser::Program &program, Scope &top) {
  ImplicitRulesMap implicitRulesMap;
  auto restorer{common::ScopedSet(sharedImplicitRulesMap, &implicitRulesMap)};
  ResolveNamesVisitor{context, implicitRulesMap, top}.Walk(program);
  return !context.AnyFatalError();
}

// Processes a module (but not internal) function when it is referenced
// in a specification expression in a sibling procedure.
void ResolveSpecificationParts(
    SemanticsContext &context, const Symbol &subprogram) {
  auto originalLocation{context.location()};
  ImplicitRulesMap implicitRulesMap;
  bool localImplicitRulesMap{false};
  if (!sharedImplicitRulesMap) {
    sharedImplicitRulesMap = &implicitRulesMap;
    localImplicitRulesMap = true;
  }
  ResolveNamesVisitor visitor{
      context, *sharedImplicitRulesMap, context.globalScope()};
  const auto &details{subprogram.get<SubprogramNameDetails>()};
  ProgramTree &node{details.node()};
  const Scope &moduleScope{subprogram.owner()};
  if (localImplicitRulesMap) {
    visitor.BeginScope(const_cast<Scope &>(moduleScope));
  } else {
    visitor.SetScope(const_cast<Scope &>(moduleScope));
  }
  visitor.ResolveSpecificationParts(node);
  context.set_location(std::move(originalLocation));
  if (localImplicitRulesMap) {
    sharedImplicitRulesMap = nullptr;
  }
}

} // namespace Fortran::semantics
