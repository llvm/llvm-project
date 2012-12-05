//===--- Scope.h - Scope interface ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Scope interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SCOPE_H
#define LLVM_CLANG_SEMA_SCOPE_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class Decl;
class UsingDirectiveDecl;

/// Scope - A scope is a transient data structure that is used while parsing the
/// program.  It assists with resolving identifiers to the appropriate
/// declaration.
///
class Scope {
public:
  /// ScopeFlags - These are bitfields that are or'd together when creating a
  /// scope, which defines the sorts of things the scope contains.
  enum ScopeFlags {
    /// FnScope - This indicates that the scope corresponds to a function, which
    /// means that labels are set here.
    FnScope       = 0x01,

    /// BreakScope - This is a while,do,switch,for, etc that can have break
    /// stmts embedded into it.
    BreakScope    = 0x02,

    /// ContinueScope - This is a while,do,for, which can have continue
    /// stmt embedded into it.
    ContinueScope = 0x04,

    /// DeclScope - This is a scope that can contain a declaration.  Some scopes
    /// just contain loop constructs but don't contain decls.
    DeclScope = 0x08,

    /// ControlScope - The controlling scope in a if/switch/while/for statement.
    ControlScope = 0x10,

    /// ClassScope - The scope of a struct/union/class definition.
    ClassScope = 0x20,

    /// BlockScope - This is a scope that corresponds to a block/closure object.
    /// Blocks serve as top-level scopes for some objects like labels, they
    /// also prevent things like break and continue.  BlockScopes always have
    /// the FnScope and DeclScope flags set as well.
    BlockScope = 0x40,

    /// TemplateParamScope - This is a scope that corresponds to the
    /// template parameters of a C++ template. Template parameter
    /// scope starts at the 'template' keyword and ends when the
    /// template declaration ends.
    TemplateParamScope = 0x80,

    /// FunctionPrototypeScope - This is a scope that corresponds to the
    /// parameters within a function prototype.
    FunctionPrototypeScope = 0x100,

    /// AtCatchScope - This is a scope that corresponds to the Objective-C
    /// \@catch statement.
    AtCatchScope = 0x200,
    
    /// ObjCMethodScope - This scope corresponds to an Objective-C method body.
    /// It always has FnScope and DeclScope set as well.
    ObjCMethodScope = 0x400,

    /// SwitchScope - This is a scope that corresponds to a switch statement.
    SwitchScope = 0x800,

    /// TryScope - This is the scope of a C++ try statement.
    TryScope = 0x1000,

    /// CatchScope - This is the scope of a C++ catch statement.
    CatchScope = 0x2000,

    /// FnTryCatchScope - This is the scope for a function-level C++ try or
    /// catch scope.
    FnTryCatchScope = 0x4000,

    /// FnTryScope - This is the scope of a function-level C++ try scope.
    FnTryScope = TryScope | FnTryCatchScope,

    /// FnCatchScope - This is the scope of a function-level C++ catch scope.
    FnCatchScope = CatchScope | FnTryCatchScope
  };
private:
  /// The parent scope for this scope.  This is null for the translation-unit
  /// scope.
  Scope *AnyParent;

  /// Depth - This is the depth of this scope.  The translation-unit scope has
  /// depth 0.
  unsigned short Depth;

  /// Flags - This contains a set of ScopeFlags, which indicates how the scope
  /// interrelates with other control flow statements.
  unsigned short Flags;

  /// PrototypeDepth - This is the number of function prototype scopes
  /// enclosing this scope, including this scope.
  unsigned short PrototypeDepth;

  /// PrototypeIndex - This is the number of parameters currently
  /// declared in this scope.
  unsigned short PrototypeIndex;

  /// FnParent - If this scope has a parent scope that is a function body, this
  /// pointer is non-null and points to it.  This is used for label processing.
  Scope *FnParent;

  /// BreakParent/ContinueParent - This is a direct link to the innermost
  /// BreakScope/ContinueScope which contains the contents of this scope
  /// for control flow purposes (and might be this scope itself), or null
  /// if there is no such scope.
  Scope *BreakParent, *ContinueParent;

  /// BlockParent - This is a direct link to the immediately containing
  /// BlockScope if this scope is not one, or null if there is none.
  Scope *BlockParent;

  /// TemplateParamParent - This is a direct link to the
  /// immediately containing template parameter scope. In the
  /// case of nested templates, template parameter scopes can have
  /// other template parameter scopes as parents.
  Scope *TemplateParamParent;

  /// DeclsInScope - This keeps track of all declarations in this scope.  When
  /// the declaration is added to the scope, it is set as the current
  /// declaration for the identifier in the IdentifierTable.  When the scope is
  /// popped, these declarations are removed from the IdentifierTable's notion
  /// of current declaration.  It is up to the current Action implementation to
  /// implement these semantics.
  typedef llvm::SmallPtrSet<Decl *, 32> DeclSetTy;
  DeclSetTy DeclsInScope;

  /// Entity - The entity with which this scope is associated. For
  /// example, the entity of a class scope is the class itself, the
  /// entity of a function scope is a function, etc. This field is
  /// maintained by the Action implementation.
  void *Entity;

  typedef SmallVector<UsingDirectiveDecl *, 2> UsingDirectivesTy;
  UsingDirectivesTy UsingDirectives;

  /// \brief Used to determine if errors occurred in this scope.
  DiagnosticErrorTrap ErrorTrap;
  
public:
  Scope(Scope *Parent, unsigned ScopeFlags, DiagnosticsEngine &Diag)
    : ErrorTrap(Diag) {
    Init(Parent, ScopeFlags);
  }

  /// getFlags - Return the flags for this scope.
  ///
  unsigned getFlags() const { return Flags; }
  void setFlags(unsigned F) { Flags = F; }

  /// isBlockScope - Return true if this scope correspond to a closure.
  bool isBlockScope() const { return Flags & BlockScope; }

  /// getParent - Return the scope that this is nested in.
  ///
  const Scope *getParent() const { return AnyParent; }
  Scope *getParent() { return AnyParent; }

  /// getFnParent - Return the closest scope that is a function body.
  ///
  const Scope *getFnParent() const { return FnParent; }
  Scope *getFnParent() { return FnParent; }

  /// getContinueParent - Return the closest scope that a continue statement
  /// would be affected by.
  Scope *getContinueParent() {
    return ContinueParent;
  }

  const Scope *getContinueParent() const {
    return const_cast<Scope*>(this)->getContinueParent();
  }

  /// getBreakParent - Return the closest scope that a break statement
  /// would be affected by.
  Scope *getBreakParent() {
    return BreakParent;
  }
  const Scope *getBreakParent() const {
    return const_cast<Scope*>(this)->getBreakParent();
  }

  Scope *getBlockParent() { return BlockParent; }
  const Scope *getBlockParent() const { return BlockParent; }

  Scope *getTemplateParamParent() { return TemplateParamParent; }
  const Scope *getTemplateParamParent() const { return TemplateParamParent; }

  /// Returns the number of function prototype scopes in this scope
  /// chain.
  unsigned getFunctionPrototypeDepth() const {
    return PrototypeDepth;
  }

  /// Return the number of parameters declared in this function
  /// prototype, increasing it by one for the next call.
  unsigned getNextFunctionPrototypeIndex() {
    assert(isFunctionPrototypeScope());
    return PrototypeIndex++;
  }

  typedef DeclSetTy::iterator decl_iterator;
  decl_iterator decl_begin() const { return DeclsInScope.begin(); }
  decl_iterator decl_end()   const { return DeclsInScope.end(); }
  bool decl_empty()          const { return DeclsInScope.empty(); }

  void AddDecl(Decl *D) {
    DeclsInScope.insert(D);
  }

  void RemoveDecl(Decl *D) {
    DeclsInScope.erase(D);
  }

  /// isDeclScope - Return true if this is the scope that the specified decl is
  /// declared in.
  bool isDeclScope(Decl *D) {
    return DeclsInScope.count(D) != 0;
  }

  void* getEntity() const { return Entity; }
  void setEntity(void *E) { Entity = E; }

  bool hasErrorOccurred() const { return ErrorTrap.hasErrorOccurred(); }
                           
  /// isClassScope - Return true if this scope is a class/struct/union scope.
  bool isClassScope() const {
    return (getFlags() & Scope::ClassScope);
  }

  /// isInCXXInlineMethodScope - Return true if this scope is a C++ inline
  /// method scope or is inside one.
  bool isInCXXInlineMethodScope() const {
    if (const Scope *FnS = getFnParent()) {
      assert(FnS->getParent() && "TUScope not created?");
      return FnS->getParent()->isClassScope();
    }
    return false;
  }
  
  /// isInObjcMethodScope - Return true if this scope is, or is contained in, an
  /// Objective-C method body.  Note that this method is not constant time.
  bool isInObjcMethodScope() const {
    for (const Scope *S = this; S; S = S->getParent()) {
      // If this scope is an objc method scope, then we succeed.
      if (S->getFlags() & ObjCMethodScope)
        return true;
    }
    return false;
  }

  /// isTemplateParamScope - Return true if this scope is a C++
  /// template parameter scope.
  bool isTemplateParamScope() const {
    return getFlags() & Scope::TemplateParamScope;
  }

  /// isFunctionPrototypeScope - Return true if this scope is a
  /// function prototype scope.
  bool isFunctionPrototypeScope() const {
    return getFlags() & Scope::FunctionPrototypeScope;
  }

  /// isAtCatchScope - Return true if this scope is \@catch.
  bool isAtCatchScope() const {
    return getFlags() & Scope::AtCatchScope;
  }

  /// isSwitchScope - Return true if this scope is a switch scope.
  bool isSwitchScope() const {
    for (const Scope *S = this; S; S = S->getParent()) {
      if (S->getFlags() & Scope::SwitchScope)
        return true;
      else if (S->getFlags() & (Scope::FnScope | Scope::ClassScope |
                                Scope::BlockScope | Scope::TemplateParamScope |
                                Scope::FunctionPrototypeScope |
                                Scope::AtCatchScope | Scope::ObjCMethodScope))
        return false;
    }
    return false;
  }
  
  /// \brief Determine whether this scope is a C++ 'try' block.
  bool isTryScope() const { return getFlags() & Scope::TryScope; }

  /// containedInPrototypeScope - Return true if this or a parent scope
  /// is a FunctionPrototypeScope.
  bool containedInPrototypeScope() const;

  typedef UsingDirectivesTy::iterator udir_iterator;
  typedef UsingDirectivesTy::const_iterator const_udir_iterator;

  void PushUsingDirective(UsingDirectiveDecl *UDir) {
    UsingDirectives.push_back(UDir);
  }

  udir_iterator using_directives_begin() {
    return UsingDirectives.begin();
  }

  udir_iterator using_directives_end() {
    return UsingDirectives.end();
  }

  const_udir_iterator using_directives_begin() const {
    return UsingDirectives.begin();
  }

  const_udir_iterator using_directives_end() const {
    return UsingDirectives.end();
  }

  /// Init - This is used by the parser to implement scope caching.
  ///
  void Init(Scope *parent, unsigned flags);
};

}  // end namespace clang

#endif
