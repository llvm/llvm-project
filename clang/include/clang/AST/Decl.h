//===--- Decl.h - Classes for representing declarations ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Decl subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECL_H
#define LLVM_CLANG_AST_DECL_H

#include "clang/AST/APValue.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Redeclarable.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExternalASTSource.h"

namespace clang {
class Expr;
class FunctionTemplateDecl;
class Stmt;
class CompoundStmt;
class StringLiteral;
class TemplateArgumentList;
class FunctionTemplateSpecializationInfo;
class TypeLoc;

/// \brief A container of type source information.
///
/// A client can read the relevant info using TypeLoc wrappers, e.g:
/// @code
/// TypeLoc TL = DeclaratorInfo->getTypeLoc();
/// if (PointerLoc *PL = dyn_cast<PointerLoc>(&TL))
///   PL->getStarLoc().print(OS, SrcMgr);
/// @endcode
///
class DeclaratorInfo {
  QualType Ty;
  // Contains a memory block after the class, used for type source information,
  // allocated by ASTContext.
  friend class ASTContext;
  DeclaratorInfo(QualType ty) : Ty(ty) { }
public:
  /// \brief Return the TypeLoc wrapper for the type source info.
  TypeLoc getTypeLoc() const;
};

/// TranslationUnitDecl - The top declaration context.
class TranslationUnitDecl : public Decl, public DeclContext {
  ASTContext &Ctx;
  
  explicit TranslationUnitDecl(ASTContext &ctx)
    : Decl(TranslationUnit, 0, SourceLocation()),
      DeclContext(TranslationUnit),
      Ctx(ctx) {}
public:
  ASTContext &getASTContext() const { return Ctx; }
  
  static TranslationUnitDecl *Create(ASTContext &C);
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == TranslationUnit; }
  static bool classof(const TranslationUnitDecl *D) { return true; }  
  static DeclContext *castToDeclContext(const TranslationUnitDecl *D) {
    return static_cast<DeclContext *>(const_cast<TranslationUnitDecl*>(D));
  }
  static TranslationUnitDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<TranslationUnitDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// NamedDecl - This represents a decl with a name.  Many decls have names such
/// as ObjCMethodDecl, but not @class, etc.
class NamedDecl : public Decl {
  /// Name - The name of this declaration, which is typically a normal
  /// identifier but may also be a special kind of name (C++
  /// constructor, Objective-C selector, etc.)
  DeclarationName Name;

protected:
  NamedDecl(Kind DK, DeclContext *DC, SourceLocation L, DeclarationName N)
    : Decl(DK, DC, L), Name(N) { }

public:
  /// getIdentifier - Get the identifier that names this declaration,
  /// if there is one. This will return NULL if this declaration has
  /// no name (e.g., for an unnamed class) or if the name is a special
  /// name (C++ constructor, Objective-C selector, etc.).
  IdentifierInfo *getIdentifier() const { return Name.getAsIdentifierInfo(); }

  /// getNameAsCString - Get the name of identifier for this declaration as a
  /// C string (const char*).  This requires that the declaration have a name
  /// and that it be a simple identifier.
  const char *getNameAsCString() const {
    assert(getIdentifier() && "Name is not a simple identifier");
    return getIdentifier()->getName();
  }

  /// getDeclName - Get the actual, stored name of the declaration,
  /// which may be a special name.
  DeclarationName getDeclName() const { return Name; }

  /// \brief Set the name of this declaration.
  void setDeclName(DeclarationName N) { Name = N; }

  /// getNameAsString - Get a human-readable name for the declaration, even if
  /// it is one of the special kinds of names (C++ constructor, Objective-C
  /// selector, etc).  Creating this name requires expensive string
  /// manipulation, so it should be called only when performance doesn't matter.
  /// For simple declarations, getNameAsCString() should suffice.
  std::string getNameAsString() const { return Name.getAsString(); }
  
  /// getQualifiedNameAsString - Returns human-readable qualified name for
  /// declaration, like A::B::i, for i being member of namespace A::B.
  /// If declaration is not member of context which can be named (record,
  /// namespace), it will return same result as getNameAsString().
  /// Creating this name is expensive, so it should be called only when
  /// performance doesn't matter.
  std::string getQualifiedNameAsString() const;

  /// declarationReplaces - Determine whether this declaration, if
  /// known to be well-formed within its context, will replace the
  /// declaration OldD if introduced into scope. A declaration will
  /// replace another declaration if, for example, it is a
  /// redeclaration of the same variable or function, but not if it is
  /// a declaration of a different kind (function vs. class) or an
  /// overloaded function.
  bool declarationReplaces(NamedDecl *OldD) const;

  /// \brief Determine whether this declaration has linkage.
  bool hasLinkage() const;

  /// \brief Looks through UsingDecls and ObjCCompatibleAliasDecls for
  /// the underlying named decl.
  NamedDecl *getUnderlyingDecl();
  const NamedDecl *getUnderlyingDecl() const {
    return const_cast<NamedDecl*>(this)->getUnderlyingDecl();
  }
  
  static bool classof(const Decl *D) {
    return D->getKind() >= NamedFirst && D->getKind() <= NamedLast;
  }
  static bool classof(const NamedDecl *D) { return true; }
};

/// NamespaceDecl - Represent a C++ namespace.
class NamespaceDecl : public NamedDecl, public DeclContext {
  SourceLocation LBracLoc, RBracLoc;
  
  // For extended namespace definitions:
  //
  // namespace A { int x; }
  // namespace A { int y; }
  //
  // there will be one NamespaceDecl for each declaration.
  // NextNamespace points to the next extended declaration.
  // OrigNamespace points to the original namespace declaration.
  // OrigNamespace of the first namespace decl points to itself.
  NamespaceDecl *OrigNamespace, *NextNamespace;
  
  NamespaceDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id)
    : NamedDecl(Namespace, DC, L, Id), DeclContext(Namespace) {
    OrigNamespace = this;
    NextNamespace = 0;
  }
public:
  static NamespaceDecl *Create(ASTContext &C, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id);
  
  virtual void Destroy(ASTContext& C);

  NamespaceDecl *getNextNamespace() { return NextNamespace; }
  const NamespaceDecl *getNextNamespace() const { return NextNamespace; }
  void setNextNamespace(NamespaceDecl *ND) { NextNamespace = ND; }

  NamespaceDecl *getOriginalNamespace() const {
    return OrigNamespace;
  }
  void setOriginalNamespace(NamespaceDecl *ND) { OrigNamespace = ND; }
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(getLocation(), RBracLoc);
  }

  SourceLocation getLBracLoc() const { return LBracLoc; }
  SourceLocation getRBracLoc() const { return RBracLoc; }
  void setLBracLoc(SourceLocation LBrace) { LBracLoc = LBrace; }
  void setRBracLoc(SourceLocation RBrace) { RBracLoc = RBrace; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Namespace; }
  static bool classof(const NamespaceDecl *D) { return true; }
  static DeclContext *castToDeclContext(const NamespaceDecl *D) {
    return static_cast<DeclContext *>(const_cast<NamespaceDecl*>(D));
  }
  static NamespaceDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<NamespaceDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// ValueDecl - Represent the declaration of a variable (in which case it is 
/// an lvalue) a function (in which case it is a function designator) or
/// an enum constant. 
class ValueDecl : public NamedDecl {
  QualType DeclType;

protected:
  ValueDecl(Kind DK, DeclContext *DC, SourceLocation L,
            DeclarationName N, QualType T) 
    : NamedDecl(DK, DC, L, N), DeclType(T) {}
public:
  QualType getType() const { return DeclType; }
  void setType(QualType newType) { DeclType = newType; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= ValueFirst && D->getKind() <= ValueLast;
  }
  static bool classof(const ValueDecl *D) { return true; }
};

/// \brief Represents a ValueDecl that came out of a declarator.
/// Contains type source information through DeclaratorInfo.
class DeclaratorDecl : public ValueDecl {
  DeclaratorInfo *DeclInfo;

protected:
  DeclaratorDecl(Kind DK, DeclContext *DC, SourceLocation L,
                 DeclarationName N, QualType T, DeclaratorInfo *DInfo)
    : ValueDecl(DK, DC, L, N, T), DeclInfo(DInfo) {}

public:
  DeclaratorInfo *getDeclaratorInfo() const { return DeclInfo; }
  void setDeclaratorInfo(DeclaratorInfo *DInfo) { DeclInfo = DInfo; }

  SourceLocation getTypeSpecStartLoc() const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= DeclaratorFirst && D->getKind() <= DeclaratorLast;
  }
  static bool classof(const DeclaratorDecl *D) { return true; }
};

/// \brief Structure used to store a statement, the constant value to
/// which it was evaluated (if any), and whether or not the statement
/// is an integral constant expression (if known).
struct EvaluatedStmt {
  EvaluatedStmt() : WasEvaluated(false), CheckedICE(false), IsICE(false) { }

  /// \brief Whether this statement was already evaluated.
  bool WasEvaluated : 1;

  /// \brief Whether we already checked whether this statement was an
  /// integral constant expression.
  bool CheckedICE : 1;

  /// \brief Whether this statement is an integral constant
  /// expression. Only valid if CheckedICE is true.
  bool IsICE : 1;

  Stmt *Value;
  APValue Evaluated;
};

/// VarDecl - An instance of this class is created to represent a variable
/// declaration or definition.
class VarDecl : public DeclaratorDecl, public Redeclarable<VarDecl> {
public:
  enum StorageClass {
    None, Auto, Register, Extern, Static, PrivateExtern
  };

  /// getStorageClassSpecifierString - Return the string used to
  /// specify the storage class \arg SC.
  ///
  /// It is illegal to call this function with SC == None.
  static const char *getStorageClassSpecifierString(StorageClass SC);

protected:
  /// \brief Placeholder type used in Init to denote an unparsed C++ default
  /// argument.
  struct UnparsedDefaultArgument;
  
  typedef llvm::PointerUnion3<Stmt *, EvaluatedStmt *, 
                              UnparsedDefaultArgument *> InitType;
  
  /// \brief The initializer for this variable or, for a ParmVarDecl, the 
  /// C++ default argument.
  mutable InitType Init;
  
private:
  // FIXME: This can be packed into the bitfields in Decl.
  unsigned SClass : 3;
  bool ThreadSpecified : 1;
  bool HasCXXDirectInit : 1; 

  /// DeclaredInCondition - Whether this variable was declared in a
  /// condition, e.g., if (int x = foo()) { ... }.
  bool DeclaredInCondition : 1;

  friend class StmtIteratorBase;
protected:
  VarDecl(Kind DK, DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
          QualType T, DeclaratorInfo *DInfo, StorageClass SC)
    : DeclaratorDecl(DK, DC, L, Id, T, DInfo), Init(),
      ThreadSpecified(false), HasCXXDirectInit(false),
      DeclaredInCondition(false) { 
    SClass = SC; 
  }

  typedef Redeclarable<VarDecl> redeclarable_base;
  virtual VarDecl *getNextRedeclaration() { return RedeclLink.getNext(); }

public:
  typedef redeclarable_base::redecl_iterator redecl_iterator;
  redecl_iterator redecls_begin() const {
    return redeclarable_base::redecls_begin();
  } 
  redecl_iterator redecls_end() const {
    return redeclarable_base::redecls_end();
  }

  static VarDecl *Create(ASTContext &C, DeclContext *DC,
                         SourceLocation L, IdentifierInfo *Id,
                         QualType T, DeclaratorInfo *DInfo, StorageClass S);

  virtual ~VarDecl();
  virtual void Destroy(ASTContext& C);

  StorageClass getStorageClass() const { return (StorageClass)SClass; }
  void setStorageClass(StorageClass SC) { SClass = SC; }
  
  virtual SourceRange getSourceRange() const;

  const Expr *getInit() const { 
    if (Init.isNull())
      return 0;

    const Stmt *S = Init.dyn_cast<Stmt *>();
    if (!S) {
      if (EvaluatedStmt *ES = Init.dyn_cast<EvaluatedStmt*>())
        S = ES->Value;
    }
    return (const Expr*) S; 
  }
  Expr *getInit() { 
    if (Init.isNull())
      return 0;

    Stmt *S = Init.dyn_cast<Stmt *>();
    if (!S) {
      if (EvaluatedStmt *ES = Init.dyn_cast<EvaluatedStmt*>())
        S = ES->Value;
    }

    return (Expr*) S; 
  }

  /// \brief Retrieve the address of the initializer expression.
  Stmt **getInitAddress() {
    if (EvaluatedStmt *ES = Init.dyn_cast<EvaluatedStmt*>())
      return &ES->Value;
    
    // This union hack tip-toes around strict-aliasing rules.
    union {
      InitType *InitPtr;
      Stmt **StmtPtr;
    };
    
    InitPtr = &Init;
    return StmtPtr;
  }

  void setInit(ASTContext &C, Expr *I);
   
  /// \brief Note that constant evaluation has computed the given
  /// value for this variable's initializer.
  void setEvaluatedValue(ASTContext &C, const APValue &Value) const {
    EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>();
    if (!Eval) {
      Stmt *S = Init.get<Stmt *>();
      Eval = new (C) EvaluatedStmt;
      Eval->Value = S;
      Init = Eval;
    }

    Eval->WasEvaluated = true;
    Eval->Evaluated = Value;
  }
   
  /// \brief Return the already-evaluated value of this variable's
  /// initializer, or NULL if the value is not yet known.
  APValue *getEvaluatedValue() const {
    if (EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>())
      if (Eval->WasEvaluated)
        return &Eval->Evaluated;

    return 0;
  }

  /// \brief Determines whether it is already known whether the
  /// initializer is an integral constant expression or not.
  bool isInitKnownICE() const {
    if (EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>())
      return Eval->CheckedICE;

    return false;
  }

  /// \brief Determines whether the initializer is an integral
  /// constant expression.
  ///
  /// \pre isInitKnownICE()
  bool isInitICE() const {
    assert(isInitKnownICE() && 
           "Check whether we already know that the initializer is an ICE");
    return Init.get<EvaluatedStmt *>()->IsICE;
  }

  /// \brief Note that we now know whether the initializer is an
  /// integral constant expression.
  void setInitKnownICE(ASTContext &C, bool IsICE) const {
    EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>();
    if (!Eval) {
      Stmt *S = Init.get<Stmt *>();
      Eval = new (C) EvaluatedStmt;
      Eval->Value = S;
      Init = Eval;
    }

    Eval->CheckedICE = true;
    Eval->IsICE = IsICE;
  }

  /// \brief Retrieve the definition of this variable, which may come
  /// from a previous declaration. Def will be set to the VarDecl that
  /// contains the initializer, and the result will be that
  /// initializer.
  const Expr *getDefinition(const VarDecl *&Def) const;

  void setThreadSpecified(bool T) { ThreadSpecified = T; }
  bool isThreadSpecified() const {
    return ThreadSpecified;
  }

  void setCXXDirectInitializer(bool T) { HasCXXDirectInit = T; }

  /// hasCXXDirectInitializer - If true, the initializer was a direct
  /// initializer, e.g: "int x(1);". The Init expression will be the expression
  /// inside the parens or a "ClassType(a,b,c)" class constructor expression for
  /// class types. Clients can distinguish between "int x(1);" and "int x=1;"
  /// by checking hasCXXDirectInitializer.
  ///
  bool hasCXXDirectInitializer() const {
    return HasCXXDirectInit;
  }
  
  /// isDeclaredInCondition - Whether this variable was declared as
  /// part of a condition in an if/switch/while statement, e.g.,
  /// @code
  /// if (int x = foo()) { ... }
  /// @endcode
  bool isDeclaredInCondition() const {
    return DeclaredInCondition;
  }
  void setDeclaredInCondition(bool InCondition) { 
    DeclaredInCondition = InCondition; 
  }

  virtual VarDecl *getCanonicalDecl();

  /// hasLocalStorage - Returns true if a variable with function scope
  ///  is a non-static local variable.
  bool hasLocalStorage() const {
    if (getStorageClass() == None)
      return !isFileVarDecl();
    
    // Return true for:  Auto, Register.
    // Return false for: Extern, Static, PrivateExtern.
    
    return getStorageClass() <= Register;
  }

  /// hasExternStorage - Returns true if a variable has extern or
  /// __private_extern__ storage.
  bool hasExternalStorage() const {
    return getStorageClass() == Extern || getStorageClass() == PrivateExtern;
  }

  /// hasGlobalStorage - Returns true for all variables that do not
  ///  have local storage.  This includs all global variables as well
  ///  as static variables declared within a function.
  bool hasGlobalStorage() const { return !hasLocalStorage(); }

  /// isBlockVarDecl - Returns true for local variable declarations.  Note that
  /// this includes static variables inside of functions.
  ///
  ///   void foo() { int x; static int y; extern int z; }
  ///
  bool isBlockVarDecl() const {
    if (getKind() != Decl::Var)
      return false;
    if (const DeclContext *DC = getDeclContext())
      return DC->getLookupContext()->isFunctionOrMethod();
    return false;
  }
  
  /// \brief Determines whether this is a static data member.
  ///
  /// This will only be true in C++, and applies to, e.g., the
  /// variable 'x' in:
  /// \code
  /// struct S {
  ///   static int x;
  /// };
  /// \endcode
  bool isStaticDataMember() const {
    return getDeclContext()->isRecord();
  }

  /// \brief If this variable is an instantiated static data member of a
  /// class template specialization, returns the templated static data member 
  /// from which it was instantiated.
  VarDecl *getInstantiatedFromStaticDataMember(); 
  
  /// isFileVarDecl - Returns true for file scoped variable declaration.
  bool isFileVarDecl() const {
    if (getKind() != Decl::Var)
      return false;
    if (const DeclContext *Ctx = getDeclContext()) {
      Ctx = Ctx->getLookupContext();
      if (isa<TranslationUnitDecl>(Ctx) || isa<NamespaceDecl>(Ctx) )
        return true;
    }
    if (isStaticDataMember() && isOutOfLine())
      return true;
    
    return false;
  }

  /// \brief Determine whether this is a tentative definition of a
  /// variable in C.
  bool isTentativeDefinition(ASTContext &Context) const;
  
  /// \brief Determines whether this variable is a variable with
  /// external, C linkage.
  bool isExternC(ASTContext &Context) const;

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= VarFirst && D->getKind() <= VarLast;
  }
  static bool classof(const VarDecl *D) { return true; }
};

class ImplicitParamDecl : public VarDecl {
protected:
  ImplicitParamDecl(Kind DK, DeclContext *DC, SourceLocation L,
                    IdentifierInfo *Id, QualType Tw) 
    : VarDecl(DK, DC, L, Id, Tw, /*DInfo=*/0, VarDecl::None) {}
public:
  static ImplicitParamDecl *Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L, IdentifierInfo *Id,
                                   QualType T);
  // Implement isa/cast/dyncast/etc.
  static bool classof(const ImplicitParamDecl *D) { return true; }
  static bool classof(const Decl *D) { return D->getKind() == ImplicitParam; }
};

/// ParmVarDecl - Represent a parameter to a function.
class ParmVarDecl : public VarDecl {
  // NOTE: VC++ treats enums as signed, avoid using the ObjCDeclQualifier enum
  /// FIXME: Also can be paced into the bitfields in Decl.
  /// in, inout, etc.
  unsigned objcDeclQualifier : 6;
  
  /// \brief Retrieves the fake "value" of an unparsed 
  static Expr *getUnparsedDefaultArgValue() {
    uintptr_t Value = (uintptr_t)-1;
    // Mask off the low bits
    Value &= ~(uintptr_t)0x07;
    return reinterpret_cast<Expr*> (Value);
  }
  
protected:
  ParmVarDecl(Kind DK, DeclContext *DC, SourceLocation L,
              IdentifierInfo *Id, QualType T, DeclaratorInfo *DInfo,
              StorageClass S, Expr *DefArg)
  : VarDecl(DK, DC, L, Id, T, DInfo, S), objcDeclQualifier(OBJC_TQ_None) {
    setDefaultArg(DefArg);
  }

public:
  static ParmVarDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             QualType T, DeclaratorInfo *DInfo,
                             StorageClass S, Expr *DefArg);
  
  ObjCDeclQualifier getObjCDeclQualifier() const {
    return ObjCDeclQualifier(objcDeclQualifier);
  }
  void setObjCDeclQualifier(ObjCDeclQualifier QTVal) {
    objcDeclQualifier = QTVal;
  }
    
  const Expr *getDefaultArg() const { 
    assert(!hasUnparsedDefaultArg() && "Default argument is not yet parsed!");
    return getInit();
  }
  Expr *getDefaultArg() { 
    assert(!hasUnparsedDefaultArg() && "Default argument is not yet parsed!");
    return getInit();
  }
  void setDefaultArg(Expr *defarg) { 
    Init = reinterpret_cast<Stmt *>(defarg);
  }

  /// hasDefaultArg - Determines whether this parameter has a default argument,
  /// either parsed or not.
  bool hasDefaultArg() const {
    return getInit() || hasUnparsedDefaultArg();
  }
  
  /// hasUnparsedDefaultArg - Determines whether this parameter has a
  /// default argument that has not yet been parsed. This will occur
  /// during the processing of a C++ class whose member functions have
  /// default arguments, e.g.,
  /// @code
  ///   class X {
  ///   public:
  ///     void f(int x = 17); // x has an unparsed default argument now
  ///   }; // x has a regular default argument now
  /// @endcode
  bool hasUnparsedDefaultArg() const {
    return Init.is<UnparsedDefaultArgument*>();
  }

  /// setUnparsedDefaultArg - Specify that this parameter has an
  /// unparsed default argument. The argument will be replaced with a
  /// real default argument via setDefaultArg when the class
  /// definition enclosing the function declaration that owns this
  /// default argument is completed.
  void setUnparsedDefaultArg() { 
    Init = (UnparsedDefaultArgument *)0;
  }

  QualType getOriginalType() const;
  
  /// setOwningFunction - Sets the function declaration that owns this
  /// ParmVarDecl. Since ParmVarDecls are often created before the
  /// FunctionDecls that own them, this routine is required to update
  /// the DeclContext appropriately.
  void setOwningFunction(DeclContext *FD) { setDeclContext(FD); }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { 
    return (D->getKind() == ParmVar ||
            D->getKind() == OriginalParmVar); 
  }
  static bool classof(const ParmVarDecl *D) { return true; }
};

/// OriginalParmVarDecl - Represent a parameter to a function, when
/// the type of the parameter has been promoted. This node represents the
/// parameter to the function with its original type.
///
class OriginalParmVarDecl : public ParmVarDecl {
  friend class ParmVarDecl;
protected:
  QualType OriginalType;
private:
  OriginalParmVarDecl(DeclContext *DC, SourceLocation L,
                              IdentifierInfo *Id, QualType T,
                              DeclaratorInfo *DInfo, 
                              QualType OT, StorageClass S,
                              Expr *DefArg)
  : ParmVarDecl(OriginalParmVar, DC, L, Id, T, DInfo, S, DefArg),
    OriginalType(OT) {}
public:
  static OriginalParmVarDecl *Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L,IdentifierInfo *Id,
                                     QualType T, DeclaratorInfo *DInfo,
                                     QualType OT, StorageClass S, Expr *DefArg);

  void setOriginalType(QualType T) { OriginalType = T; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == OriginalParmVar; }
  static bool classof(const OriginalParmVarDecl *D) { return true; }
};
  
/// FunctionDecl - An instance of this class is created to represent a
/// function declaration or definition. 
///
/// Since a given function can be declared several times in a program,
/// there may be several FunctionDecls that correspond to that
/// function. Only one of those FunctionDecls will be found when
/// traversing the list of declarations in the context of the
/// FunctionDecl (e.g., the translation unit); this FunctionDecl
/// contains all of the information known about the function. Other,
/// previous declarations of the function are available via the
/// getPreviousDeclaration() chain. 
class FunctionDecl : public DeclaratorDecl, public DeclContext,
                     public Redeclarable<FunctionDecl> {
public:
  enum StorageClass {
    None, Extern, Static, PrivateExtern
  };
  
private:  
  /// ParamInfo - new[]'d array of pointers to VarDecls for the formal
  /// parameters of this function.  This is null if a prototype or if there are
  /// no formals.
  ParmVarDecl **ParamInfo;
  
  LazyDeclStmtPtr Body;

  // FIXME: This can be packed into the bitfields in Decl.
  // NOTE: VC++ treats enums as signed, avoid using the StorageClass enum
  unsigned SClass : 2;
  bool IsInline : 1;
  bool C99InlineDefinition : 1;
  bool IsVirtualAsWritten : 1;
  bool IsPure : 1;
  bool HasInheritedPrototype : 1;
  bool HasWrittenPrototype : 1;
  bool IsDeleted : 1;
  bool IsTrivial : 1; // sunk from CXXMethodDecl
  bool IsCopyAssignment : 1;  // sunk from CXXMethodDecl
  bool HasImplicitReturnZero : 1;
  
  /// \brief End part of this FunctionDecl's source range.
  ///
  /// We could compute the full range in getSourceRange(). However, when we're
  /// dealing with a function definition deserialized from a PCH/AST file,
  /// we can only compute the full range once the function body has been
  /// de-serialized, so it's far better to have the (sometimes-redundant)
  /// EndRangeLoc.
  SourceLocation EndRangeLoc;

  /// \brief The template or declaration that this declaration
  /// describes or was instantiated from, respectively.
  /// 
  /// For non-templates, this value will be NULL. For function
  /// declarations that describe a function template, this will be a
  /// pointer to a FunctionTemplateDecl. For member functions
  /// of class template specializations, this will be the
  /// FunctionDecl from which the member function was instantiated.
  /// For function template specializations, this will be a 
  /// FunctionTemplateSpecializationInfo, which contains information about
  /// the template being specialized and the template arguments involved in 
  /// that specialization.
  llvm::PointerUnion3<FunctionTemplateDecl*, FunctionDecl*,
                      FunctionTemplateSpecializationInfo*>
    TemplateOrSpecialization;

protected:
  FunctionDecl(Kind DK, DeclContext *DC, SourceLocation L,
               DeclarationName N, QualType T, DeclaratorInfo *DInfo,
               StorageClass S, bool isInline)
    : DeclaratorDecl(DK, DC, L, N, T, DInfo), 
      DeclContext(DK),
      ParamInfo(0), Body(),
      SClass(S), IsInline(isInline), C99InlineDefinition(false), 
      IsVirtualAsWritten(false), IsPure(false), HasInheritedPrototype(false), 
      HasWrittenPrototype(true), IsDeleted(false), IsTrivial(false),
      IsCopyAssignment(false),
      HasImplicitReturnZero(false),
      EndRangeLoc(L), TemplateOrSpecialization() {}

  virtual ~FunctionDecl() {}
  virtual void Destroy(ASTContext& C);

  typedef Redeclarable<FunctionDecl> redeclarable_base;
  virtual FunctionDecl *getNextRedeclaration() { return RedeclLink.getNext(); }

public:
  typedef redeclarable_base::redecl_iterator redecl_iterator;
  redecl_iterator redecls_begin() const {
    return redeclarable_base::redecls_begin();
  } 
  redecl_iterator redecls_end() const {
    return redeclarable_base::redecls_end();
  }

  static FunctionDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                              DeclarationName N, QualType T,
                              DeclaratorInfo *DInfo,
                              StorageClass S = None, bool isInline = false,
                              bool hasWrittenPrototype = true);

  virtual SourceRange getSourceRange() const {
    return SourceRange(getLocation(), EndRangeLoc);
  }
  void setLocEnd(SourceLocation E) {
    EndRangeLoc = E;
  }

  /// getBody - Retrieve the body (definition) of the function. The
  /// function body might be in any of the (re-)declarations of this
  /// function. The variant that accepts a FunctionDecl pointer will
  /// set that function declaration to the actual declaration
  /// containing the body (if there is one).
  Stmt *getBody(const FunctionDecl *&Definition) const;

  virtual Stmt *getBody() const {
    const FunctionDecl* Definition;
    return getBody(Definition);
  }

  /// \brief If the function has a body that is immediately available,
  /// return it.
  Stmt *getBodyIfAvailable() const;

  /// isThisDeclarationADefinition - Returns whether this specific
  /// declaration of the function is also a definition. This does not
  /// determine whether the function has been defined (e.g., in a
  /// previous definition); for that information, use getBody.
  /// FIXME: Should return true if function is deleted or defaulted. However,
  /// CodeGenModule.cpp uses it, and I don't know if this would break it.
  bool isThisDeclarationADefinition() const { return Body; }

  void setBody(Stmt *B);
  void setLazyBody(uint64_t Offset) { Body = Offset; }

  /// Whether this function is marked as virtual explicitly.
  bool isVirtualAsWritten() const { return IsVirtualAsWritten; }
  void setVirtualAsWritten(bool V) { IsVirtualAsWritten = V; }

  /// Whether this virtual function is pure, i.e. makes the containing class
  /// abstract.
  bool isPure() const { return IsPure; }
  void setPure(bool P = true) { IsPure = P; }

  /// Whether this function is "trivial" in some specialized C++ senses.
  /// Can only be true for default constructors, copy constructors,
  /// copy assignment operators, and destructors.  Not meaningful until
  /// the class has been fully built by Sema.
  bool isTrivial() const { return IsTrivial; }
  void setTrivial(bool IT) { IsTrivial = IT; }
                       
  bool isCopyAssignment() const { return IsCopyAssignment; }
  void setCopyAssignment(bool CA) { IsCopyAssignment = CA; }

  /// Whether falling off this function implicitly returns null/zero.
  /// If a more specific implicit return value is required, front-ends
  /// should synthesize the appropriate return statements.
  bool hasImplicitReturnZero() const { return HasImplicitReturnZero; }
  void setHasImplicitReturnZero(bool IRZ) { HasImplicitReturnZero = IRZ; }

  /// \brief Whether this function has a prototype, either because one
  /// was explicitly written or because it was "inherited" by merging
  /// a declaration without a prototype with a declaration that has a
  /// prototype.
  bool hasPrototype() const { 
    return HasWrittenPrototype || HasInheritedPrototype; 
  }
  
  bool hasWrittenPrototype() const { return HasWrittenPrototype; }
  void setHasWrittenPrototype(bool P) { HasWrittenPrototype = P; }

  /// \brief Whether this function inherited its prototype from a
  /// previous declaration.
  bool hasInheritedPrototype() const { return HasInheritedPrototype; }
  void setHasInheritedPrototype(bool P = true) { HasInheritedPrototype = P; }

  /// \brief Whether this function has been deleted.
  ///
  /// A function that is "deleted" (via the C++0x "= delete" syntax)
  /// acts like a normal function, except that it cannot actually be
  /// called or have its address taken. Deleted functions are
  /// typically used in C++ overload resolution to attract arguments
  /// whose type or lvalue/rvalue-ness would permit the use of a
  /// different overload that would behave incorrectly. For example,
  /// one might use deleted functions to ban implicit conversion from
  /// a floating-point number to an Integer type:
  ///
  /// @code
  /// struct Integer {
  ///   Integer(long); // construct from a long
  ///   Integer(double) = delete; // no construction from float or double
  ///   Integer(long double) = delete; // no construction from long double
  /// };
  /// @endcode
  bool isDeleted() const { return IsDeleted; }
  void setDeleted(bool D = true) { IsDeleted = D; }

  /// \brief Determines whether this is a function "main", which is
  /// the entry point into an executable program.
  bool isMain(ASTContext &Context) const;

  /// \brief Determines whether this function is a function with
  /// external, C linkage.
  bool isExternC(ASTContext &Context) const;

  /// \brief Determines whether this is a global function.
  bool isGlobal() const;

  void setPreviousDeclaration(FunctionDecl * PrevDecl);

  virtual FunctionDecl *getCanonicalDecl();

  unsigned getBuiltinID(ASTContext &Context) const;

  unsigned getNumParmVarDeclsFromType() const;
  
  // Iterator access to formal parameters.
  unsigned param_size() const { return getNumParams(); }
  typedef ParmVarDecl **param_iterator;
  typedef ParmVarDecl * const *param_const_iterator;
  
  param_iterator param_begin() { return ParamInfo; }
  param_iterator param_end()   { return ParamInfo+param_size(); }
  
  param_const_iterator param_begin() const { return ParamInfo; }
  param_const_iterator param_end() const   { return ParamInfo+param_size(); }
  
  /// getNumParams - Return the number of parameters this function must have
  /// based on its functiontype.  This is the length of the PararmInfo array
  /// after it has been created.
  unsigned getNumParams() const;
  
  const ParmVarDecl *getParamDecl(unsigned i) const {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  ParmVarDecl *getParamDecl(unsigned i) {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  void setParams(ASTContext& C, ParmVarDecl **NewParamInfo, unsigned NumParams);

  /// getMinRequiredArguments - Returns the minimum number of arguments
  /// needed to call this function. This may be fewer than the number of
  /// function parameters, if some of the parameters have default
  /// arguments (in C++).
  unsigned getMinRequiredArguments() const;

  QualType getResultType() const { 
    return getType()->getAsFunctionType()->getResultType();
  }
  StorageClass getStorageClass() const { return StorageClass(SClass); }
  void setStorageClass(StorageClass SC) { SClass = SC; }

  bool isInline() const { return IsInline; }
  void setInline(bool I) { IsInline = I; }

  /// \brief Whether this function is an "inline definition" as
  /// defined by C99.
  bool isC99InlineDefinition() const { return C99InlineDefinition; }
  void setC99InlineDefinition(bool I) { C99InlineDefinition = I; }

  /// \brief Determines whether this function has a gnu_inline
  /// attribute that affects its semantics.
  ///
  /// The gnu_inline attribute only introduces GNU inline semantics
  /// when all of the inline declarations of the function are marked
  /// gnu_inline.
  bool hasActiveGNUInlineAttribute(ASTContext &Context) const;

  /// \brief Determines whether this function is a GNU "extern
  /// inline", which is roughly the opposite of a C99 "extern inline"
  /// function.
  bool isExternGNUInline(ASTContext &Context) const;

  /// isOverloadedOperator - Whether this function declaration
  /// represents an C++ overloaded operator, e.g., "operator+".
  bool isOverloadedOperator() const { 
    return getOverloadedOperator() != OO_None;
  };

  OverloadedOperatorKind getOverloadedOperator() const;

  /// \brief If this function is an instantiation of a member function
  /// of a class template specialization, retrieves the function from
  /// which it was instantiated.
  ///
  /// This routine will return non-NULL for (non-templated) member
  /// functions of class templates and for instantiations of function
  /// templates. For example, given:
  ///
  /// \code
  /// template<typename T>
  /// struct X {
  ///   void f(T);
  /// };
  /// \endcode
  ///
  /// The declaration for X<int>::f is a (non-templated) FunctionDecl
  /// whose parent is the class template specialization X<int>. For
  /// this declaration, getInstantiatedFromFunction() will return
  /// the FunctionDecl X<T>::A. When a complete definition of
  /// X<int>::A is required, it will be instantiated from the
  /// declaration returned by getInstantiatedFromMemberFunction().
  FunctionDecl *getInstantiatedFromMemberFunction() const {
    return TemplateOrSpecialization.dyn_cast<FunctionDecl*>();
  }

  /// \brief Specify that this record is an instantiation of the
  /// member function RD.
  void setInstantiationOfMemberFunction(FunctionDecl *RD) { 
    TemplateOrSpecialization = RD;
  }

  /// \brief Retrieves the function template that is described by this
  /// function declaration.
  ///
  /// Every function template is represented as a FunctionTemplateDecl
  /// and a FunctionDecl (or something derived from FunctionDecl). The
  /// former contains template properties (such as the template
  /// parameter lists) while the latter contains the actual
  /// description of the template's
  /// contents. FunctionTemplateDecl::getTemplatedDecl() retrieves the
  /// FunctionDecl that describes the function template,
  /// getDescribedFunctionTemplate() retrieves the
  /// FunctionTemplateDecl from a FunctionDecl.
  FunctionTemplateDecl *getDescribedFunctionTemplate() const {
    return TemplateOrSpecialization.dyn_cast<FunctionTemplateDecl*>();
  }

  void setDescribedFunctionTemplate(FunctionTemplateDecl *Template) {
    TemplateOrSpecialization = Template;
  }

  /// \brief Retrieve the primary template that this function template
  /// specialization either specializes or was instantiated from.
  ///
  /// If this function declaration is not a function template specialization,
  /// returns NULL.
  FunctionTemplateDecl *getPrimaryTemplate() const;
  
  /// \brief Retrieve the template arguments used to produce this function
  /// template specialization from the primary template.
  ///
  /// If this function declaration is not a function template specialization,
  /// returns NULL.
  const TemplateArgumentList *getTemplateSpecializationArgs() const;  
  
  /// \brief Specify that this function declaration is actually a function
  /// template specialization.
  ///
  /// \param Context the AST context in which this function resides.
  ///
  /// \param Template the function template that this function template
  /// specialization specializes.
  ///
  /// \param TemplateArgs the template arguments that produced this
  /// function template specialization from the template.
  void setFunctionTemplateSpecialization(ASTContext &Context,
                                         FunctionTemplateDecl *Template,
                                      const TemplateArgumentList *TemplateArgs,
                                         void *InsertPos);

  /// \brief Determine whether this is an explicit specialization of a 
  /// function template or a member function of a class template.
  bool isExplicitSpecialization() const;

  /// \brief Note that this is an explicit specialization of a function template
  /// or a member function of a class template.
  void setExplicitSpecialization(bool ES);
                                 
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= FunctionFirst && D->getKind() <= FunctionLast;
  }
  static bool classof(const FunctionDecl *D) { return true; }
  static DeclContext *castToDeclContext(const FunctionDecl *D) {
    return static_cast<DeclContext *>(const_cast<FunctionDecl*>(D));
  }
  static FunctionDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<FunctionDecl *>(const_cast<DeclContext*>(DC));
  }
};


/// FieldDecl - An instance of this class is created by Sema::ActOnField to 
/// represent a member of a struct/union/class.
class FieldDecl : public DeclaratorDecl {
  // FIXME: This can be packed into the bitfields in Decl.
  bool Mutable : 1;
  Expr *BitWidth;
protected:
  FieldDecl(Kind DK, DeclContext *DC, SourceLocation L, 
            IdentifierInfo *Id, QualType T, DeclaratorInfo *DInfo,
            Expr *BW, bool Mutable)
    : DeclaratorDecl(DK, DC, L, Id, T, DInfo), Mutable(Mutable), BitWidth(BW)
      { }

public:
  static FieldDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L, 
                           IdentifierInfo *Id, QualType T,
                           DeclaratorInfo *DInfo, Expr *BW, bool Mutable);

  /// isMutable - Determines whether this field is mutable (C++ only).
  bool isMutable() const { return Mutable; }

  /// \brief Set whether this field is mutable (C++ only).
  void setMutable(bool M) { Mutable = M; }

  /// isBitfield - Determines whether this field is a bitfield.
  bool isBitField() const { return BitWidth != NULL; }

  /// @brief Determines whether this is an unnamed bitfield.
  bool isUnnamedBitfield() const { return BitWidth != NULL && !getDeclName(); }

  /// isAnonymousStructOrUnion - Determines whether this field is a
  /// representative for an anonymous struct or union. Such fields are
  /// unnamed and are implicitly generated by the implementation to
  /// store the data for the anonymous union or struct.
  bool isAnonymousStructOrUnion() const;

  Expr *getBitWidth() const { return BitWidth; }
  void setBitWidth(Expr *BW) { BitWidth = BW; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= FieldFirst && D->getKind() <= FieldLast;
  }
  static bool classof(const FieldDecl *D) { return true; }
};

/// EnumConstantDecl - An instance of this object exists for each enum constant
/// that is defined.  For example, in "enum X {a,b}", each of a/b are
/// EnumConstantDecl's, X is an instance of EnumDecl, and the type of a/b is a
/// TagType for the X EnumDecl.
class EnumConstantDecl : public ValueDecl {
  Stmt *Init; // an integer constant expression
  llvm::APSInt Val; // The value.
protected:
  EnumConstantDecl(DeclContext *DC, SourceLocation L,
                   IdentifierInfo *Id, QualType T, Expr *E,
                   const llvm::APSInt &V)
    : ValueDecl(EnumConstant, DC, L, Id, T), Init((Stmt*)E), Val(V) {}

  virtual ~EnumConstantDecl() {}
public:

  static EnumConstantDecl *Create(ASTContext &C, EnumDecl *DC,
                                  SourceLocation L, IdentifierInfo *Id,
                                  QualType T, Expr *E,
                                  const llvm::APSInt &V);
  
  virtual void Destroy(ASTContext& C);

  const Expr *getInitExpr() const { return (const Expr*) Init; }
  Expr *getInitExpr() { return (Expr*) Init; }
  const llvm::APSInt &getInitVal() const { return Val; }

  void setInitExpr(Expr *E) { Init = (Stmt*) E; }
  void setInitVal(const llvm::APSInt &V) { Val = V; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == EnumConstant; }
  static bool classof(const EnumConstantDecl *D) { return true; }
  
  friend class StmtIteratorBase;
};


/// TypeDecl - Represents a declaration of a type.
///
class TypeDecl : public NamedDecl {
  /// TypeForDecl - This indicates the Type object that represents
  /// this TypeDecl.  It is a cache maintained by
  /// ASTContext::getTypedefType, ASTContext::getTagDeclType, and
  /// ASTContext::getTemplateTypeParmType, and TemplateTypeParmDecl.
  mutable Type *TypeForDecl;
  friend class ASTContext;
  friend class DeclContext;
  friend class TagDecl;
  friend class TemplateTypeParmDecl;
  friend class ClassTemplateSpecializationDecl;
  friend class TagType;

protected:
  TypeDecl(Kind DK, DeclContext *DC, SourceLocation L,
           IdentifierInfo *Id)
    : NamedDecl(DK, DC, L, Id), TypeForDecl(0) {}

public:
  // Low-level accessor
  Type *getTypeForDecl() const { return TypeForDecl; }
  void setTypeForDecl(Type *TD) { TypeForDecl = TD; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= TypeFirst && D->getKind() <= TypeLast;
  }
  static bool classof(const TypeDecl *D) { return true; }
};


class TypedefDecl : public TypeDecl {
  /// UnderlyingType - This is the type the typedef is set to.
  QualType UnderlyingType;
  TypedefDecl(DeclContext *DC, SourceLocation L,
              IdentifierInfo *Id, QualType T) 
    : TypeDecl(Typedef, DC, L, Id), UnderlyingType(T) {}

  virtual ~TypedefDecl() {}
public:
  
  static TypedefDecl *Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L,IdentifierInfo *Id,
                             QualType T);
  
  QualType getUnderlyingType() const { return UnderlyingType; }
  void setUnderlyingType(QualType newType) { UnderlyingType = newType; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Typedef; }
  static bool classof(const TypedefDecl *D) { return true; }
};

class TypedefDecl;
  
/// TagDecl - Represents the declaration of a struct/union/class/enum.
class TagDecl 
  : public TypeDecl, public DeclContext, public Redeclarable<TagDecl> {
public:
  enum TagKind {
    TK_struct,
    TK_union,
    TK_class,
    TK_enum
  };

private:
  // FIXME: This can be packed into the bitfields in Decl.
  /// TagDeclKind - The TagKind enum.
  unsigned TagDeclKind : 2;

  /// IsDefinition - True if this is a definition ("struct foo {};"), false if
  /// it is a declaration ("struct foo;").
  bool IsDefinition : 1;
  
  /// TypedefForAnonDecl - If a TagDecl is anonymous and part of a typedef,
  /// this points to the TypedefDecl. Used for mangling.
  TypedefDecl *TypedefForAnonDecl;

  SourceLocation TagKeywordLoc;
  SourceLocation RBraceLoc;

protected:
  TagDecl(Kind DK, TagKind TK, DeclContext *DC, SourceLocation L,
          IdentifierInfo *Id, TagDecl *PrevDecl,
          SourceLocation TKL = SourceLocation())
    : TypeDecl(DK, DC, L, Id), DeclContext(DK), TypedefForAnonDecl(0),
      TagKeywordLoc(TKL) {
    assert((DK != Enum || TK == TK_enum) &&"EnumDecl not matched with TK_enum");
    TagDeclKind = TK;
    IsDefinition = false;
    setPreviousDeclaration(PrevDecl);
  }
    
  typedef Redeclarable<TagDecl> redeclarable_base;
  virtual TagDecl *getNextRedeclaration() { return RedeclLink.getNext(); }
    
public:
  typedef redeclarable_base::redecl_iterator redecl_iterator;
  redecl_iterator redecls_begin() const {
    return redeclarable_base::redecls_begin();
  } 
  redecl_iterator redecls_end() const {
    return redeclarable_base::redecls_end();
  }
    
  SourceLocation getRBraceLoc() const { return RBraceLoc; }
  void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }

  SourceLocation getTagKeywordLoc() const { return TagKeywordLoc; }
  void setTagKeywordLoc(SourceLocation TKL) { TagKeywordLoc = TKL; }

  virtual SourceRange getSourceRange() const;
  
  virtual TagDecl* getCanonicalDecl();

  /// isDefinition - Return true if this decl has its body specified.
  bool isDefinition() const {
    return IsDefinition;
  }

  /// \brief Whether this declaration declares a type that is
  /// dependent, i.e., a type that somehow depends on template
  /// parameters.
  bool isDependentType() const { return isDependentContext(); }

  /// @brief Starts the definition of this tag declaration.
  /// 
  /// This method should be invoked at the beginning of the definition
  /// of this tag declaration. It will set the tag type into a state
  /// where it is in the process of being defined.
  void startDefinition();

  /// @brief Completes the definition of this tag declaration.
  void completeDefinition();

  /// getDefinition - Returns the TagDecl that actually defines this 
  ///  struct/union/class/enum.  When determining whether or not a
  ///  struct/union/class/enum is completely defined, one should use this method
  ///  as opposed to 'isDefinition'.  'isDefinition' indicates whether or not a
  ///  specific TagDecl is defining declaration, not whether or not the
  ///  struct/union/class/enum type is defined.  This method returns NULL if
  ///  there is no TagDecl that defines the struct/union/class/enum.
  TagDecl* getDefinition(ASTContext& C) const;
  
  const char *getKindName() const {
    switch (getTagKind()) {
    default: assert(0 && "Unknown TagKind!");
    case TK_struct: return "struct";
    case TK_union:  return "union";
    case TK_class:  return "class";
    case TK_enum:   return "enum";
    }
  }

  TagKind getTagKind() const {
    return TagKind(TagDeclKind);
  }

  void setTagKind(TagKind TK) { TagDeclKind = TK; }

  bool isStruct() const { return getTagKind() == TK_struct; }
  bool isClass()  const { return getTagKind() == TK_class; }
  bool isUnion()  const { return getTagKind() == TK_union; }
  bool isEnum()   const { return getTagKind() == TK_enum; }
  
  TypedefDecl *getTypedefForAnonDecl() const { return TypedefForAnonDecl; }
  void setTypedefForAnonDecl(TypedefDecl *TDD) { TypedefForAnonDecl = TDD; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= TagFirst && D->getKind() <= TagLast;
  }
  static bool classof(const TagDecl *D) { return true; }

  static DeclContext *castToDeclContext(const TagDecl *D) {
    return static_cast<DeclContext *>(const_cast<TagDecl*>(D));
  }
  static TagDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<TagDecl *>(const_cast<DeclContext*>(DC));
  }

  void setDefinition(bool V) { IsDefinition = V; }
};

/// EnumDecl - Represents an enum.  As an extension, we allow forward-declared
/// enums.
class EnumDecl : public TagDecl {
  /// IntegerType - This represent the integer type that the enum corresponds
  /// to for code generation purposes.  Note that the enumerator constants may
  /// have a different type than this does.
  QualType IntegerType;

  /// \brief If the enumeration was instantiated from an enumeration
  /// within a class or function template, this pointer refers to the
  /// enumeration declared within the template.
  EnumDecl *InstantiatedFrom;

  EnumDecl(DeclContext *DC, SourceLocation L,
           IdentifierInfo *Id, EnumDecl *PrevDecl, SourceLocation TKL)
    : TagDecl(Enum, TK_enum, DC, L, Id, PrevDecl, TKL), InstantiatedFrom(0) {
      IntegerType = QualType();
    }
public:
  static EnumDecl *Create(ASTContext &C, DeclContext *DC,
                          SourceLocation L, IdentifierInfo *Id,
                          SourceLocation TKL, EnumDecl *PrevDecl);
  
  virtual void Destroy(ASTContext& C);

  /// completeDefinition - When created, the EnumDecl corresponds to a
  /// forward-declared enum. This method is used to mark the
  /// declaration as being defined; it's enumerators have already been
  /// added (via DeclContext::addDecl). NewType is the new underlying
  /// type of the enumeration type.
  void completeDefinition(ASTContext &C, QualType NewType);
  
  // enumerator_iterator - Iterates through the enumerators of this
  // enumeration.
  typedef specific_decl_iterator<EnumConstantDecl> enumerator_iterator;

  enumerator_iterator enumerator_begin() const { 
    return enumerator_iterator(this->decls_begin());
  }

  enumerator_iterator enumerator_end() const { 
    return enumerator_iterator(this->decls_end());
  }

  /// getIntegerType - Return the integer type this enum decl corresponds to.
  /// This returns a null qualtype for an enum forward definition.
  QualType getIntegerType() const { return IntegerType; }

  /// \brief Set the underlying integer type.
  void setIntegerType(QualType T) { IntegerType = T; }

  /// \brief Returns the enumeration (declared within the template)
  /// from which this enumeration type was instantiated, or NULL if
  /// this enumeration was not instantiated from any template.
  EnumDecl *getInstantiatedFromMemberEnum() const {
    return InstantiatedFrom;
  }

  void setInstantiationOfMemberEnum(EnumDecl *IF) { InstantiatedFrom = IF; }

  static bool classof(const Decl *D) { return D->getKind() == Enum; }
  static bool classof(const EnumDecl *D) { return true; }
};


/// RecordDecl - Represents a struct/union/class.  For example:
///   struct X;                  // Forward declaration, no "body".
///   union Y { int A, B; };     // Has body with members A and B (FieldDecls).
/// This decl will be marked invalid if *any* members are invalid.
///
class RecordDecl : public TagDecl {
  // FIXME: This can be packed into the bitfields in Decl.
  /// HasFlexibleArrayMember - This is true if this struct ends with a flexible
  /// array member (e.g. int X[]) or if this union contains a struct that does.
  /// If so, this cannot be contained in arrays or other structs as a member.
  bool HasFlexibleArrayMember : 1;

  /// AnonymousStructOrUnion - Whether this is the type of an
  /// anonymous struct or union.
  bool AnonymousStructOrUnion : 1;
  
  /// HasObjectMember - This is true if this struct has at least one
  /// member containing an object 
  bool HasObjectMember : 1;

protected:
  RecordDecl(Kind DK, TagKind TK, DeclContext *DC,
             SourceLocation L, IdentifierInfo *Id, 
             RecordDecl *PrevDecl, SourceLocation TKL);
  virtual ~RecordDecl();

public:
  static RecordDecl *Create(ASTContext &C, TagKind TK, DeclContext *DC,
                            SourceLocation L, IdentifierInfo *Id,
                            SourceLocation TKL = SourceLocation(),
                            RecordDecl* PrevDecl = 0);

  virtual void Destroy(ASTContext& C);
      
  bool hasFlexibleArrayMember() const { return HasFlexibleArrayMember; }
  void setHasFlexibleArrayMember(bool V) { HasFlexibleArrayMember = V; }

  /// isAnonymousStructOrUnion - Whether this is an anonymous struct
  /// or union. To be an anonymous struct or union, it must have been
  /// declared without a name and there must be no objects of this
  /// type declared, e.g.,
  /// @code
  ///   union { int i; float f; };
  /// @endcode   
  /// is an anonymous union but neither of the following are:
  /// @code
  ///  union X { int i; float f; };
  ///  union { int i; float f; } obj;
  /// @endcode
  bool isAnonymousStructOrUnion() const { return AnonymousStructOrUnion; }
  void setAnonymousStructOrUnion(bool Anon) {
    AnonymousStructOrUnion = Anon;
  }

  bool hasObjectMember() const { return HasObjectMember; }
  void setHasObjectMember (bool val) { HasObjectMember = val; }
  
  /// \brief Determines whether this declaration represents the
  /// injected class name.
  ///
  /// The injected class name in C++ is the name of the class that
  /// appears inside the class itself. For example:
  ///
  /// \code
  /// struct C {
  ///   // C is implicitly declared here as a synonym for the class name.
  /// };
  ///
  /// C::C c; // same as "C c;"
  /// \endcode
  bool isInjectedClassName() const;

  /// getDefinition - Returns the RecordDecl that actually defines this 
  ///  struct/union/class.  When determining whether or not a struct/union/class
  ///  is completely defined, one should use this method as opposed to
  ///  'isDefinition'.  'isDefinition' indicates whether or not a specific
  ///  RecordDecl is defining declaration, not whether or not the record
  ///  type is defined.  This method returns NULL if there is no RecordDecl
  ///  that defines the struct/union/tag.
  RecordDecl* getDefinition(ASTContext& C) const {
    return cast_or_null<RecordDecl>(TagDecl::getDefinition(C));
  }
  
  // Iterator access to field members. The field iterator only visits
  // the non-static data members of this class, ignoring any static
  // data members, functions, constructors, destructors, etc.
  typedef specific_decl_iterator<FieldDecl> field_iterator;

  field_iterator field_begin() const {
    return field_iterator(decls_begin());
  }
  field_iterator field_end() const {
    return field_iterator(decls_end());
  }

  // field_empty - Whether there are any fields (non-static data
  // members) in this record.
  bool field_empty() const { 
    return field_begin() == field_end();
  }

  /// completeDefinition - Notes that the definition of this type is
  /// now complete.
  void completeDefinition(ASTContext& C);

  static bool classof(const Decl *D) {
    return D->getKind() >= RecordFirst && D->getKind() <= RecordLast;
  }
  static bool classof(const RecordDecl *D) { return true; }
};

class FileScopeAsmDecl : public Decl {
  StringLiteral *AsmString;
  FileScopeAsmDecl(DeclContext *DC, SourceLocation L, StringLiteral *asmstring)
    : Decl(FileScopeAsm, DC, L), AsmString(asmstring) {}
public:
  static FileScopeAsmDecl *Create(ASTContext &C, DeclContext *DC,
                                  SourceLocation L, StringLiteral *Str);

  const StringLiteral *getAsmString() const { return AsmString; }
  StringLiteral *getAsmString() { return AsmString; }
  void setAsmString(StringLiteral *Asm) { AsmString = Asm; }

  static bool classof(const Decl *D) {
    return D->getKind() == FileScopeAsm;
  }
  static bool classof(const FileScopeAsmDecl *D) { return true; }  
};

/// BlockDecl - This represents a block literal declaration, which is like an
/// unnamed FunctionDecl.  For example:
/// ^{ statement-body }   or   ^(int arg1, float arg2){ statement-body }
///
class BlockDecl : public Decl, public DeclContext {
  // FIXME: This can be packed into the bitfields in Decl.
  bool isVariadic : 1;
  /// ParamInfo - new[]'d array of pointers to ParmVarDecls for the formal
  /// parameters of this function.  This is null if a prototype or if there are
  /// no formals.
  ParmVarDecl **ParamInfo;
  unsigned NumParams;
  
  Stmt *Body;
  
protected:
  BlockDecl(DeclContext *DC, SourceLocation CaretLoc)
    : Decl(Block, DC, CaretLoc), DeclContext(Block), 
      isVariadic(false), ParamInfo(0), NumParams(0), Body(0) {}

  virtual ~BlockDecl();
  virtual void Destroy(ASTContext& C);

public:
  static BlockDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L);

  SourceLocation getCaretLocation() const { return getLocation(); }

  bool IsVariadic() const { return isVariadic; }
  void setIsVariadic(bool value) { isVariadic = value; }
  
  CompoundStmt *getCompoundBody() const { return (CompoundStmt*) Body; }
  Stmt *getBody() const { return (Stmt*) Body; }
  void setBody(CompoundStmt *B) { Body = (Stmt*) B; }

  // Iterator access to formal parameters.
  unsigned param_size() const { return getNumParams(); }
  typedef ParmVarDecl **param_iterator;
  typedef ParmVarDecl * const *param_const_iterator;
  
  bool param_empty() const { return NumParams == 0; }
  param_iterator param_begin()  { return ParamInfo; }
  param_iterator param_end()   { return ParamInfo+param_size(); }
  
  param_const_iterator param_begin() const { return ParamInfo; }
  param_const_iterator param_end() const   { return ParamInfo+param_size(); }
  
  unsigned getNumParams() const;
  const ParmVarDecl *getParamDecl(unsigned i) const {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  ParmVarDecl *getParamDecl(unsigned i) {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }
  void setParams(ASTContext& C, ParmVarDecl **NewParamInfo, unsigned NumParams);
    
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == Block; }
  static bool classof(const BlockDecl *D) { return true; }  
  static DeclContext *castToDeclContext(const BlockDecl *D) {
    return static_cast<DeclContext *>(const_cast<BlockDecl*>(D));
  }
  static BlockDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<BlockDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// Insertion operator for diagnostics.  This allows sending NamedDecl's
/// into a diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           NamedDecl* ND) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(ND), Diagnostic::ak_nameddecl);
  return DB;
}

}  // end namespace clang

#endif
