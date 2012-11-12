//===--- DumpXML.cpp - Detailed XML dumping -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Decl::dumpXML() method, a debugging tool to
//  print a detailed graph of an AST in an unspecified XML format.
//
//  There is no guarantee of stability for this format.
//
//===----------------------------------------------------------------------===//

// Only pay for this in code size in assertions-enabled builds.

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;

#ifndef NDEBUG

namespace {

enum NodeState {
  NS_Attrs, NS_LazyChildren, NS_Children
};

struct Node {
  StringRef Name;
  NodeState State;
  Node(StringRef name) : Name(name), State(NS_Attrs) {}

  bool isDoneWithAttrs() const { return State != NS_Attrs; }
};

template <class Impl> struct XMLDeclVisitor {
#define DISPATCH(NAME, CLASS) \
  static_cast<Impl*>(this)->NAME(static_cast<CLASS*>(D))

  void dispatch(Decl *D) {
    if (D->isUsed())
      static_cast<Impl*>(this)->set("used", "1");
    switch (D->getKind()) {
#define DECL(DERIVED, BASE) \
      case Decl::DERIVED: \
        DISPATCH(dispatch##DERIVED##DeclAttrs, DERIVED##Decl); \
        static_cast<Impl*>(this)->completeAttrs(); \
        DISPATCH(dispatch##DERIVED##DeclChildren, DERIVED##Decl); \
        DISPATCH(dispatch##DERIVED##DeclAsContext, DERIVED##Decl); \
        break;
#define ABSTRACT_DECL(DECL)
#include "clang/AST/DeclNodes.inc"
    }
  }

#define DECL(DERIVED, BASE) \
  void dispatch##DERIVED##DeclAttrs(DERIVED##Decl *D) { \
    DISPATCH(dispatch##BASE##Attrs, BASE); \
    DISPATCH(visit##DERIVED##DeclAttrs, DERIVED##Decl); \
  } \
  void visit##DERIVED##DeclAttrs(DERIVED##Decl *D) {} \
  void dispatch##DERIVED##DeclChildren(DERIVED##Decl *D) { \
    DISPATCH(dispatch##BASE##Children, BASE); \
    DISPATCH(visit##DERIVED##DeclChildren, DERIVED##Decl); \
  } \
  void visit##DERIVED##DeclChildren(DERIVED##Decl *D) {} \
  void dispatch##DERIVED##DeclAsContext(DERIVED##Decl *D) { \
    DISPATCH(dispatch##BASE##AsContext, BASE); \
    DISPATCH(visit##DERIVED##DeclAsContext, DERIVED##Decl); \
  } \
  void visit##DERIVED##DeclAsContext(DERIVED##Decl *D) {}
#include "clang/AST/DeclNodes.inc"

  void dispatchDeclAttrs(Decl *D) {
    DISPATCH(visitDeclAttrs, Decl);
  }
  void visitDeclAttrs(Decl *D) {}

  void dispatchDeclChildren(Decl *D) {
    DISPATCH(visitDeclChildren, Decl);
  }
  void visitDeclChildren(Decl *D) {}

  void dispatchDeclAsContext(Decl *D) {
    DISPATCH(visitDeclAsContext, Decl);
  }
  void visitDeclAsContext(Decl *D) {}

#undef DISPATCH  
};

template <class Impl> struct XMLTypeVisitor {
#define DISPATCH(NAME, CLASS) \
  static_cast<Impl*>(this)->NAME(static_cast<CLASS*>(T))

  void dispatch(Type *T) {
    switch (T->getTypeClass()) {
#define TYPE(DERIVED, BASE) \
      case Type::DERIVED: \
        DISPATCH(dispatch##DERIVED##TypeAttrs, DERIVED##Type); \
        static_cast<Impl*>(this)->completeAttrs(); \
        DISPATCH(dispatch##DERIVED##TypeChildren, DERIVED##Type); \
        break;
#define ABSTRACT_TYPE(DERIVED, BASE)
#include "clang/AST/TypeNodes.def"
    }
  }

#define TYPE(DERIVED, BASE) \
  void dispatch##DERIVED##TypeAttrs(DERIVED##Type *T) { \
    DISPATCH(dispatch##BASE##Attrs, BASE); \
    DISPATCH(visit##DERIVED##TypeAttrs, DERIVED##Type); \
  } \
  void visit##DERIVED##TypeAttrs(DERIVED##Type *T) {} \
  void dispatch##DERIVED##TypeChildren(DERIVED##Type *T) { \
    DISPATCH(dispatch##BASE##Children, BASE); \
    DISPATCH(visit##DERIVED##TypeChildren, DERIVED##Type); \
  } \
  void visit##DERIVED##TypeChildren(DERIVED##Type *T) {}
#include "clang/AST/TypeNodes.def"

  void dispatchTypeAttrs(Type *T) {
    DISPATCH(visitTypeAttrs, Type);
  }
  void visitTypeAttrs(Type *T) {}

  void dispatchTypeChildren(Type *T) {
    DISPATCH(visitTypeChildren, Type);
  }
  void visitTypeChildren(Type *T) {}

#undef DISPATCH  
};

static StringRef getTypeKindName(Type *T) {
  switch (T->getTypeClass()) {
#define TYPE(DERIVED, BASE) case Type::DERIVED: return #DERIVED "Type";
#define ABSTRACT_TYPE(DERIVED, BASE)
#include "clang/AST/TypeNodes.def"
  }

  llvm_unreachable("unknown type kind!");
}

struct XMLDumper : public XMLDeclVisitor<XMLDumper>,
                   public XMLTypeVisitor<XMLDumper> {
  raw_ostream &out;
  ASTContext &Context;
  SmallVector<Node, 16> Stack;
  unsigned Indent;
  explicit XMLDumper(raw_ostream &OS, ASTContext &context)
    : out(OS), Context(context), Indent(0) {}

  void indent() {
    for (unsigned I = Indent; I; --I)
      out << ' ';
  }

  /// Push a new node on the stack.
  void push(StringRef name) {
    if (!Stack.empty()) {
      assert(Stack.back().isDoneWithAttrs());
      if (Stack.back().State == NS_LazyChildren) {
        Stack.back().State = NS_Children;
        out << ">\n";
      }
      Indent++;
      indent();
    }
    Stack.push_back(Node(name));
    out << '<' << name;
  }

  /// Set the given attribute to the given value.
  void set(StringRef attr, StringRef value) {
    assert(!Stack.empty() && !Stack.back().isDoneWithAttrs());
    out << ' ' << attr << '=' << '"' << value << '"'; // TODO: quotation
  }

  /// Finish attributes.
  void completeAttrs() {
    assert(!Stack.empty() && !Stack.back().isDoneWithAttrs());
    Stack.back().State = NS_LazyChildren;
  }

  /// Pop a node.
  void pop() {
    assert(!Stack.empty() && Stack.back().isDoneWithAttrs());
    if (Stack.back().State == NS_LazyChildren) {
      out << "/>\n";
    } else {
      indent();
      out << "</" << Stack.back().Name << ">\n";
    }
    if (Stack.size() > 1) Indent--;
    Stack.pop_back();
  }

  //---- General utilities -------------------------------------------//

  void setPointer(StringRef prop, const void *p) {
    SmallString<10> buffer;
    llvm::raw_svector_ostream os(buffer);
    os << p;
    os.flush();
    set(prop, buffer);
  }

  void setPointer(void *p) {
    setPointer("ptr", p);
  }

  void setInteger(StringRef prop, const llvm::APSInt &v) {
    set(prop, v.toString(10));
  }

  void setInteger(StringRef prop, unsigned n) {
    SmallString<10> buffer;
    llvm::raw_svector_ostream os(buffer);
    os << n;
    os.flush();
    set(prop, buffer);
  }

  void setFlag(StringRef prop, bool flag) {
    if (flag) set(prop, "true");
  }

  void setName(DeclarationName Name) {
    if (!Name)
      return set("name", "");

    // Common case.
    if (Name.isIdentifier())
      return set("name", Name.getAsIdentifierInfo()->getName());

    set("name", Name.getAsString());
  }

  class TemporaryContainer {
    XMLDumper &Dumper;
  public:
    TemporaryContainer(XMLDumper &dumper, StringRef name)
      : Dumper(dumper) {
      Dumper.push(name);
      Dumper.completeAttrs();
    }

    ~TemporaryContainer() {
      Dumper.pop();
    }
  };

  void visitTemplateParameters(TemplateParameterList *L) {
    push("template_parameters");
    completeAttrs();
    for (TemplateParameterList::iterator
           I = L->begin(), E = L->end(); I != E; ++I)
      dispatch(*I);
    pop();
  }

  void visitTemplateArguments(const TemplateArgumentList &L) {
    push("template_arguments");
    completeAttrs();
    for (unsigned I = 0, E = L.size(); I != E; ++I)
      dispatch(L[I]);
    pop();
  }

  /// Visits a reference to the given declaration.
  void visitDeclRef(Decl *D) {
    push(D->getDeclKindName());
    setPointer("ref", D);
    completeAttrs();
    pop();
  }
  void visitDeclRef(StringRef Name, Decl *D) {
    TemporaryContainer C(*this, Name);
    if (D) visitDeclRef(D);
  }

  void dispatch(const TemplateArgument &A) {
    switch (A.getKind()) {
    case TemplateArgument::Null: {
      TemporaryContainer C(*this, "null");
      break;
    }
    case TemplateArgument::Type: {
      dispatch(A.getAsType());
      break;
    }
    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion:
    case TemplateArgument::NullPtr:
      // FIXME: Implement!
      break;
        
    case TemplateArgument::Declaration: {
      visitDeclRef(A.getAsDecl());
      break;
    }
    case TemplateArgument::Integral: {
      push("integer");
      setInteger("value", A.getAsIntegral());
      completeAttrs();
      pop();
      break;
    }
    case TemplateArgument::Expression: {
      dispatch(A.getAsExpr());
      break;
    }
    case TemplateArgument::Pack: {
      for (TemplateArgument::pack_iterator P = A.pack_begin(), 
                                        PEnd = A.pack_end();
           P != PEnd; ++P)
        dispatch(*P);
      break;
    }
    }
  }

  void dispatch(const TemplateArgumentLoc &A) {
    dispatch(A.getArgument());
  }

  //---- Declarations ------------------------------------------------//
  // Calls are made in this order:
  //   # Enter a new node.
  //   push("FieldDecl")
  //
  //   # In this phase, attributes are set on the node.
  //   visitDeclAttrs(D)
  //   visitNamedDeclAttrs(D)
  //   ...
  //   visitFieldDeclAttrs(D)
  //
  //   # No more attributes after this point.
  //   completeAttrs()
  //
  //   # Create "header" child nodes, i.e. those which logically
  //   # belong to the declaration itself.
  //   visitDeclChildren(D)
  //   visitNamedDeclChildren(D)
  //   ...
  //   visitFieldDeclChildren(D)
  //
  //   # Create nodes for the lexical children.
  //   visitDeclAsContext(D)
  //   visitNamedDeclAsContext(D)
  //   ...
  //   visitFieldDeclAsContext(D)
  //
  //   # Finish the node.
  //   pop();
  void dispatch(Decl *D) {
    push(D->getDeclKindName());
    XMLDeclVisitor<XMLDumper>::dispatch(D);
    pop();
  }
  void visitDeclAttrs(Decl *D) {
    setPointer(D);
  }

  /// Visit all the lexical decls in the given context.
  void visitDeclContext(DeclContext *DC) {
    for (DeclContext::decl_iterator
           I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I)
      dispatch(*I);

    // FIXME: point out visible declarations not in lexical context?
  }

  /// Set the "access" attribute on the current node according to the
  /// given specifier.
  void setAccess(AccessSpecifier AS) {
    switch (AS) {
    case AS_public: return set("access", "public");
    case AS_protected: return set("access", "protected");
    case AS_private: return set("access", "private");
    case AS_none: llvm_unreachable("explicit forbidden access");
    }
  }

  template <class T> void visitRedeclarableAttrs(T *D) {
    if (T *Prev = D->getPreviousDecl())
      setPointer("previous", Prev);
  }


  // TranslationUnitDecl
  void visitTranslationUnitDeclAsContext(TranslationUnitDecl *D) {
    visitDeclContext(D);
  }

  // LinkageSpecDecl
  void visitLinkageSpecDeclAttrs(LinkageSpecDecl *D) {
    StringRef lang = "";
    switch (D->getLanguage()) {
    case LinkageSpecDecl::lang_c: lang = "C"; break;
    case LinkageSpecDecl::lang_cxx: lang = "C++"; break;
    }
    set("lang", lang);
  }
  void visitLinkageSpecDeclAsContext(LinkageSpecDecl *D) {
    visitDeclContext(D);
  }

  // NamespaceDecl
  void visitNamespaceDeclAttrs(NamespaceDecl *D) {
    setFlag("inline", D->isInline());
    if (!D->isOriginalNamespace())
      setPointer("original", D->getOriginalNamespace());
  }
  void visitNamespaceDeclAsContext(NamespaceDecl *D) {
    visitDeclContext(D);
  }

  // NamedDecl
  void visitNamedDeclAttrs(NamedDecl *D) {
    setName(D->getDeclName());
  }

  // ValueDecl
  void visitValueDeclChildren(ValueDecl *D) {
    dispatch(D->getType());
  }

  // DeclaratorDecl
  void visitDeclaratorDeclChildren(DeclaratorDecl *D) {
    //dispatch(D->getTypeSourceInfo()->getTypeLoc());
  }

  // VarDecl
  void visitVarDeclAttrs(VarDecl *D) {
    visitRedeclarableAttrs(D);
    if (D->getStorageClass() != SC_None)
      set("storage",
          VarDecl::getStorageClassSpecifierString(D->getStorageClass()));
    StringRef initStyle = "";
    switch (D->getInitStyle()) {
    case VarDecl::CInit: initStyle = "c"; break;
    case VarDecl::CallInit: initStyle = "call"; break;
    case VarDecl::ListInit: initStyle = "list"; break;
    }
    set("initstyle", initStyle);
    setFlag("nrvo", D->isNRVOVariable());
    // TODO: instantiation, etc.
  }
  void visitVarDeclChildren(VarDecl *D) {
    if (D->hasInit()) dispatch(D->getInit());
  }

  // ParmVarDecl?

  // FunctionDecl
  void visitFunctionDeclAttrs(FunctionDecl *D) {
    visitRedeclarableAttrs(D);
    setFlag("pure", D->isPure());
    setFlag("trivial", D->isTrivial());
    setFlag("returnzero", D->hasImplicitReturnZero());
    setFlag("prototype", D->hasWrittenPrototype());
    setFlag("deleted", D->isDeletedAsWritten());
    if (D->getStorageClass() != SC_None)
      set("storage",
          VarDecl::getStorageClassSpecifierString(D->getStorageClass()));
    setFlag("inline", D->isInlineSpecified());
    if (const AsmLabelAttr *ALA = D->getAttr<AsmLabelAttr>())
      set("asmlabel", ALA->getLabel());
    // TODO: instantiation, etc.
  }
  void visitFunctionDeclChildren(FunctionDecl *D) {
    for (FunctionDecl::param_iterator
           I = D->param_begin(), E = D->param_end(); I != E; ++I)
      dispatch(*I);
    for (llvm::ArrayRef<NamedDecl*>::iterator
           I = D->getDeclsInPrototypeScope().begin(), E = D->getDeclsInPrototypeScope().end();
         I != E; ++I)
      dispatch(*I);
    if (D->doesThisDeclarationHaveABody())
      dispatch(D->getBody());
  }

  // CXXMethodDecl ?
  // CXXConstructorDecl ?
  // CXXDestructorDecl ?
  // CXXConversionDecl ?

  void dispatch(CXXCtorInitializer *Init) {
    // TODO
  }

  // FieldDecl
  void visitFieldDeclAttrs(FieldDecl *D) {
    setFlag("mutable", D->isMutable());
  }
  void visitFieldDeclChildren(FieldDecl *D) {
    if (D->isBitField()) {
      TemporaryContainer C(*this, "bitwidth");
      dispatch(D->getBitWidth());
    }
    // TODO: C++0x member initializer
  }

  // EnumConstantDecl
  void visitEnumConstantDeclChildren(EnumConstantDecl *D) {
    // value in any case?
    if (D->getInitExpr()) dispatch(D->getInitExpr());
  }

  // IndirectFieldDecl
  void visitIndirectFieldDeclChildren(IndirectFieldDecl *D) {
    for (IndirectFieldDecl::chain_iterator
           I = D->chain_begin(), E = D->chain_end(); I != E; ++I) {
      NamedDecl *VD = const_cast<NamedDecl*>(*I);
      push(isa<VarDecl>(VD) ? "variable" : "field");
      setPointer("ptr", VD);
      completeAttrs();
      pop();
    }
  }

  // TypeDecl
  void visitTypeDeclAttrs(TypeDecl *D) {
    setPointer("typeptr", D->getTypeForDecl());
  }

  // TypedefDecl
  void visitTypedefDeclAttrs(TypedefDecl *D) {
    visitRedeclarableAttrs<TypedefNameDecl>(D);
  }
  void visitTypedefDeclChildren(TypedefDecl *D) {
    dispatch(D->getTypeSourceInfo()->getTypeLoc());
  }

  // TypeAliasDecl
  void visitTypeAliasDeclAttrs(TypeAliasDecl *D) {
    visitRedeclarableAttrs<TypedefNameDecl>(D);
  }
  void visitTypeAliasDeclChildren(TypeAliasDecl *D) {
    dispatch(D->getTypeSourceInfo()->getTypeLoc());
  }

  // TagDecl
  void visitTagDeclAttrs(TagDecl *D) {
    visitRedeclarableAttrs(D);
  }
  void visitTagDeclAsContext(TagDecl *D) {
    visitDeclContext(D);
  }

  // EnumDecl
  void visitEnumDeclAttrs(EnumDecl *D) {
    setFlag("scoped", D->isScoped());
    setFlag("fixed", D->isFixed());
  }
  void visitEnumDeclChildren(EnumDecl *D) {
    {
      TemporaryContainer C(*this, "promotion_type");
      dispatch(D->getPromotionType());
    }
    {
      TemporaryContainer C(*this, "integer_type");
      dispatch(D->getIntegerType());
    }
  }

  // RecordDecl ?

  void visitCXXRecordDeclChildren(CXXRecordDecl *D) {
    if (!D->isThisDeclarationADefinition()) return;

    for (CXXRecordDecl::base_class_iterator
           I = D->bases_begin(), E = D->bases_end(); I != E; ++I) {
      push("base");
      setAccess(I->getAccessSpecifier());
      completeAttrs();
      dispatch(I->getTypeSourceInfo()->getTypeLoc());
      pop();
    }
  }

  // ClassTemplateSpecializationDecl ?

  // FileScopeAsmDecl ?

  // BlockDecl
  void visitBlockDeclAttrs(BlockDecl *D) {
    setFlag("variadic", D->isVariadic());
  }
  void visitBlockDeclChildren(BlockDecl *D) {
    for (FunctionDecl::param_iterator
           I = D->param_begin(), E = D->param_end(); I != E; ++I)
      dispatch(*I);
    dispatch(D->getBody());
  }

  // AccessSpecDecl
  void visitAccessSpecDeclAttrs(AccessSpecDecl *D) {
    setAccess(D->getAccess());
  }

  // TemplateDecl
  void visitTemplateDeclChildren(TemplateDecl *D) {
    visitTemplateParameters(D->getTemplateParameters());
    if (D->getTemplatedDecl())
      dispatch(D->getTemplatedDecl());
  }

  // FunctionTemplateDecl
  void visitFunctionTemplateDeclAttrs(FunctionTemplateDecl *D) {
    visitRedeclarableAttrs(D);
  }
  void visitFunctionTemplateDeclChildren(FunctionTemplateDecl *D) {
    // Mention all the specializations which don't have explicit
    // declarations elsewhere.
    for (FunctionTemplateDecl::spec_iterator
           I = D->spec_begin(), E = D->spec_end(); I != E; ++I) {
      FunctionTemplateSpecializationInfo *Info
        = I->getTemplateSpecializationInfo();

      bool Unknown = false;
      switch (Info->getTemplateSpecializationKind()) {
      case TSK_ImplicitInstantiation: Unknown = false; break;
      case TSK_Undeclared: Unknown = true; break;

      // These will be covered at their respective sites.
      case TSK_ExplicitSpecialization: continue;
      case TSK_ExplicitInstantiationDeclaration: continue;
      case TSK_ExplicitInstantiationDefinition: continue;
      }

      TemporaryContainer C(*this,
                           Unknown ? "uninstantiated" : "instantiation");
      visitTemplateArguments(*Info->TemplateArguments);
      dispatch(Info->Function);
    }
  }

  // ClasTemplateDecl
  void visitClassTemplateDeclAttrs(ClassTemplateDecl *D) {
    visitRedeclarableAttrs(D);
  }
  void visitClassTemplateDeclChildren(ClassTemplateDecl *D) {
    // Mention all the specializations which don't have explicit
    // declarations elsewhere.
    for (ClassTemplateDecl::spec_iterator
           I = D->spec_begin(), E = D->spec_end(); I != E; ++I) {

      bool Unknown = false;
      switch (I->getTemplateSpecializationKind()) {
      case TSK_ImplicitInstantiation: Unknown = false; break;
      case TSK_Undeclared: Unknown = true; break;

      // These will be covered at their respective sites.
      case TSK_ExplicitSpecialization: continue;
      case TSK_ExplicitInstantiationDeclaration: continue;
      case TSK_ExplicitInstantiationDefinition: continue;
      }

      TemporaryContainer C(*this,
                           Unknown ? "uninstantiated" : "instantiation");
      visitTemplateArguments(I->getTemplateArgs());
      dispatch(*I);
    }
  }

  // TemplateTypeParmDecl
  void visitTemplateTypeParmDeclAttrs(TemplateTypeParmDecl *D) {
    setInteger("depth", D->getDepth());
    setInteger("index", D->getIndex());
  }
  void visitTemplateTypeParmDeclChildren(TemplateTypeParmDecl *D) {
    if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
      dispatch(D->getDefaultArgumentInfo()->getTypeLoc());
    // parameter pack?
  }

  // NonTypeTemplateParmDecl
  void visitNonTypeTemplateParmDeclAttrs(NonTypeTemplateParmDecl *D) {
    setInteger("depth", D->getDepth());
    setInteger("index", D->getIndex());
  }
  void visitNonTypeTemplateParmDeclChildren(NonTypeTemplateParmDecl *D) {
    if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
      dispatch(D->getDefaultArgument());
    // parameter pack?
  }

  // TemplateTemplateParmDecl
  void visitTemplateTemplateParmDeclAttrs(TemplateTemplateParmDecl *D) {
    setInteger("depth", D->getDepth());
    setInteger("index", D->getIndex());
  }
  void visitTemplateTemplateParmDeclChildren(TemplateTemplateParmDecl *D) {
    if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
      dispatch(D->getDefaultArgument());
    // parameter pack?
  }

  // FriendDecl
  void visitFriendDeclChildren(FriendDecl *D) {
    if (TypeSourceInfo *T = D->getFriendType())
      dispatch(T->getTypeLoc());
    else
      dispatch(D->getFriendDecl());
  }

  // UsingDirectiveDecl ?
  // UsingDecl ?
  // UsingShadowDecl ?
  // NamespaceAliasDecl ?
  // UnresolvedUsingValueDecl ?
  // UnresolvedUsingTypenameDecl ?
  // StaticAssertDecl ?

  // ObjCImplDecl
  void visitObjCImplDeclChildren(ObjCImplDecl *D) {
    visitDeclRef(D->getClassInterface());
  }
  void visitObjCImplDeclAsContext(ObjCImplDecl *D) {
    visitDeclContext(D);
  }

  // ObjCInterfaceDecl
  void visitCategoryList(ObjCCategoryDecl *D) {
    if (!D) return;

    TemporaryContainer C(*this, "categories");
    for (; D; D = D->getNextClassCategory())
      visitDeclRef(D);
  }
  void visitObjCInterfaceDeclAttrs(ObjCInterfaceDecl *D) {
    setPointer("typeptr", D->getTypeForDecl());
    setFlag("forward_decl", !D->isThisDeclarationADefinition());
    setFlag("implicit_interface", D->isImplicitInterfaceDecl());
  }
  void visitObjCInterfaceDeclChildren(ObjCInterfaceDecl *D) {
    visitDeclRef("super", D->getSuperClass());
    visitDeclRef("implementation", D->getImplementation());
    if (D->protocol_begin() != D->protocol_end()) {
      TemporaryContainer C(*this, "protocols");
      for (ObjCInterfaceDecl::protocol_iterator
             I = D->protocol_begin(), E = D->protocol_end(); I != E; ++I)
        visitDeclRef(*I);
    }
    visitCategoryList(D->getCategoryList());
  }
  void visitObjCInterfaceDeclAsContext(ObjCInterfaceDecl *D) {
    visitDeclContext(D);
  }

  // ObjCCategoryDecl
  void visitObjCCategoryDeclAttrs(ObjCCategoryDecl *D) {
    setFlag("extension", D->IsClassExtension());
  }
  void visitObjCCategoryDeclChildren(ObjCCategoryDecl *D) {
    visitDeclRef("interface", D->getClassInterface());
    visitDeclRef("implementation", D->getImplementation());
    if (D->protocol_begin() != D->protocol_end()) {
      TemporaryContainer C(*this, "protocols");
      for (ObjCCategoryDecl::protocol_iterator
             I = D->protocol_begin(), E = D->protocol_end(); I != E; ++I)
        visitDeclRef(*I);
    }
  }
  void visitObjCCategoryDeclAsContext(ObjCCategoryDecl *D) {
    visitDeclContext(D);
  }

  // ObjCCategoryImplDecl
  void visitObjCCategoryImplDeclAttrs(ObjCCategoryImplDecl *D) {
    set("identifier", D->getName());
  }
  void visitObjCCategoryImplDeclChildren(ObjCCategoryImplDecl *D) {
    visitDeclRef(D->getCategoryDecl());
  }

  // ObjCImplementationDecl
  void visitObjCImplementationDeclAttrs(ObjCImplementationDecl *D) {
    set("identifier", D->getName());
  }
  void visitObjCImplementationDeclChildren(ObjCImplementationDecl *D) {
    visitDeclRef("super", D->getSuperClass());
    if (D->init_begin() != D->init_end()) {
      TemporaryContainer C(*this, "initializers");
      for (ObjCImplementationDecl::init_iterator
             I = D->init_begin(), E = D->init_end(); I != E; ++I)
        dispatch(*I);
    }
  }

  // ObjCProtocolDecl
  void visitObjCProtocolDeclChildren(ObjCProtocolDecl *D) {
    if (!D->isThisDeclarationADefinition())
      return;
    
    if (D->protocol_begin() != D->protocol_end()) {
      TemporaryContainer C(*this, "protocols");
      for (ObjCInterfaceDecl::protocol_iterator
             I = D->protocol_begin(), E = D->protocol_end(); I != E; ++I)
        visitDeclRef(*I);
    }
  }
  void visitObjCProtocolDeclAsContext(ObjCProtocolDecl *D) {
    if (!D->isThisDeclarationADefinition())
      return;
    
    visitDeclContext(D);
  }

  // ObjCMethodDecl
  void visitObjCMethodDeclAttrs(ObjCMethodDecl *D) {
    // decl qualifier?
    // implementation control?

    setFlag("instance", D->isInstanceMethod());
    setFlag("variadic", D->isVariadic());
    setFlag("property_accessor", D->isPropertyAccessor());
    setFlag("defined", D->isDefined());
    setFlag("related_result_type", D->hasRelatedResultType());
  }
  void visitObjCMethodDeclChildren(ObjCMethodDecl *D) {
    dispatch(D->getResultType());
    for (ObjCMethodDecl::param_iterator
           I = D->param_begin(), E = D->param_end(); I != E; ++I)
      dispatch(*I);
    if (D->isThisDeclarationADefinition())
      dispatch(D->getBody());
  }

  // ObjCIvarDecl
  void setAccessControl(StringRef prop, ObjCIvarDecl::AccessControl AC) {
    switch (AC) {
    case ObjCIvarDecl::None: return set(prop, "none");
    case ObjCIvarDecl::Private: return set(prop, "private");
    case ObjCIvarDecl::Protected: return set(prop, "protected");
    case ObjCIvarDecl::Public: return set(prop, "public");
    case ObjCIvarDecl::Package: return set(prop, "package");
    }
  }
  void visitObjCIvarDeclAttrs(ObjCIvarDecl *D) {
    setFlag("synthesize", D->getSynthesize());
    setAccessControl("access", D->getAccessControl());
  }

  // ObjCCompatibleAliasDecl
  void visitObjCCompatibleAliasDeclChildren(ObjCCompatibleAliasDecl *D) {
    visitDeclRef(D->getClassInterface());
  }

  // FIXME: ObjCPropertyDecl
  // FIXME: ObjCPropertyImplDecl

  //---- Types -----------------------------------------------------//
  void dispatch(TypeLoc TL) {
    dispatch(TL.getType()); // for now
  }

  void dispatch(QualType T) {
    if (T.hasLocalQualifiers()) {
      push("QualType");
      Qualifiers Qs = T.getLocalQualifiers();
      setFlag("const", Qs.hasConst());
      setFlag("volatile", Qs.hasVolatile());
      setFlag("restrict", Qs.hasRestrict());
      if (Qs.hasAddressSpace()) setInteger("addrspace", Qs.getAddressSpace());
      if (Qs.hasObjCGCAttr()) {
        switch (Qs.getObjCGCAttr()) {
        case Qualifiers::Weak: set("gc", "weak"); break;
        case Qualifiers::Strong: set("gc", "strong"); break;
        case Qualifiers::GCNone: llvm_unreachable("explicit none");
        }
      }
      
      completeAttrs();
      dispatch(QualType(T.getTypePtr(), 0));
      pop();
      return;
    }

    Type *Ty = const_cast<Type*>(T.getTypePtr());
    push(getTypeKindName(Ty));
    XMLTypeVisitor<XMLDumper>::dispatch(const_cast<Type*>(T.getTypePtr()));
    pop();
  }

  void setCallingConv(CallingConv CC) {
    switch (CC) {
    case CC_Default: return;
    case CC_C: return set("cc", "cdecl");
    case CC_X86FastCall: return set("cc", "x86_fastcall");
    case CC_X86StdCall: return set("cc", "x86_stdcall");
    case CC_X86ThisCall: return set("cc", "x86_thiscall");
    case CC_X86Pascal: return set("cc", "x86_pascal");
    case CC_AAPCS: return set("cc", "aapcs");
    case CC_AAPCS_VFP: return set("cc", "aapcs_vfp");
    case CC_PnaclCall: return set("cc", "pnaclcall");
    }
  }

  void visitTypeAttrs(Type *D) {
    setPointer(D);
    setFlag("dependent", D->isDependentType());
    setFlag("variably_modified", D->isVariablyModifiedType());

    setPointer("canonical", D->getCanonicalTypeInternal().getAsOpaquePtr());
  }

  void visitPointerTypeChildren(PointerType *T) {
    dispatch(T->getPointeeType());
  }
  void visitReferenceTypeChildren(ReferenceType *T) {
    dispatch(T->getPointeeType());
  }
  void visitObjCObjectPointerTypeChildren(ObjCObjectPointerType *T) {
    dispatch(T->getPointeeType());
  }
  void visitBlockPointerTypeChildren(BlockPointerType *T) {
    dispatch(T->getPointeeType());
  }

  // Types that just wrap declarations.
  void visitTagTypeChildren(TagType *T) {
    visitDeclRef(T->getDecl());
  }
  void visitTypedefTypeChildren(TypedefType *T) {
    visitDeclRef(T->getDecl());
  }
  void visitObjCInterfaceTypeChildren(ObjCInterfaceType *T) {
    visitDeclRef(T->getDecl());
  }
  void visitUnresolvedUsingTypeChildren(UnresolvedUsingType *T) {
    visitDeclRef(T->getDecl());
  }
  void visitInjectedClassNameTypeChildren(InjectedClassNameType *T) {
    visitDeclRef(T->getDecl());
  }

  void visitFunctionTypeAttrs(FunctionType *T) {
    setFlag("noreturn", T->getNoReturnAttr());
    setCallingConv(T->getCallConv());
    if (T->getHasRegParm()) setInteger("regparm", T->getRegParmType());
  }
  void visitFunctionTypeChildren(FunctionType *T) {
    dispatch(T->getResultType());
  }

  void visitFunctionProtoTypeAttrs(FunctionProtoType *T) {
    setFlag("const", T->isConst());
    setFlag("volatile", T->isVolatile());
    setFlag("restrict", T->isRestrict());
    switch (T->getExceptionSpecType()) {
    case EST_None: break;
    case EST_DynamicNone: set("exception_spec", "throw()"); break;
    case EST_Dynamic: set("exception_spec", "throw(T)"); break;
    case EST_MSAny: set("exception_spec", "throw(...)"); break;
    case EST_BasicNoexcept: set("exception_spec", "noexcept"); break;
    case EST_ComputedNoexcept: set("exception_spec", "noexcept(expr)"); break;
    case EST_Unevaluated: set("exception_spec", "unevaluated"); break;
    case EST_Uninstantiated: set("exception_spec", "uninstantiated"); break;
    }
  }
  void visitFunctionProtoTypeChildren(FunctionProtoType *T) {
    push("parameters");
    setFlag("variadic", T->isVariadic());
    completeAttrs();
    for (FunctionProtoType::arg_type_iterator
           I = T->arg_type_begin(), E = T->arg_type_end(); I != E; ++I)
      dispatch(*I);
    pop();

    if (T->hasDynamicExceptionSpec()) {
      push("exception_specifiers");
      setFlag("any", T->getExceptionSpecType() == EST_MSAny);
      completeAttrs();
      for (FunctionProtoType::exception_iterator
             I = T->exception_begin(), E = T->exception_end(); I != E; ++I)
        dispatch(*I);
      pop();
    }
    // FIXME: noexcept specifier
  }

  void visitTemplateSpecializationTypeChildren(TemplateSpecializationType *T) {
    if (const RecordType *RT = T->getAs<RecordType>())
      visitDeclRef(RT->getDecl());

    // TODO: TemplateName

    push("template_arguments");
    completeAttrs();
    for (unsigned I = 0, E = T->getNumArgs(); I != E; ++I)
      dispatch(T->getArg(I));
    pop();
  }

  //---- Statements ------------------------------------------------//
  void dispatch(Stmt *S) {
    // FIXME: this is not really XML at all
    push("Stmt");
    out << ">\n";
    Stack.back().State = NS_Children; // explicitly become non-lazy
    S->dump(out, Context.getSourceManager());
    out << '\n';
    pop();
  }
};
}

void Decl::dumpXML() const {
  dumpXML(llvm::errs());
}

void Decl::dumpXML(raw_ostream &out) const {
  XMLDumper(out, getASTContext()).dispatch(const_cast<Decl*>(this));
}

#else /* ifndef NDEBUG */

void Decl::dumpXML() const {}
void Decl::dumpXML(raw_ostream &out) const {}

#endif
