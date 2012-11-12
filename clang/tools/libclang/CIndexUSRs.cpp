//===- CIndexUSR.cpp - Clang-C Source Indexing Library --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generation and use of USRs from CXEntities.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXCursor.h"
#include "CXString.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::cxstring;

//===----------------------------------------------------------------------===//
// USR generation.
//===----------------------------------------------------------------------===//

namespace {
class USRGenerator : public DeclVisitor<USRGenerator> {
  OwningPtr<SmallString<128> > OwnedBuf;
  SmallVectorImpl<char> &Buf;
  llvm::raw_svector_ostream Out;
  bool IgnoreResults;
  ASTContext *Context;
  bool generatedLoc;
  
  llvm::DenseMap<const Type *, unsigned> TypeSubstitutions;
  
public:
  explicit USRGenerator(ASTContext *Ctx = 0, SmallVectorImpl<char> *extBuf = 0)
  : OwnedBuf(extBuf ? 0 : new SmallString<128>()),
    Buf(extBuf ? *extBuf : *OwnedBuf.get()),
    Out(Buf),
    IgnoreResults(false),
    Context(Ctx),
    generatedLoc(false)
  {
    // Add the USR space prefix.
    Out << "c:";
  }

  StringRef str() {
    return Out.str();
  }

  USRGenerator* operator->() { return this; }

  template <typename T>
  llvm::raw_svector_ostream &operator<<(const T &x) {
    Out << x;
    return Out;
  }

  bool ignoreResults() const { return IgnoreResults; }

  // Visitation methods from generating USRs from AST elements.
  void VisitDeclContext(DeclContext *D);
  void VisitFieldDecl(FieldDecl *D);
  void VisitFunctionDecl(FunctionDecl *D);
  void VisitNamedDecl(NamedDecl *D);
  void VisitNamespaceDecl(NamespaceDecl *D);
  void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
  void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
  void VisitClassTemplateDecl(ClassTemplateDecl *D);
  void VisitObjCContainerDecl(ObjCContainerDecl *CD);
  void VisitObjCMethodDecl(ObjCMethodDecl *MD);
  void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
  void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
  void VisitTagDecl(TagDecl *D);
  void VisitTypedefDecl(TypedefDecl *D);
  void VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
  void VisitVarDecl(VarDecl *D);
  void VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
  void VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
  void VisitLinkageSpecDecl(LinkageSpecDecl *D) {
    IgnoreResults = true;
  }
  void VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
    IgnoreResults = true;
  }
  void VisitUsingDecl(UsingDecl *D) { 
    IgnoreResults = true;
  }
  void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) { 
    IgnoreResults = true;
  }
  void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D) { 
    IgnoreResults = true;
  }
  
  /// Generate the string component containing the location of the
  ///  declaration.
  bool GenLoc(const Decl *D);

  /// String generation methods used both by the visitation methods
  /// and from other clients that want to directly generate USRs.  These
  /// methods do not construct complete USRs (which incorporate the parents
  /// of an AST element), but only the fragments concerning the AST element
  /// itself.

  /// Generate a USR for an Objective-C class.
  void GenObjCClass(StringRef cls);
  /// Generate a USR for an Objective-C class category.
  void GenObjCCategory(StringRef cls, StringRef cat);
  /// Generate a USR fragment for an Objective-C instance variable.  The
  /// complete USR can be created by concatenating the USR for the
  /// encompassing class with this USR fragment.
  void GenObjCIvar(StringRef ivar);
  /// Generate a USR fragment for an Objective-C method.
  void GenObjCMethod(StringRef sel, bool isInstanceMethod);
  /// Generate a USR fragment for an Objective-C property.
  void GenObjCProperty(StringRef prop);
  /// Generate a USR for an Objective-C protocol.
  void GenObjCProtocol(StringRef prot);

  void VisitType(QualType T);
  void VisitTemplateParameterList(const TemplateParameterList *Params);
  void VisitTemplateName(TemplateName Name);
  void VisitTemplateArgument(const TemplateArgument &Arg);
  
  /// Emit a Decl's name using NamedDecl::printName() and return true if
  ///  the decl had no name.
  bool EmitDeclName(const NamedDecl *D);
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Generating USRs from ASTS.
//===----------------------------------------------------------------------===//

bool USRGenerator::EmitDeclName(const NamedDecl *D) {
  Out.flush();
  const unsigned startSize = Buf.size();
  D->printName(Out);
  Out.flush();
  const unsigned endSize = Buf.size();
  return startSize == endSize;
}

static bool InAnonymousNamespace(const Decl *D) {
  if (const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(D->getDeclContext()))
    return ND->isAnonymousNamespace();
  return false;
}

static inline bool ShouldGenerateLocation(const NamedDecl *D) {
  return D->getLinkage() != ExternalLinkage && !InAnonymousNamespace(D);
}

void USRGenerator::VisitDeclContext(DeclContext *DC) {
  if (NamedDecl *D = dyn_cast<NamedDecl>(DC))
    Visit(D);
}

void USRGenerator::VisitFieldDecl(FieldDecl *D) {
  // The USR for an ivar declared in a class extension is based on the
  // ObjCInterfaceDecl, not the ObjCCategoryDecl.
  if (ObjCInterfaceDecl *ID = Context->getObjContainingInterface(D))
    Visit(ID);
  else
    VisitDeclContext(D->getDeclContext());
  Out << (isa<ObjCIvarDecl>(D) ? "@" : "@FI@");
  if (EmitDeclName(D)) {
    // Bit fields can be anonymous.
    IgnoreResults = true;
    return;
  }
}

void USRGenerator::VisitFunctionDecl(FunctionDecl *D) {
  if (ShouldGenerateLocation(D) && GenLoc(D))
    return;

  VisitDeclContext(D->getDeclContext());
  if (FunctionTemplateDecl *FunTmpl = D->getDescribedFunctionTemplate()) {
    Out << "@FT@";
    VisitTemplateParameterList(FunTmpl->getTemplateParameters());
  } else
    Out << "@F@";
  D->printName(Out);

  ASTContext &Ctx = *Context;
  if (!Ctx.getLangOpts().CPlusPlus || D->isExternC())
    return;

  if (const TemplateArgumentList *
        SpecArgs = D->getTemplateSpecializationArgs()) {
    Out << '<';
    for (unsigned I = 0, N = SpecArgs->size(); I != N; ++I) {
      Out << '#';
      VisitTemplateArgument(SpecArgs->get(I));
    }
    Out << '>';
  }

  // Mangle in type information for the arguments.
  for (FunctionDecl::param_iterator I = D->param_begin(), E = D->param_end();
       I != E; ++I) {
    Out << '#';
    if (ParmVarDecl *PD = *I)
      VisitType(PD->getType());
  }
  if (D->isVariadic())
    Out << '.';
  Out << '#';
  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(D)) {
    if (MD->isStatic())
      Out << 'S';
    if (unsigned quals = MD->getTypeQualifiers())
      Out << (char)('0' + quals);
  }
}

void USRGenerator::VisitNamedDecl(NamedDecl *D) {
  VisitDeclContext(D->getDeclContext());
  Out << "@";

  if (EmitDeclName(D)) {
    // The string can be empty if the declaration has no name; e.g., it is
    // the ParmDecl with no name for declaration of a function pointer type,
    // e.g.: void  (*f)(void *);
    // In this case, don't generate a USR.
    IgnoreResults = true;
  }
}

void USRGenerator::VisitVarDecl(VarDecl *D) {
  // VarDecls can be declared 'extern' within a function or method body,
  // but their enclosing DeclContext is the function, not the TU.  We need
  // to check the storage class to correctly generate the USR.
  if (ShouldGenerateLocation(D) && GenLoc(D))
    return;

  VisitDeclContext(D->getDeclContext());

  // Variables always have simple names.
  StringRef s = D->getName();

  // The string can be empty if the declaration has no name; e.g., it is
  // the ParmDecl with no name for declaration of a function pointer type, e.g.:
  //    void  (*f)(void *);
  // In this case, don't generate a USR.
  if (s.empty())
    IgnoreResults = true;
  else
    Out << '@' << s;
}

void USRGenerator::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  GenLoc(D);
  return;
}

void USRGenerator::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  GenLoc(D);
  return;
}

void USRGenerator::VisitNamespaceDecl(NamespaceDecl *D) {
  if (D->isAnonymousNamespace()) {
    Out << "@aN";
    return;
  }

  VisitDeclContext(D->getDeclContext());
  if (!IgnoreResults)
    Out << "@N@" << D->getName();
}

void USRGenerator::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  VisitFunctionDecl(D->getTemplatedDecl());
}

void USRGenerator::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  VisitTagDecl(D->getTemplatedDecl());
}

void USRGenerator::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  VisitDeclContext(D->getDeclContext());
  if (!IgnoreResults)
    Out << "@NA@" << D->getName();  
}

void USRGenerator::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  DeclContext *container = D->getDeclContext();
  if (ObjCProtocolDecl *pd = dyn_cast<ObjCProtocolDecl>(container)) {
    Visit(pd);
  }
  else {
    // The USR for a method declared in a class extension or category is based on
    // the ObjCInterfaceDecl, not the ObjCCategoryDecl.
    ObjCInterfaceDecl *ID = D->getClassInterface();
    if (!ID) {
      IgnoreResults = true;
      return;
    }
    Visit(ID);
  }
  // Ideally we would use 'GenObjCMethod', but this is such a hot path
  // for Objective-C code that we don't want to use
  // DeclarationName::getAsString().
  Out << (D->isInstanceMethod() ? "(im)" : "(cm)");
  DeclarationName N(D->getSelector());
  N.printName(Out);
}

void USRGenerator::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  switch (D->getKind()) {
    default:
      llvm_unreachable("Invalid ObjC container.");
    case Decl::ObjCInterface:
    case Decl::ObjCImplementation:
      GenObjCClass(D->getName());
      break;
    case Decl::ObjCCategory: {
      ObjCCategoryDecl *CD = cast<ObjCCategoryDecl>(D);
      ObjCInterfaceDecl *ID = CD->getClassInterface();
      if (!ID) {
        // Handle invalid code where the @interface might not
        // have been specified.
        // FIXME: We should be able to generate this USR even if the
        // @interface isn't available.
        IgnoreResults = true;
        return;
      }
      // Specially handle class extensions, which are anonymous categories.
      // We want to mangle in the location to uniquely distinguish them.
      if (CD->IsClassExtension()) {
        Out << "objc(ext)" << ID->getName() << '@';
        GenLoc(CD);
      }
      else
        GenObjCCategory(ID->getName(), CD->getName());

      break;
    }
    case Decl::ObjCCategoryImpl: {
      ObjCCategoryImplDecl *CD = cast<ObjCCategoryImplDecl>(D);
      ObjCInterfaceDecl *ID = CD->getClassInterface();
      if (!ID) {
        // Handle invalid code where the @interface might not
        // have been specified.
        // FIXME: We should be able to generate this USR even if the
        // @interface isn't available.
        IgnoreResults = true;
        return;
      }
      GenObjCCategory(ID->getName(), CD->getName());
      break;
    }
    case Decl::ObjCProtocol:
      GenObjCProtocol(cast<ObjCProtocolDecl>(D)->getName());
      break;
  }
}

void USRGenerator::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  // The USR for a property declared in a class extension or category is based
  // on the ObjCInterfaceDecl, not the ObjCCategoryDecl.
  if (ObjCInterfaceDecl *ID = Context->getObjContainingInterface(D))
    Visit(ID);
  else
    Visit(cast<Decl>(D->getDeclContext()));
  GenObjCProperty(D->getName());
}

void USRGenerator::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  if (ObjCPropertyDecl *PD = D->getPropertyDecl()) {
    VisitObjCPropertyDecl(PD);
    return;
  }

  IgnoreResults = true;
}

void USRGenerator::VisitTagDecl(TagDecl *D) {
  // Add the location of the tag decl to handle resolution across
  // translation units.
  if (ShouldGenerateLocation(D) && GenLoc(D))
    return;

  D = D->getCanonicalDecl();
  VisitDeclContext(D->getDeclContext());

  bool AlreadyStarted = false;
  if (CXXRecordDecl *CXXRecord = dyn_cast<CXXRecordDecl>(D)) {
    if (ClassTemplateDecl *ClassTmpl = CXXRecord->getDescribedClassTemplate()) {
      AlreadyStarted = true;
      
      switch (D->getTagKind()) {
      case TTK_Interface:
      case TTK_Struct: Out << "@ST"; break;
      case TTK_Class:  Out << "@CT"; break;
      case TTK_Union:  Out << "@UT"; break;
      case TTK_Enum: llvm_unreachable("enum template");
      }
      VisitTemplateParameterList(ClassTmpl->getTemplateParameters());
    } else if (ClassTemplatePartialSpecializationDecl *PartialSpec
                = dyn_cast<ClassTemplatePartialSpecializationDecl>(CXXRecord)) {
      AlreadyStarted = true;
      
      switch (D->getTagKind()) {
      case TTK_Interface:
      case TTK_Struct: Out << "@SP"; break;
      case TTK_Class:  Out << "@CP"; break;
      case TTK_Union:  Out << "@UP"; break;
      case TTK_Enum: llvm_unreachable("enum partial specialization");
      }      
      VisitTemplateParameterList(PartialSpec->getTemplateParameters());
    }
  }
  
  if (!AlreadyStarted) {
    switch (D->getTagKind()) {
      case TTK_Interface:
      case TTK_Struct: Out << "@S"; break;
      case TTK_Class:  Out << "@C"; break;
      case TTK_Union:  Out << "@U"; break;
      case TTK_Enum:   Out << "@E"; break;
    }
  }
  
  Out << '@';
  Out.flush();
  assert(Buf.size() > 0);
  const unsigned off = Buf.size() - 1;

  if (EmitDeclName(D)) {
    if (const TypedefNameDecl *TD = D->getTypedefNameForAnonDecl()) {
      Buf[off] = 'A';
      Out << '@' << *TD;
    }
    else
      Buf[off] = 'a';
  }
  
  // For a class template specialization, mangle the template arguments.
  if (ClassTemplateSpecializationDecl *Spec
                              = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    const TemplateArgumentList &Args = Spec->getTemplateInstantiationArgs();
    Out << '>';
    for (unsigned I = 0, N = Args.size(); I != N; ++I) {
      Out << '#';
      VisitTemplateArgument(Args.get(I));
    }
  }
}

void USRGenerator::VisitTypedefDecl(TypedefDecl *D) {
  if (ShouldGenerateLocation(D) && GenLoc(D))
    return;
  DeclContext *DC = D->getDeclContext();
  if (NamedDecl *DCN = dyn_cast<NamedDecl>(DC))
    Visit(DCN);
  Out << "@T@";
  Out << D->getName();
}

void USRGenerator::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  GenLoc(D);
  return;
}

bool USRGenerator::GenLoc(const Decl *D) {
  if (generatedLoc)
    return IgnoreResults;
  generatedLoc = true;
  
  // Guard against null declarations in invalid code.
  if (!D) {
    IgnoreResults = true;
    return true;
  }

  // Use the location of canonical decl.
  D = D->getCanonicalDecl();

  const SourceManager &SM = Context->getSourceManager();
  SourceLocation L = D->getLocStart();
  if (L.isInvalid()) {
    IgnoreResults = true;
    return true;
  }
  L = SM.getExpansionLoc(L);
  const std::pair<FileID, unsigned> &Decomposed = SM.getDecomposedLoc(L);
  const FileEntry *FE = SM.getFileEntryForID(Decomposed.first);
  if (FE) {
    Out << llvm::sys::path::filename(FE->getName());
  }
  else {
    // This case really isn't interesting.
    IgnoreResults = true;
    return true;
  }
  // Use the offest into the FileID to represent the location.  Using
  // a line/column can cause us to look back at the original source file,
  // which is expensive.
  Out << '@' << Decomposed.second;
  return IgnoreResults;
}

void USRGenerator::VisitType(QualType T) {
  // This method mangles in USR information for types.  It can possibly
  // just reuse the naming-mangling logic used by codegen, although the
  // requirements for USRs might not be the same.
  ASTContext &Ctx = *Context;

  do {
    T = Ctx.getCanonicalType(T);
    Qualifiers Q = T.getQualifiers();
    unsigned qVal = 0;
    if (Q.hasConst())
      qVal |= 0x1;
    if (Q.hasVolatile())
      qVal |= 0x2;
    if (Q.hasRestrict())
      qVal |= 0x4;
    if(qVal)
      Out << ((char) ('0' + qVal));

    // Mangle in ObjC GC qualifiers?

    if (const PackExpansionType *Expansion = T->getAs<PackExpansionType>()) {
      Out << 'P';
      T = Expansion->getPattern();
    }
    
    if (const BuiltinType *BT = T->getAs<BuiltinType>()) {
      unsigned char c = '\0';
      switch (BT->getKind()) {
        case BuiltinType::Void:
          c = 'v'; break;
        case BuiltinType::Bool:
          c = 'b'; break;
        case BuiltinType::Char_U:
        case BuiltinType::UChar:
          c = 'c'; break;
        case BuiltinType::Char16:
          c = 'q'; break;
        case BuiltinType::Char32:
          c = 'w'; break;
        case BuiltinType::UShort:
          c = 's'; break;
        case BuiltinType::UInt:
          c = 'i'; break;
        case BuiltinType::ULong:
          c = 'l'; break;
        case BuiltinType::ULongLong:
          c = 'k'; break;
        case BuiltinType::UInt128:
          c = 'j'; break;
        case BuiltinType::Char_S:
        case BuiltinType::SChar:
          c = 'C'; break;
        case BuiltinType::WChar_S:
        case BuiltinType::WChar_U:
          c = 'W'; break;
        case BuiltinType::Short:
          c = 'S'; break;
        case BuiltinType::Int:
          c = 'I'; break;
        case BuiltinType::Long:
          c = 'L'; break;
        case BuiltinType::LongLong:
          c = 'K'; break;
        case BuiltinType::Int128:
          c = 'J'; break;
        case BuiltinType::Half:
          c = 'h'; break;
        case BuiltinType::Float:
          c = 'f'; break;
        case BuiltinType::Double:
          c = 'd'; break;
        case BuiltinType::LongDouble:
          c = 'D'; break;
        case BuiltinType::NullPtr:
          c = 'n'; break;
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
        case BuiltinType::Dependent:
          IgnoreResults = true;
          return;
        case BuiltinType::ObjCId:
          c = 'o'; break;
        case BuiltinType::ObjCClass:
          c = 'O'; break;
        case BuiltinType::ObjCSel:
          c = 'e'; break;
      }
      Out << c;
      return;
    }

    // If we have already seen this (non-built-in) type, use a substitution
    // encoding.
    llvm::DenseMap<const Type *, unsigned>::iterator Substitution
      = TypeSubstitutions.find(T.getTypePtr());
    if (Substitution != TypeSubstitutions.end()) {
      Out << 'S' << Substitution->second << '_';
      return;
    } else {
      // Record this as a substitution.
      unsigned Number = TypeSubstitutions.size();
      TypeSubstitutions[T.getTypePtr()] = Number;
    }
    
    if (const PointerType *PT = T->getAs<PointerType>()) {
      Out << '*';
      T = PT->getPointeeType();
      continue;
    }
    if (const ReferenceType *RT = T->getAs<ReferenceType>()) {
      Out << '&';
      T = RT->getPointeeType();
      continue;
    }
    if (const FunctionProtoType *FT = T->getAs<FunctionProtoType>()) {
      Out << 'F';
      VisitType(FT->getResultType());
      for (FunctionProtoType::arg_type_iterator
            I = FT->arg_type_begin(), E = FT->arg_type_end(); I!=E; ++I) {
        VisitType(*I);
      }
      if (FT->isVariadic())
        Out << '.';
      return;
    }
    if (const BlockPointerType *BT = T->getAs<BlockPointerType>()) {
      Out << 'B';
      T = BT->getPointeeType();
      continue;
    }
    if (const ComplexType *CT = T->getAs<ComplexType>()) {
      Out << '<';
      T = CT->getElementType();
      continue;
    }
    if (const TagType *TT = T->getAs<TagType>()) {
      Out << '$';
      VisitTagDecl(TT->getDecl());
      return;
    }
    if (const TemplateTypeParmType *TTP = T->getAs<TemplateTypeParmType>()) {
      Out << 't' << TTP->getDepth() << '.' << TTP->getIndex();
      return;
    }
    if (const TemplateSpecializationType *Spec
                                    = T->getAs<TemplateSpecializationType>()) {
      Out << '>';
      VisitTemplateName(Spec->getTemplateName());
      Out << Spec->getNumArgs();
      for (unsigned I = 0, N = Spec->getNumArgs(); I != N; ++I)
        VisitTemplateArgument(Spec->getArg(I));
      return;
    }
    
    // Unhandled type.
    Out << ' ';
    break;
  } while (true);
}

void USRGenerator::VisitTemplateParameterList(
                                         const TemplateParameterList *Params) {
  if (!Params)
    return;
  Out << '>' << Params->size();
  for (TemplateParameterList::const_iterator P = Params->begin(),
                                          PEnd = Params->end();
       P != PEnd; ++P) {
    Out << '#';
    if (isa<TemplateTypeParmDecl>(*P)) {
      if (cast<TemplateTypeParmDecl>(*P)->isParameterPack())
        Out<< 'p';
      Out << 'T';
      continue;
    }
    
    if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(*P)) {
      if (NTTP->isParameterPack())
        Out << 'p';
      Out << 'N';
      VisitType(NTTP->getType());
      continue;
    }
    
    TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(*P);
    if (TTP->isParameterPack())
      Out << 'p';
    Out << 't';
    VisitTemplateParameterList(TTP->getTemplateParameters());
  }
}

void USRGenerator::VisitTemplateName(TemplateName Name) {
  if (TemplateDecl *Template = Name.getAsTemplateDecl()) {
    if (TemplateTemplateParmDecl *TTP
                              = dyn_cast<TemplateTemplateParmDecl>(Template)) {
      Out << 't' << TTP->getDepth() << '.' << TTP->getIndex();
      return;
    }
    
    Visit(Template);
    return;
  }
  
  // FIXME: Visit dependent template names.
}

void USRGenerator::VisitTemplateArgument(const TemplateArgument &Arg) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    break;

  case TemplateArgument::Declaration:
    Visit(Arg.getAsDecl());
    break;

  case TemplateArgument::NullPtr:
    break;

  case TemplateArgument::TemplateExpansion:
    Out << 'P'; // pack expansion of...
    // Fall through
  case TemplateArgument::Template:
    VisitTemplateName(Arg.getAsTemplateOrTemplatePattern());
    break;
      
  case TemplateArgument::Expression:
    // FIXME: Visit expressions.
    break;
      
  case TemplateArgument::Pack:
    Out << 'p' << Arg.pack_size();
    for (TemplateArgument::pack_iterator P = Arg.pack_begin(), PEnd = Arg.pack_end();
         P != PEnd; ++P)
      VisitTemplateArgument(*P);
    break;
      
  case TemplateArgument::Type:
    VisitType(Arg.getAsType());
    break;
      
  case TemplateArgument::Integral:
    Out << 'V';
    VisitType(Arg.getIntegralType());
    Out << Arg.getAsIntegral();
    break;
  }
}

//===----------------------------------------------------------------------===//
// General purpose USR generation methods.
//===----------------------------------------------------------------------===//

void USRGenerator::GenObjCClass(StringRef cls) {
  Out << "objc(cs)" << cls;
}

void USRGenerator::GenObjCCategory(StringRef cls, StringRef cat) {
  Out << "objc(cy)" << cls << '@' << cat;
}

void USRGenerator::GenObjCIvar(StringRef ivar) {
  Out << '@' << ivar;
}

void USRGenerator::GenObjCMethod(StringRef meth, bool isInstanceMethod) {
  Out << (isInstanceMethod ? "(im)" : "(cm)") << meth;
}

void USRGenerator::GenObjCProperty(StringRef prop) {
  Out << "(py)" << prop;
}

void USRGenerator::GenObjCProtocol(StringRef prot) {
  Out << "objc(pl)" << prot;
}

//===----------------------------------------------------------------------===//
// API hooks.
//===----------------------------------------------------------------------===//

static inline StringRef extractUSRSuffix(StringRef s) {
  return s.startswith("c:") ? s.substr(2) : "";
}

bool cxcursor::getDeclCursorUSR(const Decl *D, SmallVectorImpl<char> &Buf) {
  // Don't generate USRs for things with invalid locations.
  if (!D || D->getLocStart().isInvalid())
    return true;

  USRGenerator UG(&D->getASTContext(), &Buf);
  UG->Visit(const_cast<Decl*>(D));

  if (UG->ignoreResults())
    return true;

  return false;
}

extern "C" {

CXString clang_getCursorUSR(CXCursor C) {
  const CXCursorKind &K = clang_getCursorKind(C);

  if (clang_isDeclaration(K)) {
    Decl *D = cxcursor::getCursorDecl(C);
    if (!D)
      return createCXString("");

    CXTranslationUnit TU = cxcursor::getCursorTU(C);
    if (!TU)
      return createCXString("");

    CXStringBuf *buf = cxstring::getCXStringBuf(TU);
    if (!buf)
      return createCXString("");

    bool Ignore = cxcursor::getDeclCursorUSR(D, buf->Data);
    if (Ignore) {
      disposeCXStringBuf(buf);
      return createCXString("");
    }

    // Return the C-string, but don't make a copy since it is already in
    // the string buffer.
    buf->Data.push_back('\0');
    return createCXString(buf);
  }

  if (K == CXCursor_MacroDefinition) {
    CXTranslationUnit TU = cxcursor::getCursorTU(C);
    if (!TU)
      return createCXString("");

    CXStringBuf *buf = cxstring::getCXStringBuf(TU);
    if (!buf)
      return createCXString("");

    {
      USRGenerator UG(&cxcursor::getCursorASTUnit(C)->getASTContext(),
                      &buf->Data);
      UG << "macro@"
        << cxcursor::getCursorMacroDefinition(C)->getName()->getNameStart();
    }
    buf->Data.push_back('\0');
    return createCXString(buf);
  }

  return createCXString("");
}

CXString clang_constructUSR_ObjCIvar(const char *name, CXString classUSR) {
  USRGenerator UG;
  UG << extractUSRSuffix(clang_getCString(classUSR));
  UG->GenObjCIvar(name);
  return createCXString(UG.str(), true);
}

CXString clang_constructUSR_ObjCMethod(const char *name,
                                       unsigned isInstanceMethod,
                                       CXString classUSR) {
  USRGenerator UG;
  UG << extractUSRSuffix(clang_getCString(classUSR));
  UG->GenObjCMethod(name, isInstanceMethod);
  return createCXString(UG.str(), true);
}

CXString clang_constructUSR_ObjCClass(const char *name) {
  USRGenerator UG;
  UG->GenObjCClass(name);
  return createCXString(UG.str(), true);
}

CXString clang_constructUSR_ObjCProtocol(const char *name) {
  USRGenerator UG;
  UG->GenObjCProtocol(name);
  return createCXString(UG.str(), true);
}

CXString clang_constructUSR_ObjCCategory(const char *class_name,
                                         const char *category_name) {
  USRGenerator UG;
  UG->GenObjCCategory(class_name, category_name);
  return createCXString(UG.str(), true);
}

CXString clang_constructUSR_ObjCProperty(const char *property,
                                         CXString classUSR) {
  USRGenerator UG;
  UG << extractUSRSuffix(clang_getCString(classUSR));
  UG->GenObjCProperty(property);
  return createCXString(UG.str(), true);
}

} // end extern "C"
