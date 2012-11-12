//===--- DeclPrinter.cpp - Printing implementation for Decl ASTs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl::dump method, which pretty print the
// AST back out to C/Objective-C/C++/Objective-C++ code.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Module.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {
  class DeclPrinter : public DeclVisitor<DeclPrinter> {
    raw_ostream &Out;
    PrintingPolicy Policy;
    unsigned Indentation;
    bool PrintInstantiation;

    raw_ostream& Indent() { return Indent(Indentation); }
    raw_ostream& Indent(unsigned Indentation);
    void ProcessDeclGroup(SmallVectorImpl<Decl*>& Decls);

    void Print(AccessSpecifier AS);

  public:
    DeclPrinter(raw_ostream &Out, const PrintingPolicy &Policy,
                unsigned Indentation = 0, bool PrintInstantiation = false)
      : Out(Out), Policy(Policy), Indentation(Indentation),
        PrintInstantiation(PrintInstantiation) { }

    void VisitDeclContext(DeclContext *DC, bool Indent = true);

    void VisitTranslationUnitDecl(TranslationUnitDecl *D);
    void VisitTypedefDecl(TypedefDecl *D);
    void VisitTypeAliasDecl(TypeAliasDecl *D);
    void VisitEnumDecl(EnumDecl *D);
    void VisitRecordDecl(RecordDecl *D);
    void VisitEnumConstantDecl(EnumConstantDecl *D);
    void VisitFunctionDecl(FunctionDecl *D);
    void VisitFieldDecl(FieldDecl *D);
    void VisitVarDecl(VarDecl *D);
    void VisitLabelDecl(LabelDecl *D);
    void VisitParmVarDecl(ParmVarDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *D);
    void VisitImportDecl(ImportDecl *D);
    void VisitStaticAssertDecl(StaticAssertDecl *D);
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    void VisitCXXRecordDecl(CXXRecordDecl *D);
    void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    void VisitTemplateDecl(const TemplateDecl *D);
    void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    void VisitClassTemplateDecl(ClassTemplateDecl *D);
    void VisitObjCMethodDecl(ObjCMethodDecl *D);
    void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
    void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    void VisitUsingDecl(UsingDecl *D);
    void VisitUsingShadowDecl(UsingShadowDecl *D);

    void PrintTemplateParameters(const TemplateParameterList *Params,
                                 const TemplateArgumentList *Args);
    void prettyPrintAttributes(Decl *D);
  };
}

void Decl::print(raw_ostream &Out, unsigned Indentation,
                 bool PrintInstantiation) const {
  print(Out, getASTContext().getPrintingPolicy(), Indentation, PrintInstantiation);
}

void Decl::print(raw_ostream &Out, const PrintingPolicy &Policy,
                 unsigned Indentation, bool PrintInstantiation) const {
  DeclPrinter Printer(Out, Policy, Indentation, PrintInstantiation);
  Printer.Visit(const_cast<Decl*>(this));
}

static QualType GetBaseType(QualType T) {
  // FIXME: This should be on the Type class!
  QualType BaseType = T;
  while (!BaseType->isSpecifierType()) {
    if (isa<TypedefType>(BaseType))
      break;
    else if (const PointerType* PTy = BaseType->getAs<PointerType>())
      BaseType = PTy->getPointeeType();
    else if (const BlockPointerType *BPy = BaseType->getAs<BlockPointerType>())
      BaseType = BPy->getPointeeType();
    else if (const ArrayType* ATy = dyn_cast<ArrayType>(BaseType))
      BaseType = ATy->getElementType();
    else if (const FunctionType* FTy = BaseType->getAs<FunctionType>())
      BaseType = FTy->getResultType();
    else if (const VectorType *VTy = BaseType->getAs<VectorType>())
      BaseType = VTy->getElementType();
    else if (const ReferenceType *RTy = BaseType->getAs<ReferenceType>())
      BaseType = RTy->getPointeeType();
    else
      llvm_unreachable("Unknown declarator!");
  }
  return BaseType;
}

static QualType getDeclType(Decl* D) {
  if (TypedefNameDecl* TDD = dyn_cast<TypedefNameDecl>(D))
    return TDD->getUnderlyingType();
  if (ValueDecl* VD = dyn_cast<ValueDecl>(D))
    return VD->getType();
  return QualType();
}

void Decl::printGroup(Decl** Begin, unsigned NumDecls,
                      raw_ostream &Out, const PrintingPolicy &Policy,
                      unsigned Indentation) {
  if (NumDecls == 1) {
    (*Begin)->print(Out, Policy, Indentation);
    return;
  }

  Decl** End = Begin + NumDecls;
  TagDecl* TD = dyn_cast<TagDecl>(*Begin);
  if (TD)
    ++Begin;

  PrintingPolicy SubPolicy(Policy);
  if (TD && TD->isCompleteDefinition()) {
    TD->print(Out, Policy, Indentation);
    Out << " ";
    SubPolicy.SuppressTag = true;
  }

  bool isFirst = true;
  for ( ; Begin != End; ++Begin) {
    if (isFirst) {
      SubPolicy.SuppressSpecifiers = false;
      isFirst = false;
    } else {
      if (!isFirst) Out << ", ";
      SubPolicy.SuppressSpecifiers = true;
    }

    (*Begin)->print(Out, SubPolicy, Indentation);
  }
}

void DeclContext::dumpDeclContext() const {
  // Get the translation unit
  const DeclContext *DC = this;
  while (!DC->isTranslationUnit())
    DC = DC->getParent();
  
  ASTContext &Ctx = cast<TranslationUnitDecl>(DC)->getASTContext();
  DeclPrinter Printer(llvm::errs(), Ctx.getPrintingPolicy(), 0);
  Printer.VisitDeclContext(const_cast<DeclContext *>(this), /*Indent=*/false);
}

void Decl::dump() const {
  dump(llvm::errs());
}

void Decl::dump(raw_ostream &Out) const {
  PrintingPolicy Policy = getASTContext().getPrintingPolicy();
  Policy.DumpSourceManager = &getASTContext().getSourceManager();
  print(Out, Policy, /*Indentation*/ 0, /*PrintInstantiation*/ true);
}

raw_ostream& DeclPrinter::Indent(unsigned Indentation) {
  for (unsigned i = 0; i != Indentation; ++i)
    Out << "  ";
  return Out;
}

void DeclPrinter::prettyPrintAttributes(Decl *D) {
  if (Policy.SuppressAttributes)
    return;
  
  if (D->hasAttrs()) {
    AttrVec &Attrs = D->getAttrs();
    for (AttrVec::const_iterator i=Attrs.begin(), e=Attrs.end(); i!=e; ++i) {
      Attr *A = *i;
      A->printPretty(Out, Policy);
    }
  }
}

void DeclPrinter::ProcessDeclGroup(SmallVectorImpl<Decl*>& Decls) {
  this->Indent();
  Decl::printGroup(Decls.data(), Decls.size(), Out, Policy, Indentation);
  Out << ";\n";
  Decls.clear();

}

void DeclPrinter::Print(AccessSpecifier AS) {
  switch(AS) {
  case AS_none:      llvm_unreachable("No access specifier!");
  case AS_public:    Out << "public"; break;
  case AS_protected: Out << "protected"; break;
  case AS_private:   Out << "private"; break;
  }
}

//----------------------------------------------------------------------------
// Common C declarations
//----------------------------------------------------------------------------

void DeclPrinter::VisitDeclContext(DeclContext *DC, bool Indent) {
  if (Policy.TerseOutput)
    return;

  if (Indent)
    Indentation += Policy.Indentation;

  SmallVector<Decl*, 2> Decls;
  for (DeclContext::decl_iterator D = DC->decls_begin(), DEnd = DC->decls_end();
       D != DEnd; ++D) {

    // Don't print ObjCIvarDecls, as they are printed when visiting the
    // containing ObjCInterfaceDecl.
    if (isa<ObjCIvarDecl>(*D))
      continue;

    if (!Policy.DumpSourceManager) {
      // Skip over implicit declarations in pretty-printing mode.
      if (D->isImplicit()) continue;
      // FIXME: Ugly hack so we don't pretty-print the builtin declaration
      // of __builtin_va_list or __[u]int128_t.  There should be some other way
      // to check that.
      if (NamedDecl *ND = dyn_cast<NamedDecl>(*D)) {
        if (IdentifierInfo *II = ND->getIdentifier()) {
          if (II->isStr("__builtin_va_list") ||
              II->isStr("__int128_t") || II->isStr("__uint128_t"))
            continue;
        }
      }
    }

    // The next bits of code handles stuff like "struct {int x;} a,b"; we're
    // forced to merge the declarations because there's no other way to
    // refer to the struct in question.  This limited merging is safe without
    // a bunch of other checks because it only merges declarations directly
    // referring to the tag, not typedefs.
    //
    // Check whether the current declaration should be grouped with a previous
    // unnamed struct.
    QualType CurDeclType = getDeclType(*D);
    if (!Decls.empty() && !CurDeclType.isNull()) {
      QualType BaseType = GetBaseType(CurDeclType);
      if (!BaseType.isNull() && isa<TagType>(BaseType) &&
          cast<TagType>(BaseType)->getDecl() == Decls[0]) {
        Decls.push_back(*D);
        continue;
      }
    }

    // If we have a merged group waiting to be handled, handle it now.
    if (!Decls.empty())
      ProcessDeclGroup(Decls);

    // If the current declaration is an unnamed tag type, save it
    // so we can merge it with the subsequent declaration(s) using it.
    if (isa<TagDecl>(*D) && !cast<TagDecl>(*D)->getIdentifier()) {
      Decls.push_back(*D);
      continue;
    }

    if (isa<AccessSpecDecl>(*D)) {
      Indentation -= Policy.Indentation;
      this->Indent();
      Print(D->getAccess());
      Out << ":\n";
      Indentation += Policy.Indentation;
      continue;
    }

    this->Indent();
    Visit(*D);

    // FIXME: Need to be able to tell the DeclPrinter when
    const char *Terminator = 0;
    if (isa<FunctionDecl>(*D) &&
        cast<FunctionDecl>(*D)->isThisDeclarationADefinition())
      Terminator = 0;
    else if (isa<ObjCMethodDecl>(*D) && cast<ObjCMethodDecl>(*D)->getBody())
      Terminator = 0;
    else if (isa<NamespaceDecl>(*D) || isa<LinkageSpecDecl>(*D) ||
             isa<ObjCImplementationDecl>(*D) ||
             isa<ObjCInterfaceDecl>(*D) ||
             isa<ObjCProtocolDecl>(*D) ||
             isa<ObjCCategoryImplDecl>(*D) ||
             isa<ObjCCategoryDecl>(*D))
      Terminator = 0;
    else if (isa<EnumConstantDecl>(*D)) {
      DeclContext::decl_iterator Next = D;
      ++Next;
      if (Next != DEnd)
        Terminator = ",";
    } else
      Terminator = ";";

    if (Terminator)
      Out << Terminator;
    Out << "\n";
  }

  if (!Decls.empty())
    ProcessDeclGroup(Decls);

  if (Indent)
    Indentation -= Policy.Indentation;
}

void DeclPrinter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  VisitDeclContext(D, false);
}

void DeclPrinter::VisitTypedefDecl(TypedefDecl *D) {
  if (!Policy.SuppressSpecifiers) {
    Out << "typedef ";
    
    if (D->isModulePrivate())
      Out << "__module_private__ ";
  }
  D->getUnderlyingType().print(Out, Policy, D->getName());
  prettyPrintAttributes(D);
}

void DeclPrinter::VisitTypeAliasDecl(TypeAliasDecl *D) {
  Out << "using " << *D << " = " << D->getUnderlyingType().getAsString(Policy);
}

void DeclPrinter::VisitEnumDecl(EnumDecl *D) {
  if (!Policy.SuppressSpecifiers && D->isModulePrivate())
    Out << "__module_private__ ";
  Out << "enum ";
  if (D->isScoped()) {
    if (D->isScopedUsingClassTag())
      Out << "class ";
    else
      Out << "struct ";
  }
  Out << *D;

  if (D->isFixed())
    Out << " : " << D->getIntegerType().stream(Policy);

  if (D->isCompleteDefinition()) {
    Out << " {\n";
    VisitDeclContext(D);
    Indent() << "}";
  }
  prettyPrintAttributes(D);
}

void DeclPrinter::VisitRecordDecl(RecordDecl *D) {
  if (!Policy.SuppressSpecifiers && D->isModulePrivate())
    Out << "__module_private__ ";
  Out << D->getKindName();
  if (D->getIdentifier())
    Out << ' ' << *D;

  if (D->isCompleteDefinition()) {
    Out << " {\n";
    VisitDeclContext(D);
    Indent() << "}";
  }
}

void DeclPrinter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  Out << *D;
  if (Expr *Init = D->getInitExpr()) {
    Out << " = ";
    Init->printPretty(Out, 0, Policy, Indentation);
  }
}

void DeclPrinter::VisitFunctionDecl(FunctionDecl *D) {
  if (!Policy.SuppressSpecifiers) {
    switch (D->getStorageClassAsWritten()) {
    case SC_None: break;
    case SC_Extern: Out << "extern "; break;
    case SC_Static: Out << "static "; break;
    case SC_PrivateExtern: Out << "__private_extern__ "; break;
    case SC_Auto: case SC_Register: case SC_OpenCLWorkGroupLocal:
      llvm_unreachable("invalid for functions");
    }

    if (D->isInlineSpecified())  Out << "inline ";
    if (D->isVirtualAsWritten()) Out << "virtual ";
    if (D->isModulePrivate())    Out << "__module_private__ ";
  }

  PrintingPolicy SubPolicy(Policy);
  SubPolicy.SuppressSpecifiers = false;
  std::string Proto = D->getNameInfo().getAsString();

  QualType Ty = D->getType();
  while (const ParenType *PT = dyn_cast<ParenType>(Ty)) {
    Proto = '(' + Proto + ')';
    Ty = PT->getInnerType();
  }

  if (isa<FunctionType>(Ty)) {
    const FunctionType *AFT = Ty->getAs<FunctionType>();
    const FunctionProtoType *FT = 0;
    if (D->hasWrittenPrototype())
      FT = dyn_cast<FunctionProtoType>(AFT);

    Proto += "(";
    if (FT) {
      llvm::raw_string_ostream POut(Proto);
      DeclPrinter ParamPrinter(POut, SubPolicy, Indentation);
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        if (i) POut << ", ";
        ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
      }

      if (FT->isVariadic()) {
        if (D->getNumParams()) POut << ", ";
        POut << "...";
      }
    } else if (D->doesThisDeclarationHaveABody() && !D->hasPrototype()) {
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        if (i)
          Proto += ", ";
        Proto += D->getParamDecl(i)->getNameAsString();
      }
    }

    Proto += ")";
    
    if (FT) {
      if (FT->isConst())
        Proto += " const";
      if (FT->isVolatile())
        Proto += " volatile";
      if (FT->isRestrict())
        Proto += " restrict";
    }

    if (FT && FT->hasDynamicExceptionSpec()) {
      Proto += " throw(";
      if (FT->getExceptionSpecType() == EST_MSAny)
        Proto += "...";
      else 
        for (unsigned I = 0, N = FT->getNumExceptions(); I != N; ++I) {
          if (I)
            Proto += ", ";

          Proto += FT->getExceptionType(I).getAsString(SubPolicy);
        }
      Proto += ")";
    } else if (FT && isNoexceptExceptionSpec(FT->getExceptionSpecType())) {
      Proto += " noexcept";
      if (FT->getExceptionSpecType() == EST_ComputedNoexcept) {
        Proto += "(";
        llvm::raw_string_ostream EOut(Proto);
        FT->getNoexceptExpr()->printPretty(EOut, 0, SubPolicy,
                                           Indentation);
        EOut.flush();
        Proto += EOut.str();
        Proto += ")";
      }
    }

    if (CXXConstructorDecl *CDecl = dyn_cast<CXXConstructorDecl>(D)) {
      bool HasInitializerList = false;
      for (CXXConstructorDecl::init_const_iterator B = CDecl->init_begin(),
           E = CDecl->init_end();
           B != E; ++B) {
        CXXCtorInitializer * BMInitializer = (*B);
        if (BMInitializer->isInClassMemberInitializer())
          continue;

        if (!HasInitializerList) {
          Proto += " : ";
          Out << Proto;
          Proto.clear();
          HasInitializerList = true;
        } else
          Out << ", ";

        if (BMInitializer->isAnyMemberInitializer()) {
          FieldDecl *FD = BMInitializer->getAnyMember();
          Out << *FD;
        } else {
          Out << QualType(BMInitializer->getBaseClass(), 0).getAsString(Policy);
        }
        
        Out << "(";
        if (!BMInitializer->getInit()) {
          // Nothing to print
        } else {
          Expr *Init = BMInitializer->getInit();
          if (ExprWithCleanups *Tmp = dyn_cast<ExprWithCleanups>(Init))
            Init = Tmp->getSubExpr();
          
          Init = Init->IgnoreParens();
          
          Expr *SimpleInit = 0;
          Expr **Args = 0;
          unsigned NumArgs = 0;
          if (ParenListExpr *ParenList = dyn_cast<ParenListExpr>(Init)) {
            Args = ParenList->getExprs();
            NumArgs = ParenList->getNumExprs();
          } else if (CXXConstructExpr *Construct
                                        = dyn_cast<CXXConstructExpr>(Init)) {
            Args = Construct->getArgs();
            NumArgs = Construct->getNumArgs();
          } else
            SimpleInit = Init;
          
          if (SimpleInit)
            SimpleInit->printPretty(Out, 0, Policy, Indentation);
          else {
            for (unsigned I = 0; I != NumArgs; ++I) {
              if (isa<CXXDefaultArgExpr>(Args[I]))
                break;
              
              if (I)
                Out << ", ";
              Args[I]->printPretty(Out, 0, Policy, Indentation);
            }
          }
        }
        Out << ")";
      }
    }
    else
      AFT->getResultType().print(Out, Policy, Proto);
  } else {
    Ty.print(Out, Policy, Proto);
  }

  prettyPrintAttributes(D);

  if (D->isPure())
    Out << " = 0";
  else if (D->isDeletedAsWritten())
    Out << " = delete";
  else if (D->doesThisDeclarationHaveABody() && !Policy.TerseOutput) {
    if (!D->hasPrototype() && D->getNumParams()) {
      // This is a K&R function definition, so we need to print the
      // parameters.
      Out << '\n';
      DeclPrinter ParamPrinter(Out, SubPolicy, Indentation);
      Indentation += Policy.Indentation;
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        Indent();
        ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
        Out << ";\n";
      }
      Indentation -= Policy.Indentation;
    } else
      Out << ' ';

    D->getBody()->printPretty(Out, 0, SubPolicy, Indentation);
    Out << '\n';
  }
}

void DeclPrinter::VisitFieldDecl(FieldDecl *D) {
  if (!Policy.SuppressSpecifiers && D->isMutable())
    Out << "mutable ";
  if (!Policy.SuppressSpecifiers && D->isModulePrivate())
    Out << "__module_private__ ";

  Out << D->getType().stream(Policy, D->getName());

  if (D->isBitField()) {
    Out << " : ";
    D->getBitWidth()->printPretty(Out, 0, Policy, Indentation);
  }

  Expr *Init = D->getInClassInitializer();
  if (!Policy.SuppressInitializers && Init) {
    if (D->getInClassInitStyle() == ICIS_ListInit)
      Out << " ";
    else
      Out << " = ";
    Init->printPretty(Out, 0, Policy, Indentation);
  }
  prettyPrintAttributes(D);
}

void DeclPrinter::VisitLabelDecl(LabelDecl *D) {
  Out << *D << ":";
}


void DeclPrinter::VisitVarDecl(VarDecl *D) {
  StorageClass SCAsWritten = D->getStorageClassAsWritten();
  if (!Policy.SuppressSpecifiers && SCAsWritten != SC_None)
    Out << VarDecl::getStorageClassSpecifierString(SCAsWritten) << " ";

  if (!Policy.SuppressSpecifiers && D->isThreadSpecified())
    Out << "__thread ";
  if (!Policy.SuppressSpecifiers && D->isModulePrivate())
    Out << "__module_private__ ";

  QualType T = D->getType();
  if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D))
    T = Parm->getOriginalType();
  T.print(Out, Policy, D->getName());
  Expr *Init = D->getInit();
  if (!Policy.SuppressInitializers && Init) {
    bool ImplicitInit = false;
    if (CXXConstructExpr *Construct = dyn_cast<CXXConstructExpr>(Init))
      ImplicitInit = D->getInitStyle() == VarDecl::CallInit &&
          Construct->getNumArgs() == 0 && !Construct->isListInitialization();
    if (!ImplicitInit) {
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        Out << "(";
      else if (D->getInitStyle() == VarDecl::CInit) {
        Out << " = ";
      }
      Init->printPretty(Out, 0, Policy, Indentation);
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        Out << ")";
    }
  }
  prettyPrintAttributes(D);
}

void DeclPrinter::VisitParmVarDecl(ParmVarDecl *D) {
  VisitVarDecl(D);
}

void DeclPrinter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
  Out << "__asm (";
  D->getAsmString()->printPretty(Out, 0, Policy, Indentation);
  Out << ")";
}

void DeclPrinter::VisitImportDecl(ImportDecl *D) {
  Out << "@__experimental_modules_import " << D->getImportedModule()->getFullModuleName()
      << ";\n";
}

void DeclPrinter::VisitStaticAssertDecl(StaticAssertDecl *D) {
  Out << "static_assert(";
  D->getAssertExpr()->printPretty(Out, 0, Policy, Indentation);
  Out << ", ";
  D->getMessage()->printPretty(Out, 0, Policy, Indentation);
  Out << ")";
}

//----------------------------------------------------------------------------
// C++ declarations
//----------------------------------------------------------------------------
void DeclPrinter::VisitNamespaceDecl(NamespaceDecl *D) {
  if (D->isInline())
    Out << "inline ";
  Out << "namespace " << *D << " {\n";
  VisitDeclContext(D);
  Indent() << "}";
}

void DeclPrinter::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  Out << "using namespace ";
  if (D->getQualifier())
    D->getQualifier()->print(Out, Policy);
  Out << *D->getNominatedNamespaceAsWritten();
}

void DeclPrinter::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  Out << "namespace " << *D << " = ";
  if (D->getQualifier())
    D->getQualifier()->print(Out, Policy);
  Out << *D->getAliasedNamespace();
}

void DeclPrinter::VisitCXXRecordDecl(CXXRecordDecl *D) {
  if (!Policy.SuppressSpecifiers && D->isModulePrivate())
    Out << "__module_private__ ";
  Out << D->getKindName();
  if (D->getIdentifier())
    Out << ' ' << *D;

  if (D->isCompleteDefinition()) {
    // Print the base classes
    if (D->getNumBases()) {
      Out << " : ";
      for (CXXRecordDecl::base_class_iterator Base = D->bases_begin(),
             BaseEnd = D->bases_end(); Base != BaseEnd; ++Base) {
        if (Base != D->bases_begin())
          Out << ", ";

        if (Base->isVirtual())
          Out << "virtual ";

        AccessSpecifier AS = Base->getAccessSpecifierAsWritten();
        if (AS != AS_none)
          Print(AS);
        Out << " " << Base->getType().getAsString(Policy);

        if (Base->isPackExpansion())
          Out << "...";
      }
    }

    // Print the class definition
    // FIXME: Doesn't print access specifiers, e.g., "public:"
    Out << " {\n";
    VisitDeclContext(D);
    Indent() << "}";
  }
}

void DeclPrinter::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  const char *l;
  if (D->getLanguage() == LinkageSpecDecl::lang_c)
    l = "C";
  else {
    assert(D->getLanguage() == LinkageSpecDecl::lang_cxx &&
           "unknown language in linkage specification");
    l = "C++";
  }

  Out << "extern \"" << l << "\" ";
  if (D->hasBraces()) {
    Out << "{\n";
    VisitDeclContext(D);
    Indent() << "}";
  } else
    Visit(*D->decls_begin());
}

void DeclPrinter::PrintTemplateParameters(
    const TemplateParameterList *Params, const TemplateArgumentList *Args = 0) {
  assert(Params);
  assert(!Args || Params->size() == Args->size());

  Out << "template <";

  for (unsigned i = 0, e = Params->size(); i != e; ++i) {
    if (i != 0)
      Out << ", ";

    const Decl *Param = Params->getParam(i);
    if (const TemplateTypeParmDecl *TTP =
          dyn_cast<TemplateTypeParmDecl>(Param)) {

      if (TTP->wasDeclaredWithTypename())
        Out << "typename ";
      else
        Out << "class ";

      if (TTP->isParameterPack())
        Out << "... ";

      Out << *TTP;

      if (Args) {
        Out << " = ";
        Args->get(i).print(Policy, Out);
      } else if (TTP->hasDefaultArgument()) {
        Out << " = ";
        Out << TTP->getDefaultArgument().getAsString(Policy);
      };
    } else if (const NonTypeTemplateParmDecl *NTTP =
                 dyn_cast<NonTypeTemplateParmDecl>(Param)) {
      Out << NTTP->getType().getAsString(Policy);

      if (NTTP->isParameterPack() && !isa<PackExpansionType>(NTTP->getType()))
        Out << "...";
        
      if (IdentifierInfo *Name = NTTP->getIdentifier()) {
        Out << ' ';
        Out << Name->getName();
      }

      if (Args) {
        Out << " = ";
        Args->get(i).print(Policy, Out);
      } else if (NTTP->hasDefaultArgument()) {
        Out << " = ";
        NTTP->getDefaultArgument()->printPretty(Out, 0, Policy, Indentation);
      }
    } else if (const TemplateTemplateParmDecl *TTPD =
                 dyn_cast<TemplateTemplateParmDecl>(Param)) {
      VisitTemplateDecl(TTPD);
      // FIXME: print the default argument, if present.
    }
  }

  Out << "> ";
}

void DeclPrinter::VisitTemplateDecl(const TemplateDecl *D) {
  PrintTemplateParameters(D->getTemplateParameters());

  if (const TemplateTemplateParmDecl *TTP =
        dyn_cast<TemplateTemplateParmDecl>(D)) {
    Out << "class ";
    if (TTP->isParameterPack())
      Out << "...";
    Out << D->getName();
  } else {
    Visit(D->getTemplatedDecl());
  }
}

void DeclPrinter::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  if (PrintInstantiation) {
    TemplateParameterList *Params = D->getTemplateParameters();
    for (FunctionTemplateDecl::spec_iterator I = D->spec_begin(), E = D->spec_end();
         I != E; ++I) {
      PrintTemplateParameters(Params, (*I)->getTemplateSpecializationArgs());
      Visit(*I);
    }
  }

  return VisitRedeclarableTemplateDecl(D);
}

void DeclPrinter::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  if (PrintInstantiation) {
    TemplateParameterList *Params = D->getTemplateParameters();
    for (ClassTemplateDecl::spec_iterator I = D->spec_begin(), E = D->spec_end();
         I != E; ++I) {
      PrintTemplateParameters(Params, &(*I)->getTemplateArgs());
      Visit(*I);
      Out << '\n';
    }
  }

  return VisitRedeclarableTemplateDecl(D);
}

//----------------------------------------------------------------------------
// Objective-C declarations
//----------------------------------------------------------------------------

void DeclPrinter::VisitObjCMethodDecl(ObjCMethodDecl *OMD) {
  if (OMD->isInstanceMethod())
    Out << "- ";
  else
    Out << "+ ";
  if (!OMD->getResultType().isNull())
    Out << '(' << OMD->getResultType().getAsString(Policy) << ")";

  std::string name = OMD->getSelector().getAsString();
  std::string::size_type pos, lastPos = 0;
  for (ObjCMethodDecl::param_iterator PI = OMD->param_begin(),
       E = OMD->param_end(); PI != E; ++PI) {
    // FIXME: selector is missing here!
    pos = name.find_first_of(':', lastPos);
    Out << " " << name.substr(lastPos, pos - lastPos);
    Out << ":(" << (*PI)->getType().getAsString(Policy) << ')' << **PI;
    lastPos = pos + 1;
  }

  if (OMD->param_begin() == OMD->param_end())
    Out << " " << name;

  if (OMD->isVariadic())
      Out << ", ...";

  if (OMD->getBody() && !Policy.TerseOutput) {
    Out << ' ';
    OMD->getBody()->printPretty(Out, 0, Policy);
    Out << '\n';
  }
}

void DeclPrinter::VisitObjCImplementationDecl(ObjCImplementationDecl *OID) {
  std::string I = OID->getNameAsString();
  ObjCInterfaceDecl *SID = OID->getSuperClass();

  if (SID)
    Out << "@implementation " << I << " : " << *SID;
  else
    Out << "@implementation " << I;
  Out << "\n";
  VisitDeclContext(OID, false);
  Out << "@end";
}

void DeclPrinter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *OID) {
  std::string I = OID->getNameAsString();
  ObjCInterfaceDecl *SID = OID->getSuperClass();

  if (!OID->isThisDeclarationADefinition()) {
    Out << "@class " << I << ";";
    return;
  }
  
  if (SID)
    Out << "@interface " << I << " : " << *SID;
  else
    Out << "@interface " << I;

  // Protocols?
  const ObjCList<ObjCProtocolDecl> &Protocols = OID->getReferencedProtocols();
  if (!Protocols.empty()) {
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
         E = Protocols.end(); I != E; ++I)
      Out << (I == Protocols.begin() ? '<' : ',') << **I;
  }

  if (!Protocols.empty())
    Out << "> ";

  if (OID->ivar_size() > 0) {
    Out << "{\n";
    Indentation += Policy.Indentation;
    for (ObjCInterfaceDecl::ivar_iterator I = OID->ivar_begin(),
         E = OID->ivar_end(); I != E; ++I) {
      Indent() << I->getType().getAsString(Policy) << ' ' << **I << ";\n";
    }
    Indentation -= Policy.Indentation;
    Out << "}\n";
  }

  VisitDeclContext(OID, false);
  Out << "@end";
  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCProtocolDecl(ObjCProtocolDecl *PID) {
  if (!PID->isThisDeclarationADefinition()) {
    Out << "@protocol " << PID->getIdentifier() << ";\n";
    return;
  }
  
  Out << "@protocol " << *PID << '\n';
  VisitDeclContext(PID, false);
  Out << "@end";
}

void DeclPrinter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *PID) {
  Out << "@implementation " << *PID->getClassInterface() << '(' << *PID <<")\n";

  VisitDeclContext(PID, false);
  Out << "@end";
  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCCategoryDecl(ObjCCategoryDecl *PID) {
  Out << "@interface " << *PID->getClassInterface() << '(' << *PID << ")\n";
  VisitDeclContext(PID, false);
  Out << "@end";

  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *AID) {
  Out << "@compatibility_alias " << *AID
      << ' ' << *AID->getClassInterface() << ";\n";
}

/// PrintObjCPropertyDecl - print a property declaration.
///
void DeclPrinter::VisitObjCPropertyDecl(ObjCPropertyDecl *PDecl) {
  if (PDecl->getPropertyImplementation() == ObjCPropertyDecl::Required)
    Out << "@required\n";
  else if (PDecl->getPropertyImplementation() == ObjCPropertyDecl::Optional)
    Out << "@optional\n";

  Out << "@property";
  if (PDecl->getPropertyAttributes() != ObjCPropertyDecl::OBJC_PR_noattr) {
    bool first = true;
    Out << " (";
    if (PDecl->getPropertyAttributes() &
        ObjCPropertyDecl::OBJC_PR_readonly) {
      Out << (first ? ' ' : ',') << "readonly";
      first = false;
    }

    if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_getter) {
      Out << (first ? ' ' : ',') << "getter = "
          << PDecl->getGetterName().getAsString();
      first = false;
    }
    if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_setter) {
      Out << (first ? ' ' : ',') << "setter = "
          << PDecl->getSetterName().getAsString();
      first = false;
    }

    if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_assign) {
      Out << (first ? ' ' : ',') << "assign";
      first = false;
    }

    if (PDecl->getPropertyAttributes() &
        ObjCPropertyDecl::OBJC_PR_readwrite) {
      Out << (first ? ' ' : ',') << "readwrite";
      first = false;
    }

    if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_retain) {
      Out << (first ? ' ' : ',') << "retain";
      first = false;
    }

    if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_strong) {
      Out << (first ? ' ' : ',') << "strong";
      first = false;
    }

    if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_copy) {
      Out << (first ? ' ' : ',') << "copy";
      first = false;
    }

    if (PDecl->getPropertyAttributes() &
        ObjCPropertyDecl::OBJC_PR_nonatomic) {
      Out << (first ? ' ' : ',') << "nonatomic";
      first = false;
    }
    if (PDecl->getPropertyAttributes() &
        ObjCPropertyDecl::OBJC_PR_atomic) {
      Out << (first ? ' ' : ',') << "atomic";
      first = false;
    }
    
    (void) first; // Silence dead store warning due to idiomatic code.
    Out << " )";
  }
  Out << ' ' << PDecl->getType().getAsString(Policy) << ' ' << *PDecl;
}

void DeclPrinter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *PID) {
  if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize)
    Out << "@synthesize ";
  else
    Out << "@dynamic ";
  Out << *PID->getPropertyDecl();
  if (PID->getPropertyIvarDecl())
    Out << '=' << *PID->getPropertyIvarDecl();
}

void DeclPrinter::VisitUsingDecl(UsingDecl *D) {
  Out << "using ";
  D->getQualifier()->print(Out, Policy);
  Out << *D;
}

void
DeclPrinter::VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D) {
  Out << "using typename ";
  D->getQualifier()->print(Out, Policy);
  Out << D->getDeclName();
}

void DeclPrinter::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  Out << "using ";
  D->getQualifier()->print(Out, Policy);
  Out << D->getDeclName();
}

void DeclPrinter::VisitUsingShadowDecl(UsingShadowDecl *D) {
  // ignore
}
