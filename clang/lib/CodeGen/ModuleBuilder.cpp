//===--- ModuleBuilder.cpp - Emit LLVM Code from ASTs ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This builds an AST and converts it to LLVM Code.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/ModuleBuilder.h"
#include "CodeGenModule.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/DataLayout.h"
#include "llvm/ADT/OwningPtr.h"
using namespace clang;

namespace {
  class CodeGeneratorImpl : public CodeGenerator {
    DiagnosticsEngine &Diags;
    OwningPtr<const llvm::DataLayout> TD;
    ASTContext *Ctx;
    const CodeGenOptions CodeGenOpts;  // Intentionally copied in.
  protected:
    OwningPtr<llvm::Module> M;
    OwningPtr<CodeGen::CodeGenModule> Builder;
  public:
    CodeGeneratorImpl(DiagnosticsEngine &diags, const std::string& ModuleName,
                      const CodeGenOptions &CGO, llvm::LLVMContext& C)
      : Diags(diags), CodeGenOpts(CGO), M(new llvm::Module(ModuleName, C)) {}

    virtual ~CodeGeneratorImpl() {}

    virtual llvm::Module* GetModule() {
      return M.get();
    }

    virtual llvm::Module* ReleaseModule() {
      return M.take();
    }

    virtual void Initialize(ASTContext &Context) {
      Ctx = &Context;

      M->setTargetTriple(Ctx->getTargetInfo().getTriple().getTriple());
      M->setDataLayout(Ctx->getTargetInfo().getTargetDescription());
      TD.reset(new llvm::DataLayout(Ctx->getTargetInfo().getTargetDescription()));
      Builder.reset(new CodeGen::CodeGenModule(Context, CodeGenOpts,
                                               *M, *TD, Diags));
    }

    virtual void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
      Builder->HandleCXXStaticMemberVarInstantiation(VD);
    }

    virtual bool HandleTopLevelDecl(DeclGroupRef DG) {
      // Make sure to emit all elements of a Decl.
      for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I)
        Builder->EmitTopLevelDecl(*I);
      return true;
    }

    /// HandleTagDeclDefinition - This callback is invoked each time a TagDecl
    /// to (e.g. struct, union, enum, class) is completed. This allows the
    /// client hack on the type, which can occur at any point in the file
    /// (because these can be defined in declspecs).
    virtual void HandleTagDeclDefinition(TagDecl *D) {
      Builder->UpdateCompletedType(D);
      
      // In C++, we may have member functions that need to be emitted at this 
      // point.
      if (Ctx->getLangOpts().CPlusPlus && !D->isDependentContext()) {
        for (DeclContext::decl_iterator M = D->decls_begin(), 
                                     MEnd = D->decls_end();
             M != MEnd; ++M)
          if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(*M))
            if (Method->doesThisDeclarationHaveABody() &&
                (Method->hasAttr<UsedAttr>() || 
                 Method->hasAttr<ConstructorAttr>()))
              Builder->EmitTopLevelDecl(Method);
      }
    }

    virtual void HandleTranslationUnit(ASTContext &Ctx) {
      if (Diags.hasErrorOccurred()) {
        M.reset();
        return;
      }

      if (Builder)
        Builder->Release();
    }

    virtual void CompleteTentativeDefinition(VarDecl *D) {
      if (Diags.hasErrorOccurred())
        return;

      Builder->EmitTentativeDefinition(D);
    }

    virtual void HandleVTable(CXXRecordDecl *RD, bool DefinitionRequired) {
      if (Diags.hasErrorOccurred())
        return;

      Builder->EmitVTable(RD, DefinitionRequired);
    }
  };
}

void CodeGenerator::anchor() { }

CodeGenerator *clang::CreateLLVMCodeGen(DiagnosticsEngine &Diags,
                                        const std::string& ModuleName,
                                        const CodeGenOptions &CGO,
                                        llvm::LLVMContext& C) {
  return new CodeGeneratorImpl(Diags, ModuleName, CGO, C);
}
