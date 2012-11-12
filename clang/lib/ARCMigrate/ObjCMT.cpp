//===--- ObjCMT.cpp - ObjC Migrate Tool -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/ARCMigrate/ARCMTActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/NSAPI.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Edit/Rewriters.h"
#include "clang/Edit/EditedSource.h"
#include "clang/Edit/Commit.h"
#include "clang/Edit/EditsReceiver.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/FileManager.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;
using namespace arcmt;

namespace {

class ObjCMigrateASTConsumer : public ASTConsumer {
  void migrateDecl(Decl *D);

public:
  std::string MigrateDir;
  bool MigrateLiterals;
  bool MigrateSubscripting;
  llvm::OwningPtr<NSAPI> NSAPIObj;
  llvm::OwningPtr<edit::EditedSource> Editor;
  FileRemapper &Remapper;
  FileManager &FileMgr;
  const PreprocessingRecord *PPRec;
  bool IsOutputFile;

  ObjCMigrateASTConsumer(StringRef migrateDir,
                         bool migrateLiterals,
                         bool migrateSubscripting,
                         FileRemapper &remapper,
                         FileManager &fileMgr,
                         const PreprocessingRecord *PPRec,
                         bool isOutputFile = false)
  : MigrateDir(migrateDir),
    MigrateLiterals(migrateLiterals),
    MigrateSubscripting(migrateSubscripting),
    Remapper(remapper), FileMgr(fileMgr), PPRec(PPRec),
    IsOutputFile(isOutputFile) { }

protected:
  virtual void Initialize(ASTContext &Context) {
    NSAPIObj.reset(new NSAPI(Context));
    Editor.reset(new edit::EditedSource(Context.getSourceManager(),
                                        Context.getLangOpts(),
                                        PPRec));
  }

  virtual bool HandleTopLevelDecl(DeclGroupRef DG) {
    for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I)
      migrateDecl(*I);
    return true;
  }
  virtual void HandleInterestingDecl(DeclGroupRef DG) {
    // Ignore decls from the PCH.
  }
  virtual void HandleTopLevelDeclInObjCContainer(DeclGroupRef DG) {
    ObjCMigrateASTConsumer::HandleTopLevelDecl(DG);
  }

  virtual void HandleTranslationUnit(ASTContext &Ctx);
};

}

ObjCMigrateAction::ObjCMigrateAction(FrontendAction *WrappedAction,
                             StringRef migrateDir,
                             bool migrateLiterals,
                             bool migrateSubscripting)
  : WrapperFrontendAction(WrappedAction), MigrateDir(migrateDir),
    MigrateLiterals(migrateLiterals), MigrateSubscripting(migrateSubscripting),
    CompInst(0) {
  if (MigrateDir.empty())
    MigrateDir = "."; // user current directory if none is given.
}

ASTConsumer *ObjCMigrateAction::CreateASTConsumer(CompilerInstance &CI,
                                                  StringRef InFile) {
  ASTConsumer *
    WrappedConsumer = WrapperFrontendAction::CreateASTConsumer(CI, InFile);
  ASTConsumer *MTConsumer = new ObjCMigrateASTConsumer(MigrateDir,
                                                       MigrateLiterals,
                                                       MigrateSubscripting,
                                                       Remapper,
                                                    CompInst->getFileManager(),
                          CompInst->getPreprocessor().getPreprocessingRecord()); 
  ASTConsumer *Consumers[] = { MTConsumer, WrappedConsumer };
  return new MultiplexConsumer(Consumers);
}

bool ObjCMigrateAction::BeginInvocation(CompilerInstance &CI) {
  Remapper.initFromDisk(MigrateDir, CI.getDiagnostics(),
                        /*ignoreIfFilesChanges=*/true);
  CompInst = &CI;
  CI.getDiagnostics().setIgnoreAllWarnings(true);
  CI.getPreprocessorOpts().DetailedRecord = true;
  CI.getPreprocessorOpts().DetailedRecordConditionalDirectives = true;
  return true;
}

namespace {
class ObjCMigrator : public RecursiveASTVisitor<ObjCMigrator> {
  ObjCMigrateASTConsumer &Consumer;

public:
  ObjCMigrator(ObjCMigrateASTConsumer &consumer) : Consumer(consumer) { }

  bool shouldVisitTemplateInstantiations() const { return false; }
  bool shouldWalkTypesOfTypeLocs() const { return false; }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    if (Consumer.MigrateLiterals) {
      edit::Commit commit(*Consumer.Editor);
      edit::rewriteToObjCLiteralSyntax(E, *Consumer.NSAPIObj, commit);
      Consumer.Editor->commit(commit);
    }

    if (Consumer.MigrateSubscripting) {
      edit::Commit commit(*Consumer.Editor);
      edit::rewriteToObjCSubscriptSyntax(E, *Consumer.NSAPIObj, commit);
      Consumer.Editor->commit(commit);
    }

    return true;
  }

  bool TraverseObjCMessageExpr(ObjCMessageExpr *E) {
    // Do depth first; we want to rewrite the subexpressions first so that if
    // we have to move expressions we will move them already rewritten.
    for (Stmt::child_range range = E->children(); range; ++range)
      if (!TraverseStmt(*range))
        return false;

    return WalkUpFromObjCMessageExpr(E);
  }
};
}

void ObjCMigrateASTConsumer::migrateDecl(Decl *D) {
  if (!D)
    return;
  if (isa<ObjCMethodDecl>(D))
    return; // Wait for the ObjC container declaration.

  ObjCMigrator(*this).TraverseDecl(D);
}

namespace {

class RewritesReceiver : public edit::EditsReceiver {
  Rewriter &Rewrite;

public:
  RewritesReceiver(Rewriter &Rewrite) : Rewrite(Rewrite) { }

  virtual void insert(SourceLocation loc, StringRef text) {
    Rewrite.InsertText(loc, text);
  }
  virtual void replace(CharSourceRange range, StringRef text) {
    Rewrite.ReplaceText(range.getBegin(), Rewrite.getRangeSize(range), text);
  }
};

}

void ObjCMigrateASTConsumer::HandleTranslationUnit(ASTContext &Ctx) {
  Rewriter rewriter(Ctx.getSourceManager(), Ctx.getLangOpts());
  RewritesReceiver Rec(rewriter);
  Editor->applyRewrites(Rec);

  for (Rewriter::buffer_iterator
        I = rewriter.buffer_begin(), E = rewriter.buffer_end(); I != E; ++I) {
    FileID FID = I->first;
    RewriteBuffer &buf = I->second;
    const FileEntry *file = Ctx.getSourceManager().getFileEntryForID(FID);
    assert(file);
    llvm::SmallString<512> newText;
    llvm::raw_svector_ostream vecOS(newText);
    buf.write(vecOS);
    vecOS.flush();
    llvm::MemoryBuffer *memBuf = llvm::MemoryBuffer::getMemBufferCopy(
                   StringRef(newText.data(), newText.size()), file->getName());
    llvm::SmallString<64> filePath(file->getName());
    FileMgr.FixupRelativePath(filePath);
    Remapper.remap(filePath.str(), memBuf);
  }

  if (IsOutputFile) {
    Remapper.flushToFile(MigrateDir, Ctx.getDiagnostics());
  } else {
    Remapper.flushToDisk(MigrateDir, Ctx.getDiagnostics());
  }
}

bool MigrateSourceAction::BeginInvocation(CompilerInstance &CI) {
  CI.getDiagnostics().setIgnoreAllWarnings(true);
  CI.getPreprocessorOpts().DetailedRecord = true;
  CI.getPreprocessorOpts().DetailedRecordConditionalDirectives = true;
  return true;
}

ASTConsumer *MigrateSourceAction::CreateASTConsumer(CompilerInstance &CI,
                                                  StringRef InFile) {
  return new ObjCMigrateASTConsumer(CI.getFrontendOpts().OutputFile,
                                    /*MigrateLiterals=*/true,
                                    /*MigrateSubscripting=*/true,
                                    Remapper,
                                    CI.getFileManager(),
                                  CI.getPreprocessor().getPreprocessingRecord(),
                                    /*isOutputFile=*/true); 
}
