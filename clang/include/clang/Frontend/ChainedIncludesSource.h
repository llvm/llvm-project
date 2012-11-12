//===- ChainedIncludesSource.h - Chained PCHs in Memory ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ChainedIncludesSource class, which converts headers
//  to chained PCHs in memory, mainly used for testing.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SERIALIZATION_CHAINEDINCLUDESSOURCE_H
#define LLVM_CLANG_SERIALIZATION_CHAINEDINCLUDESSOURCE_H

#include "clang/Sema/ExternalSemaSource.h"
#include <vector>

namespace clang {
  class CompilerInstance;

class ChainedIncludesSource : public ExternalSemaSource {
public:
  virtual ~ChainedIncludesSource();

  static ChainedIncludesSource *create(CompilerInstance &CI);

private:
  ExternalSemaSource &getFinalReader() const { return *FinalReader; }

  std::vector<CompilerInstance *> CIs;
  OwningPtr<ExternalSemaSource> FinalReader;

  
protected:

//===----------------------------------------------------------------------===//
// ExternalASTSource interface.
//===----------------------------------------------------------------------===//

  virtual Decl *GetExternalDecl(uint32_t ID);
  virtual Selector GetExternalSelector(uint32_t ID);
  virtual uint32_t GetNumExternalSelectors();
  virtual Stmt *GetExternalDeclStmt(uint64_t Offset);
  virtual CXXBaseSpecifier *GetExternalCXXBaseSpecifiers(uint64_t Offset);
  virtual DeclContextLookupResult
  FindExternalVisibleDeclsByName(const DeclContext *DC, DeclarationName Name);
  virtual ExternalLoadResult FindExternalLexicalDecls(const DeclContext *DC,
                                        bool (*isKindWeWant)(Decl::Kind),
                                        SmallVectorImpl<Decl*> &Result);
  virtual void CompleteType(TagDecl *Tag);
  virtual void CompleteType(ObjCInterfaceDecl *Class);
  virtual void StartedDeserializing();
  virtual void FinishedDeserializing();
  virtual void StartTranslationUnit(ASTConsumer *Consumer);
  virtual void PrintStats();

  /// Return the amount of memory used by memory buffers, breaking down
  /// by heap-backed versus mmap'ed memory.
  virtual void getMemoryBufferSizes(MemoryBufferSizes &sizes) const;

//===----------------------------------------------------------------------===//
// ExternalSemaSource interface.
//===----------------------------------------------------------------------===//

  virtual void InitializeSema(Sema &S);
  virtual void ForgetSema();
  virtual void ReadMethodPool(Selector Sel);
  virtual bool LookupUnqualified(LookupResult &R, Scope *S);
};

}

#endif
