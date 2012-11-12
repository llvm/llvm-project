//===--- ASTConsumer.h - Abstract interface for reading ASTs ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTConsumer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTCONSUMER_H
#define LLVM_CLANG_AST_ASTCONSUMER_H

namespace clang {
  class ASTContext;
  class CXXRecordDecl;
  class DeclGroupRef;
  class HandleTagDeclDefinition;
  class PPMutationListener;
  class ASTMutationListener;
  class ASTDeserializationListener; // layering violation because void* is ugly
  class SemaConsumer; // layering violation required for safe SemaConsumer
  class TagDecl;
  class VarDecl;
  class FunctionDecl;
  class ImportDecl;

/// ASTConsumer - This is an abstract interface that should be implemented by
/// clients that read ASTs.  This abstraction layer allows the client to be
/// independent of the AST producer (e.g. parser vs AST dump file reader, etc).
class ASTConsumer {
  /// \brief Whether this AST consumer also requires information about
  /// semantic analysis.
  bool SemaConsumer;

  friend class SemaConsumer;

public:
  ASTConsumer() : SemaConsumer(false) { }

  virtual ~ASTConsumer() {}

  /// Initialize - This is called to initialize the consumer, providing the
  /// ASTContext.
  virtual void Initialize(ASTContext &Context) {}

  /// HandleTopLevelDecl - Handle the specified top-level declaration.  This is
  /// called by the parser to process every top-level Decl*. Note that D can be
  /// the head of a chain of Decls (e.g. for `int a, b` the chain will have two
  /// elements). Use Decl::getNextDeclarator() to walk the chain.
  ///
  /// \returns true to continue parsing, or false to abort parsing.
  virtual bool HandleTopLevelDecl(DeclGroupRef D);

  /// HandleInterestingDecl - Handle the specified interesting declaration. This
  /// is called by the AST reader when deserializing things that might interest
  /// the consumer. The default implementation forwards to HandleTopLevelDecl.
  virtual void HandleInterestingDecl(DeclGroupRef D);

  /// HandleTranslationUnit - This method is called when the ASTs for entire
  /// translation unit have been parsed.
  virtual void HandleTranslationUnit(ASTContext &Ctx) {}

  /// HandleTagDeclDefinition - This callback is invoked each time a TagDecl
  /// (e.g. struct, union, enum, class) is completed.  This allows the client to
  /// hack on the type, which can occur at any point in the file (because these
  /// can be defined in declspecs).
  virtual void HandleTagDeclDefinition(TagDecl *D) {}

  /// \brief Invoked when a function is implicitly instantiated.
  /// Note that at this point point it does not have a body, its body is
  /// instantiated at the end of the translation unit and passed to
  /// HandleTopLevelDecl.
  virtual void HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) {}

  /// \brief Handle the specified top-level declaration that occurred inside
  /// and ObjC container.
  /// The default implementation ignored them.
  virtual void HandleTopLevelDeclInObjCContainer(DeclGroupRef D);

  /// \brief Handle an ImportDecl that was implicitly created due to an
  /// inclusion directive.
  /// The default implementation passes it to HandleTopLevelDecl.
  virtual void HandleImplicitImportDecl(ImportDecl *D);

  /// CompleteTentativeDefinition - Callback invoked at the end of a translation
  /// unit to notify the consumer that the given tentative definition should be
  /// completed.
  ///
  /// The variable declaration itself will be a tentative
  /// definition. If it had an incomplete array type, its type will
  /// have already been changed to an array of size 1. However, the
  /// declaration remains a tentative definition and has not been
  /// modified by the introduction of an implicit zero initializer.
  virtual void CompleteTentativeDefinition(VarDecl *D) {}

  /// HandleCXXStaticMemberVarInstantiation - Tell the consumer that this
  // variable has been instantiated.
  virtual void HandleCXXStaticMemberVarInstantiation(VarDecl *D) {}

  /// \brief Callback involved at the end of a translation unit to
  /// notify the consumer that a vtable for the given C++ class is
  /// required.
  ///
  /// \param RD The class whose vtable was used.
  ///
  /// \param DefinitionRequired Whether a definition of this vtable is
  /// required in this translation unit; otherwise, it is only needed if
  /// it was actually used.
  virtual void HandleVTable(CXXRecordDecl *RD, bool DefinitionRequired) {}

  /// \brief If the consumer is interested in preprocessor entities getting
  /// modified after their initial creation, it should return a pointer to
  /// a PPMutationListener here.
  virtual PPMutationListener *GetPPMutationListener() { return 0; }

  /// \brief If the consumer is interested in entities getting modified after
  /// their initial creation, it should return a pointer to
  /// an ASTMutationListener here.
  virtual ASTMutationListener *GetASTMutationListener() { return 0; }

  /// \brief If the consumer is interested in entities being deserialized from
  /// AST files, it should return a pointer to a ASTDeserializationListener here
  virtual ASTDeserializationListener *GetASTDeserializationListener() {
    return 0;
  }

  /// PrintStats - If desired, print any statistics.
  virtual void PrintStats() {}
};

} // end namespace clang.

#endif
