//===--- Mangle.h - Mangle C++ Names ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the C++ name mangling interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MANGLE_H
#define LLVM_CLANG_AST_MANGLE_H

#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
  class ASTContext;
  class BlockDecl;
  class CXXConstructorDecl;
  class CXXDestructorDecl;
  class CXXMethodDecl;
  class FunctionDecl;
  class NamedDecl;
  class ObjCMethodDecl;
  class VarDecl;
  struct ThisAdjustment;
  struct ThunkInfo;

/// MangleBuffer - a convenient class for storing a name which is
/// either the result of a mangling or is a constant string with
/// external memory ownership.
class MangleBuffer {
public:
  void setString(StringRef Ref) {
    String = Ref;
  }

  SmallVectorImpl<char> &getBuffer() {
    return Buffer;
  }

  StringRef getString() const {
    if (!String.empty()) return String;
    return Buffer.str();
  }

  operator StringRef() const {
    return getString();
  }

private:
  StringRef String;
  SmallString<256> Buffer;
};

/// MangleContext - Context for tracking state which persists across multiple
/// calls to the C++ name mangler.
class MangleContext {
  virtual void anchor();

  ASTContext &Context;
  DiagnosticsEngine &Diags;

  llvm::DenseMap<const BlockDecl*, unsigned> GlobalBlockIds;
  llvm::DenseMap<const BlockDecl*, unsigned> LocalBlockIds;
  
public:
  explicit MangleContext(ASTContext &Context,
                         DiagnosticsEngine &Diags)
    : Context(Context), Diags(Diags) { }

  virtual ~MangleContext() { }

  ASTContext &getASTContext() const { return Context; }

  DiagnosticsEngine &getDiags() const { return Diags; }

  virtual void startNewFunction() { LocalBlockIds.clear(); }
  
  unsigned getBlockId(const BlockDecl *BD, bool Local) {
    llvm::DenseMap<const BlockDecl *, unsigned> &BlockIds
      = Local? LocalBlockIds : GlobalBlockIds;
    std::pair<llvm::DenseMap<const BlockDecl *, unsigned>::iterator, bool>
      Result = BlockIds.insert(std::make_pair(BD, BlockIds.size()));
    return Result.first->second;
  }
  
  /// @name Mangler Entry Points
  /// @{

  virtual bool shouldMangleDeclName(const NamedDecl *D) = 0;
  virtual void mangleName(const NamedDecl *D, raw_ostream &)=0;
  virtual void mangleThunk(const CXXMethodDecl *MD,
                          const ThunkInfo &Thunk,
                          raw_ostream &) = 0;
  virtual void mangleCXXDtorThunk(const CXXDestructorDecl *DD, CXXDtorType Type,
                                  const ThisAdjustment &ThisAdjustment,
                                  raw_ostream &) = 0;
  virtual void mangleReferenceTemporary(const VarDecl *D,
                                        raw_ostream &) = 0;
  virtual void mangleCXXVTable(const CXXRecordDecl *RD,
                               raw_ostream &) = 0;
  virtual void mangleCXXVTT(const CXXRecordDecl *RD,
                            raw_ostream &) = 0;
  virtual void mangleCXXCtorVTable(const CXXRecordDecl *RD, int64_t Offset,
                                   const CXXRecordDecl *Type,
                                   raw_ostream &) = 0;
  virtual void mangleCXXRTTI(QualType T, raw_ostream &) = 0;
  virtual void mangleCXXRTTIName(QualType T, raw_ostream &) = 0;
  virtual void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                             raw_ostream &) = 0;
  virtual void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                             raw_ostream &) = 0;

  void mangleGlobalBlock(const BlockDecl *BD,
                         const NamedDecl *ID,
                         raw_ostream &Out);
  void mangleCtorBlock(const CXXConstructorDecl *CD, CXXCtorType CT,
                       const BlockDecl *BD, raw_ostream &Out);
  void mangleDtorBlock(const CXXDestructorDecl *CD, CXXDtorType DT,
                       const BlockDecl *BD, raw_ostream &Out);
  void mangleBlock(const DeclContext *DC, const BlockDecl *BD,
                   raw_ostream &Out);
  // Do the right thing.
  void mangleBlock(const BlockDecl *BD, raw_ostream &Out,
                   const NamedDecl *ID=0);

  void mangleObjCMethodName(const ObjCMethodDecl *MD,
                            raw_ostream &);

  // This is pretty lame.
  virtual void mangleItaniumGuardVariable(const VarDecl *D,
                                          raw_ostream &) {
    llvm_unreachable("Target does not support mangling guard variables");
  }
  /// @}
};

MangleContext *createItaniumMangleContext(ASTContext &Context,
                                          DiagnosticsEngine &Diags);
MangleContext *createMicrosoftMangleContext(ASTContext &Context,
                                            DiagnosticsEngine &Diags);

}

#endif
