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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
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
public:
  enum ManglerKind {
    MK_Itanium,
    MK_Microsoft
  };

private:
  virtual void anchor();

  ASTContext &Context;
  DiagnosticsEngine &Diags;
  const ManglerKind Kind;

  llvm::DenseMap<const BlockDecl*, unsigned> GlobalBlockIds;
  llvm::DenseMap<const BlockDecl*, unsigned> LocalBlockIds;

public:
  ManglerKind getKind() const { return Kind; }

  explicit MangleContext(ASTContext &Context,
                         DiagnosticsEngine &Diags,
                         ManglerKind Kind)
      : Context(Context), Diags(Diags), Kind(Kind) {}

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

  bool shouldMangleDeclName(const NamedDecl *D);
  virtual bool shouldMangleCXXName(const NamedDecl *D) = 0;

  // FIXME: consider replacing raw_ostream & with something like SmallString &.
  void mangleName(const NamedDecl *D, raw_ostream &);
  virtual void mangleCXXName(const NamedDecl *D, raw_ostream &) = 0;
  virtual void mangleThunk(const CXXMethodDecl *MD,
                          const ThunkInfo &Thunk,
                          raw_ostream &) = 0;
  virtual void mangleCXXDtorThunk(const CXXDestructorDecl *DD, CXXDtorType Type,
                                  const ThisAdjustment &ThisAdjustment,
                                  raw_ostream &) = 0;
  virtual void mangleReferenceTemporary(const VarDecl *D,
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

  void mangleObjCMethodName(const ObjCMethodDecl *MD, raw_ostream &);

  virtual void mangleStaticGuardVariable(const VarDecl *D, raw_ostream &) = 0;

  virtual void mangleDynamicInitializer(const VarDecl *D, raw_ostream &) = 0;

  virtual void mangleDynamicAtExitDestructor(const VarDecl *D,
                                             raw_ostream &) = 0;

  /// Generates a unique string for an externally visible type for use with TBAA
  /// or type uniquing.
  /// TODO: Extend this to internal types by generating names that are unique
  /// across translation units so it can be used with LTO.
  virtual void mangleTypeName(QualType T, raw_ostream &) = 0;

  /// @}
};

class ItaniumMangleContext : public MangleContext {
public:
  explicit ItaniumMangleContext(ASTContext &C, DiagnosticsEngine &D)
      : MangleContext(C, D, MK_Itanium) {}

  virtual void mangleCXXVTable(const CXXRecordDecl *RD, raw_ostream &) = 0;
  virtual void mangleCXXVTT(const CXXRecordDecl *RD, raw_ostream &) = 0;
  virtual void mangleCXXCtorVTable(const CXXRecordDecl *RD, int64_t Offset,
                                   const CXXRecordDecl *Type,
                                   raw_ostream &) = 0;
  virtual void mangleItaniumThreadLocalInit(const VarDecl *D,
                                            raw_ostream &) = 0;
  virtual void mangleItaniumThreadLocalWrapper(const VarDecl *D,
                                               raw_ostream &) = 0;

  static bool classof(const MangleContext *C) {
    return C->getKind() == MK_Itanium;
  }

  static ItaniumMangleContext *create(ASTContext &Context,
                                      DiagnosticsEngine &Diags);
};

class MicrosoftMangleContext : public MangleContext {
public:
  explicit MicrosoftMangleContext(ASTContext &C, DiagnosticsEngine &D)
      : MangleContext(C, D, MK_Microsoft) {}

  /// \brief Mangle vftable symbols.  Only a subset of the bases along the path
  /// to the vftable are included in the name.  It's up to the caller to pick
  /// them correctly.
  virtual void mangleCXXVFTable(const CXXRecordDecl *Derived,
                                ArrayRef<const CXXRecordDecl *> BasePath,
                                raw_ostream &Out) = 0;

  /// \brief Mangle vbtable symbols.  Only a subset of the bases along the path
  /// to the vbtable are included in the name.  It's up to the caller to pick
  /// them correctly.
  virtual void mangleCXXVBTable(const CXXRecordDecl *Derived,
                                ArrayRef<const CXXRecordDecl *> BasePath,
                                raw_ostream &Out) = 0;

  virtual void mangleVirtualMemPtrThunk(const CXXMethodDecl *MD,
                                        uint64_t OffsetInVFTable,
                                        raw_ostream &) = 0;

  static bool classof(const MangleContext *C) {
    return C->getKind() == MK_Microsoft;
  }

  static MicrosoftMangleContext *create(ASTContext &Context,
                                        DiagnosticsEngine &Diags);
};
}

#endif
