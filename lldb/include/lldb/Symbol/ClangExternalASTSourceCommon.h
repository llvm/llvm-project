//===-- ClangExternalASTSourceCommon.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExternalASTSourceCommon_h
#define liblldb_ClangExternalASTSourceCommon_h

// Clang headers like to use NDEBUG inside of them to enable/disable debug
// related features using "#ifndef NDEBUG" preprocessor blocks to do one thing
// or another. This is bad because it means that if clang was built in release
// mode, it assumes that you are building in release mode which is not always
// the case. You can end up with functions that are defined as empty in header
// files when NDEBUG is not defined, and this can cause link errors with the
// clang .a files that you have since you might be missing functions in the .a
// file. So we have to define NDEBUG when including clang headers to avoid any
// mismatches. This is covered by rdar://problem/8691220

#if !defined(NDEBUG) && !defined(LLVM_NDEBUG_OFF)
#define LLDB_DEFINED_NDEBUG_FOR_CLANG
#define NDEBUG
// Need to include assert.h so it is as clang would expect it to be (disabled)
#include <assert.h>
#endif

#ifdef LLDB_DEFINED_NDEBUG_FOR_CLANG
#undef NDEBUG
#undef LLDB_DEFINED_NDEBUG_FOR_CLANG
// Need to re-include assert.h so it is as _we_ would expect it to be (enabled)
#include <assert.h>
#endif

#include "clang/AST/ExternalASTSource.h"

#include "lldb/Core/dwarf.h"
#include "lldb/Symbol/ClangASTMetadata.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"

namespace lldb_private {

class ClangExternalASTSourceCommon : public clang::ExternalASTSource {

  /// LLVM-style RTTI.
  static char ID;

public:
  ~ClangExternalASTSourceCommon() override;

  ClangASTMetadata *GetMetadata(const clang::Decl *object);
  void SetMetadata(const clang::Decl *object,
                   const ClangASTMetadata &metadata) {
    m_decl_metadata[object] = metadata;
  }

  ClangASTMetadata *GetMetadata(const clang::Type *object);
  void SetMetadata(const clang::Type *object,
                   const ClangASTMetadata &metadata) {
    m_type_metadata[object] = metadata;
  }

  /// LLVM-style RTTI.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || ExternalASTSource::isA(ClassID);
  }
  static bool classof(const ExternalASTSource *S) { return S->isA(&ID); }
  /// \}
private:
  typedef llvm::DenseMap<const clang::Decl *, ClangASTMetadata> DeclMetadataMap;
  typedef llvm::DenseMap<const clang::Type *, ClangASTMetadata> TypeMetadataMap;

  DeclMetadataMap m_decl_metadata;
  TypeMetadataMap m_type_metadata;
};

} // namespace lldb_private

#endif // liblldb_ClangExternalASTSourceCommon_h
