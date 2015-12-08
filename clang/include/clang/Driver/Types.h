//===--- Types.h - Input & Temporary Driver Types ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_TYPES_H
#define LLVM_CLANG_DRIVER_TYPES_H

#include "clang/Driver/Phases.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace driver {
namespace types {
  enum ID {
    TY_INVALID,
#define TYPE(NAME, ID, PP_TYPE, TEMP_SUFFIX, FLAGS) TY_##ID,
#include "clang/Driver/Types.def"
#undef TYPE
    TY_LAST
  };

  /// getTypeName - Return the name of the type for \p Id.
  const char *getTypeName(ID Id);

  /// getPreprocessedType - Get the ID of the type for this input when
  /// it has been preprocessed, or INVALID if this input is not
  /// preprocessed.
  ID getPreprocessedType(ID Id);

  /// getTypeTempSuffix - Return the suffix to use when creating a
  /// temp file of this type, or null if unspecified.
  const char *getTypeTempSuffix(ID Id, bool CLMode = false);

  /// onlyAssembleType - Should this type only be assembled.
  bool onlyAssembleType(ID Id);

  /// onlyPrecompileType - Should this type only be precompiled.
  bool onlyPrecompileType(ID Id);

  /// canTypeBeUserSpecified - Can this type be specified on the
  /// command line (by the type name); this is used when forwarding
  /// commands to gcc.
  bool canTypeBeUserSpecified(ID Id);

  /// appendSuffixForType - When generating outputs of this type,
  /// should the suffix be appended (instead of replacing the existing
  /// suffix).
  bool appendSuffixForType(ID Id);

  /// canLipoType - Is this type acceptable as the output of a
  /// universal build (currently, just the Nothing, Image, and Object
  /// types).
  bool canLipoType(ID Id);

  /// isAcceptedByClang - Can clang handle this input type.
  bool isAcceptedByClang(ID Id);

  /// isCXX - Is this a "C++" input (C++ and Obj-C++ sources and headers).
  bool isCXX(ID Id);

  /// Is this LLVM IR.
  bool isLLVMIR(ID Id);

  /// isCuda - Is this a CUDA input.
  bool isCuda(ID Id);

  /// isObjC - Is this an "ObjC" input (Obj-C and Obj-C++ sources and headers).
  bool isObjC(ID Id);

  /// lookupTypeForExtension - Lookup the type to use for the file
  /// extension \p Ext.
  ID lookupTypeForExtension(const char *Ext);

  /// lookupTypeForTypSpecifier - Lookup the type to use for a user
  /// specified type name.
  ID lookupTypeForTypeSpecifier(const char *Name);

  /// getCompilationPhases - Get the list of compilation phases ('Phases') to be
  /// done for type 'Id'.
  void getCompilationPhases(
    ID Id,
    llvm::SmallVectorImpl<phases::ID> &Phases);

  /// lookupCXXTypeForCType - Lookup CXX input type that corresponds to given
  /// C type (used for clang++ emulation of g++ behaviour)
  ID lookupCXXTypeForCType(ID Id);

} // end namespace types
} // end namespace driver
} // end namespace clang

#endif
