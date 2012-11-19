//===--- Types.cpp - Driver input & temporary type information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Types.h"

#include "llvm/ADT/StringSwitch.h"
#include <string.h>
#include <cassert>

using namespace clang::driver;
using namespace clang::driver::types;

struct TypeInfo {
  const char *Name;
  const char *Flags;
  const char *TempSuffix;
  ID PreprocessedType;
};

static const TypeInfo TypeInfos[] = {
#define TYPE(NAME, ID, PP_TYPE, TEMP_SUFFIX, FLAGS) \
  { NAME, FLAGS, TEMP_SUFFIX, TY_##PP_TYPE, },
#include "clang/Driver/Types.def"
#undef TYPE
};
static const unsigned numTypes = sizeof(TypeInfos) / sizeof(TypeInfos[0]);

static const TypeInfo &getInfo(unsigned id) {
  assert(id > 0 && id - 1 < numTypes && "Invalid Type ID.");
  return TypeInfos[id - 1];
}

const char *types::getTypeName(ID Id) {
  return getInfo(Id).Name;
}

types::ID types::getPreprocessedType(ID Id) {
  return getInfo(Id).PreprocessedType;
}

const char *types::getTypeTempSuffix(ID Id) {
  return getInfo(Id).TempSuffix;
}

bool types::onlyAssembleType(ID Id) {
  return strchr(getInfo(Id).Flags, 'a');
}

bool types::onlyPrecompileType(ID Id) {
  return strchr(getInfo(Id).Flags, 'p');
}

bool types::canTypeBeUserSpecified(ID Id) {
  return strchr(getInfo(Id).Flags, 'u');
}

bool types::appendSuffixForType(ID Id) {
  return strchr(getInfo(Id).Flags, 'A');
}

bool types::canLipoType(ID Id) {
  return (Id == TY_Nothing ||
          Id == TY_Image ||
          Id == TY_Object ||
          Id == TY_LTO_BC);
}

bool types::isAcceptedByClang(ID Id) {
  switch (Id) {
  default:
    return false;

  case TY_Asm:
  case TY_C: case TY_PP_C:
  case TY_CL:
  case TY_CUDA:
  case TY_ObjC: case TY_PP_ObjC: case TY_PP_ObjC_Alias:
  case TY_CXX: case TY_PP_CXX:
  case TY_ObjCXX: case TY_PP_ObjCXX: case TY_PP_ObjCXX_Alias:
  case TY_CHeader: case TY_PP_CHeader:
  case TY_CLHeader:
  case TY_ObjCHeader: case TY_PP_ObjCHeader:
  case TY_CXXHeader: case TY_PP_CXXHeader:
  case TY_ObjCXXHeader: case TY_PP_ObjCXXHeader:
  case TY_AST:
  case TY_LLVM_IR: case TY_LLVM_BC:
    return true;
  }
}

bool types::isObjC(ID Id) {
  switch (Id) {
  default:
    return false;

  case TY_ObjC: case TY_PP_ObjC: case TY_PP_ObjC_Alias:
  case TY_ObjCXX: case TY_PP_ObjCXX:
  case TY_ObjCHeader: case TY_PP_ObjCHeader:
  case TY_ObjCXXHeader: case TY_PP_ObjCXXHeader: case TY_PP_ObjCXX_Alias:
    return true;
  }
}

bool types::isCXX(ID Id) {
  switch (Id) {
  default:
    return false;

  case TY_CXX: case TY_PP_CXX:
  case TY_ObjCXX: case TY_PP_ObjCXX:
  case TY_CXXHeader: case TY_PP_CXXHeader:
  case TY_ObjCXXHeader: case TY_PP_ObjCXXHeader:
  case TY_CUDA:
    return true;
  }
}

types::ID types::lookupTypeForExtension(const char *Ext) {
  return llvm::StringSwitch<types::ID>(Ext)
           .Case("c", TY_C)
           .Case("i", TY_PP_C)
           .Case("m", TY_ObjC)
           .Case("M", TY_ObjCXX)
           .Case("h", TY_CHeader)
           .Case("C", TY_CXX)
           .Case("H", TY_CXXHeader)
           .Case("f", TY_PP_Fortran)
           .Case("F", TY_Fortran)
           .Case("s", TY_PP_Asm)
           .Case("S", TY_Asm)
           .Case("o", TY_Object)
           .Case("ii", TY_PP_CXX)
           .Case("mi", TY_PP_ObjC)
           .Case("mm", TY_ObjCXX)
           .Case("bc", TY_LLVM_BC)
           .Case("cc", TY_CXX)
           .Case("CC", TY_CXX)
           .Case("cl", TY_CL)
           .Case("cp", TY_CXX)
           .Case("cu", TY_CUDA)
           .Case("hh", TY_CXXHeader)
           .Case("ll", TY_LLVM_IR)
           .Case("hpp", TY_CXXHeader)
           .Case("ads", TY_Ada)
           .Case("adb", TY_Ada)
           .Case("ast", TY_AST)
           .Case("c++", TY_CXX)
           .Case("C++", TY_CXX)
           .Case("cxx", TY_CXX)
           .Case("cpp", TY_CXX)
           .Case("CPP", TY_CXX)
           .Case("CXX", TY_CXX)
           .Case("for", TY_PP_Fortran)
           .Case("FOR", TY_PP_Fortran)
           .Case("fpp", TY_Fortran)
           .Case("FPP", TY_Fortran)
           .Case("f90", TY_PP_Fortran)
           .Case("f95", TY_PP_Fortran)
           .Case("F90", TY_Fortran)
           .Case("F95", TY_Fortran)
           .Case("mii", TY_PP_ObjCXX)
           .Default(TY_INVALID);
}

types::ID types::lookupTypeForTypeSpecifier(const char *Name) {
  unsigned N = strlen(Name);

  for (unsigned i=0; i<numTypes; ++i) {
    types::ID Id = (types::ID) (i + 1);
    if (canTypeBeUserSpecified(Id) &&
        memcmp(Name, getInfo(Id).Name, N + 1) == 0)
      return Id;
  }

  return TY_INVALID;
}

// FIXME: Why don't we just put this list in the defs file, eh.

unsigned types::getNumCompilationPhases(ID Id) {
  if (Id == TY_Object)
    return 1;

  unsigned N = 0;
  if (getPreprocessedType(Id) != TY_INVALID)
    N += 1;

  if (onlyAssembleType(Id))
    return N + 2; // assemble, link
  if (onlyPrecompileType(Id))
    return N + 1; // precompile

  return N + 3; // compile, assemble, link
}

phases::ID types::getCompilationPhase(ID Id, unsigned N) {
  assert(N < getNumCompilationPhases(Id) && "Invalid index.");

  if (Id == TY_Object)
    return phases::Link;

  if (getPreprocessedType(Id) != TY_INVALID) {
    if (N == 0)
      return phases::Preprocess;
    --N;
  }

  if (onlyAssembleType(Id))
    return N == 0 ? phases::Assemble : phases::Link;

  if (onlyPrecompileType(Id))
    return phases::Precompile;

  if (N == 0)
    return phases::Compile;
  if (N == 1)
    return phases::Assemble;

  return phases::Link;
}

ID types::lookupCXXTypeForCType(ID Id) {
  switch (Id) {
  default:
    return Id;
    
  case types::TY_C:
    return types::TY_CXX;
  case types::TY_PP_C:
    return types::TY_PP_CXX;
  case types::TY_CHeader:
    return types::TY_CXXHeader;
  case types::TY_PP_CHeader:
    return types::TY_PP_CXXHeader;
  }
}
