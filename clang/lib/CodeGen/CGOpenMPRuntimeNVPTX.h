//===----- CGOpenMPRuntimeNVPTX.h - Interface to OpenMP NVPTX Runtimes ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to NVPTX
// targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H

#include "CGOpenMPRuntime.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeNVPTX : public CGOpenMPRuntime {
public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM);
};

} // CodeGen namespace.
} // clang namespace.

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
