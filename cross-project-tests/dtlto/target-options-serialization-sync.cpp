// Verify that adding a TargetOptions field makes the real serialization guard
// fail to compile until the field is handled.
//
// REQUIRES: clang
// RUN: not %clangxx -std=c++17 -fsyntax-only \
// RUN:   -I%llvm_src_root/include -I%llvm_obj_root/include \
// RUN:   -I%llvm_src_root/lib/LTO %s 2>&1 | FileCheck %s

// Inject an extra field at the final TargetOptions field declaration. Undefine
// the macro before including the implementation so its structured binding
// still contains the production field list.
#define ObjectFilenameForDebug                                                \
  ObjectFilenameForDebug;                                                     \
  bool SerializationTestExtraField
#include "llvm/LTO/TargetOptionsBitcode.h"
#undef ObjectFilenameForDebug

#include "TargetOptionsBitcode.cpp"

// CHECK: type 'const TargetOptions' {{binds to|decomposes into}} 63 elements,
// CHECK-SAME: but only 62 names were provided
