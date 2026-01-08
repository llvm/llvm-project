// RUN: %clang_cc1 -debug-info-kind=constructor -emit-llvm %s -o - \
// RUN:        | FileCheck %s -check-prefix=CTOR_HOMING
// RUN: %clang_cc1 -debug-info-kind=limited -fuse-ctor-homing -emit-llvm %s -o - \
// RUN:        | FileCheck %s -check-prefix=CTOR_HOMING
// RUN: %clang_cc1 -debug-info-kind=standalone -fuse-ctor-homing -emit-llvm %s -o - \
// RUN:        | FileCheck %s -check-prefix=FULL_DEBUG
// RUN: %clang_cc1 -debug-info-kind=line-tables-only -fuse-ctor-homing -emit-llvm %s -o - \
// RUN:        | FileCheck %s -check-prefix=NO_DEBUG
// RUN: %clang_cc1 -fuse-ctor-homing -emit-llvm %s -o - \
// RUN:        | FileCheck %s -check-prefix=NO_DEBUG
//
// RUN: %clang_cc1 -debug-info-kind=constructor -fno-use-ctor-homing \
// RUN:        -emit-llvm %s -o - | FileCheck %s -check-prefix=FULL_DEBUG

// This tests that the -fuse-ctor-homing is only used if limited debug info would have
// been used otherwise.

// CTOR_HOMING: !DICompositeType(tag: DW_TAG_structure_type, name: "A"{{.*}}flags: DIFlagFwdDecl
// FULL_DEBUG: !DICompositeType(tag: DW_TAG_structure_type, name: "A"{{.*}}DIFlagTypePassByValue
// NO_DEBUG-NOT: !DICompositeType(tag: DW_TAG_structure_type, name: "A"
struct A {
  A();
} TestA;
