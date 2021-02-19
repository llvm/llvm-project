// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64 -S -o - < %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64 -S -fbasic-block-sections=all -fbasic-block-sections=none -o - < %s | FileCheck %s --check-prefix=PLAIN

// RUN: %clang_cc1 -triple x86_64 -S -fbasic-block-sections=all -o - < %s | FileCheck %s --check-prefix=BB_WORLD --check-prefix=BB_ALL
// RUN: %clang_cc1 -triple x86_64 -S -fbasic-block-sections=list=%S/Inputs/basic-block-sections.funcnames -o - < %s | FileCheck %s --check-prefix=BB_WORLD --check-prefix=BB_LIST
// RUN: %clang_cc1 -triple x86_64 -S -fbasic-block-sections=all -funique-basic-block-section-names -o - < %s | FileCheck %s --check-prefix=UNIQUE
// RUN: rm -f %t
// RUN: not %clang_cc1 -fbasic-block-sections=list= -emit-obj -o %t %s 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=ERROR
// RUN: not ls %t

int world(int a) {
  if (a > 10)
    return 10;
  else if (a > 5)
    return 5;
  else
    return 0;
}

int another(int a) {
  if (a > 10)
    return 20;
  return 0;
}

// PLAIN-NOT: section
// PLAIN: world:
//
// BB_WORLD: .section .text.world,"ax",@progbits{{$}}
// BB_WORLD: world:
// BB_WORLD: .section .text.world,"ax",@progbits,unique
// BB_WORLD: world.__part.1:
// BB_ALL: .section .text.another,"ax",@progbits
// BB_ALL: another.__part.1:
// BB_LIST-NOT: .section .text.another,"ax",@progbits
// BB_LIST: another:
// BB_LIST-NOT: another.__part.1:
//
// UNIQUE: .section .text.world.world.__part.1,
// UNIQUE: .section .text.another.another.__part.1,
// ERROR: error:  unable to load basic block sections function list: '[[MSG]]'
