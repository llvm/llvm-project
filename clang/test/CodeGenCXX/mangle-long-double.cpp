// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefix=POWER64-LINUX
// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu   %s -emit-llvm -o - | FileCheck %s --check-prefix=POWER-LINUX
// RUN: %clang_cc1 -triple s390x-unknown-linux-gnu     %s -emit-llvm -o - | FileCheck %s --check-prefix=S390X-LINUX

void f(long double) {}
// POWER64-LINUX:  _Z1fg
// POWER-LINUX:    _Z1fg
// S390X-LINUX:    _Z1fg
