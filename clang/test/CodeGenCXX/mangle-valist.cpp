#include "stdarg.h"

namespace test1 {
  void test1(const char *fmt, va_list ap) {
  }
}

class Test2 {
public:
  void test2(const char *fmt, va_list ap);
};

void Test2::test2(const char *fmt, va_list ap) {
}

// RUN: %clang_cc1 %s -emit-llvm -o - \
// RUN:     -triple armv7-unknown-linux \
// RUN:   | FileCheck -check-prefix=MANGLE-ARM-AAPCS %s
// CHECK-MANGLE-ARM-AAPCS: @_ZN5test15test1EPKcSt9__va_list
// CHECK-MANGLE-ARM-AAPCS: @_ZN5Test25test2EPKcSt9__va_list

// RUN: %clang_cc1 %s -emit-llvm -o - \
// RUN:     -triple armv7-unknown-linux -target-abi apcs-gnu \
// RUN:   | FileCheck -check-prefix=MANGLE-ARM-APCS %s
// CHECK-MANGLE-ARM-APCS: @_ZN5test15test1EPKcPv
// CHECK-MANGLE-ARM-APCS: @_ZN5Test25test2EPKcPv

// RUN: %clang_cc1 %s -emit-llvm -o - \
// RUN:     -triple mipsel-unknown-linux \
// RUN:   | FileCheck -check-prefix=MANGLE-MIPSEL %s
// CHECK-MANGLE-MIPSEL: @_ZN5test15test1EPKcPv
// CHECK-MANGLE-MIPSEL: @_ZN5Test25test2EPKcPv

// RUN: %clang_cc1 %s -emit-llvm -o - \
// RUN:     -triple i686-unknown-linux \
// RUN:   | FileCheck -check-prefix=MANGLE-X86 %s
// CHECK-MANGLE-X86: @_ZN5test15test1EPKcPc
// CHECK-MANGLE-X86: @_ZN5Test25test2EPKcPc

// RUN: %clang_cc1 %s -emit-llvm -o - \
// RUN:     -triple x86_64-unknown-linux \
// RUN:   | FileCheck -check-prefix=MANGLE-X86-64 %s
// CHECK-MANGLE-X86-64: @_ZN5test15test1EPKcP13__va_list_tag
// CHECK-MANGLE-X86-64: @_ZN5Test25test2EPKcP13__va_list_tag
