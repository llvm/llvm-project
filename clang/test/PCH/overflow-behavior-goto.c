// RUN: %clang_cc1 -fexperimental-overflow-behavior-types -emit-pch -o %t %s
// RUN: %clang_cc1 -fexperimental-overflow-behavior-types -include-pch %t -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fexperimental-overflow-behavior-types -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -fexperimental-overflow-behavior-types -include-pch %t -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=IR

#ifndef HEADER
#define HEADER

// CHECK: return __builtin_overflow_goto(a + 1, too_big);
// DUMP: OverflowGotoExpr {{.*}} '__ob_trap int' too_big 0x{{[0-9a-f]+}}
// DUMP-NEXT: BinaryOperator {{.*}} '+'

static inline int add(int __ob_trap a) {
  return __builtin_overflow_goto(a + 1, too_big);
too_big:
  return -1;
}

#else

// IR-LABEL: define {{.*}} @add(
// IR: @llvm.sadd.with.overflow.i32
// IR: br label %too_big
int use(void) { return add(3); }

#endif
