// REQUIRES: x86-registered-target

/// Check that FatLTO with ASan currently emits a redundant instrumentation warning
/// for both Full LTO and ThinLTO at -O0 and -O2, while instrumenting both the
/// embedded bitcode and the native object code.

// RUN: %clang --target=x86_64-unknown-linux-gnu -flto=full -ffat-lto-objects -fsanitize=address -c %s -o %t.full.o0.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARN
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.full.o0.bc %t.full.o0.o
// RUN: llvm-dis < %t.full.o0.bc | FileCheck %s --check-prefix=IR
// RUN: llvm-nm %t.full.o0.o | FileCheck %s --check-prefix=OBJ

// RUN: %clang --target=x86_64-unknown-linux-gnu -O2 -flto=full -ffat-lto-objects -fsanitize=address -c %s -o %t.full.o2.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARN
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.full.o2.bc %t.full.o2.o
// RUN: llvm-dis < %t.full.o2.bc | FileCheck %s --check-prefix=IR
// RUN: llvm-nm %t.full.o2.o | FileCheck %s --check-prefix=OBJ

// RUN: %clang --target=x86_64-unknown-linux-gnu -flto=thin -ffat-lto-objects -fsanitize=address -c %s -o %t.thin.o0.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARN
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.thin.o0.bc %t.thin.o0.o
// RUN: llvm-dis < %t.thin.o0.bc | FileCheck %s --check-prefix=IR
// RUN: llvm-nm %t.thin.o0.o | FileCheck %s --check-prefix=OBJ

// RUN: %clang --target=x86_64-unknown-linux-gnu -O2 -flto=thin -ffat-lto-objects -fsanitize=address -c %s -o %t.thin.o2.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARN
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.thin.o2.bc %t.thin.o2.o
// RUN: llvm-dis < %t.thin.o2.bc | FileCheck %s --check-prefix=IR
// RUN: llvm-nm %t.thin.o2.o | FileCheck %s --check-prefix=OBJ

// WARN: warning: Redundant instrumentation detected, with module flag: nosanitize_address [-Wbackend-plugin]

// IR-DAG: define internal void @asan.module_ctor()
// IR-DAG: call void @__asan_report_load4
// IR-DAG: !{i32 4, !"nosanitize_address", i32 1}

// OBJ-DAG: __asan_init
// OBJ-DAG: __asan_report_load4
// OBJ-DAG: asan.module_ctor

int foo(int *a) {
  return *a;
}
