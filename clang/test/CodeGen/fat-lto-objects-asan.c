// REQUIRES: x86-registered-target

/// Check that FatLTO with ASan instruments both the embedded bitcode section
/// and the native object code without running sanitizer passes twice or
/// emitting redundant instrumentation warnings.

// RUN: %clang --target=x86_64-unknown-linux-gnu -Werror=backend-plugin -Xclang -fdebug-pass-manager \
// RUN:   -flto=full -ffat-lto-objects -fsanitize=address -c %s -o %t.full.o0.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PM
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.full.o0.bc %t.full.o0.o
// RUN: llvm-dis < %t.full.o0.bc | FileCheck %s --check-prefix=IR
// RUN: llvm-nm %t.full.o0.o | FileCheck %s --check-prefix=OBJ

// RUN: %clang --target=x86_64-unknown-linux-gnu -Werror=backend-plugin -Xclang -fdebug-pass-manager \
// RUN:   -O2 -flto=full -ffat-lto-objects -fsanitize=address -c %s -o %t.full.o2.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PM
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.full.o2.bc %t.full.o2.o
// RUN: llvm-dis < %t.full.o2.bc | FileCheck %s --check-prefix=IR
// RUN: llvm-nm %t.full.o2.o | FileCheck %s --check-prefix=OBJ

// RUN: %clang --target=x86_64-unknown-linux-gnu -Werror=backend-plugin -Xclang -fdebug-pass-manager \
// RUN:   -flto=thin -ffat-lto-objects -fsanitize=address -c %s -o %t.thin.o0.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PM
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.thin.o0.bc %t.thin.o0.o
// RUN: llvm-dis < %t.thin.o0.bc | FileCheck %s --check-prefix=IR
// RUN: llvm-nm %t.thin.o0.o | FileCheck %s --check-prefix=OBJ

// RUN: %clang --target=x86_64-unknown-linux-gnu -Werror=backend-plugin -Xclang -fdebug-pass-manager \
// RUN:   -O2 -flto=thin -ffat-lto-objects -fsanitize=address -c %s -o %t.thin.o2.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PM
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.thin.o2.bc %t.thin.o2.o
// RUN: llvm-dis < %t.thin.o2.bc | FileCheck %s --check-prefix=IR
// RUN: llvm-nm %t.thin.o2.o | FileCheck %s --check-prefix=OBJ

// RUN: %clang --target=x86_64-unknown-linux-gnu -Werror=backend-plugin -Xclang -fdebug-pass-manager \
// RUN:   -O2 -flto=thin -ffat-lto-objects -fsanitize=address -mllvm -sanitizer-early-opt-ep \
// RUN:   -c %s -o %t.early.o 2>&1 | FileCheck %s --check-prefix=PM

// RUN: %clang --target=x86_64-unknown-linux-gnu -Werror=backend-plugin -Xclang -fdebug-pass-manager \
// RUN:   -O2 -flto=thin -ffat-lto-objects -fsanitize=address -fsanitize-coverage=trace-pc-guard \
// RUN:   -c %s -o %t.sancov.o 2>&1 | FileCheck %s --check-prefixes=PM,SANCOV

// PM-NOT: Redundant instrumentation detected
// SANCOV: Running pass: SanitizerCoveragePass on [module]
// PM: Running pass: AddressSanitizerPass on [module]
// SANCOV-NOT: Running pass: SanitizerCoveragePass on [module]
// PM-NOT: Running pass: AddressSanitizerPass on [module]
// PM-NOT: Redundant instrumentation detected

// IR-DAG: define internal void @asan.module_ctor()
// IR-DAG: call void @__asan_report_load4
// IR-DAG: !{i32 4, !"nosanitize_address", i32 1}

// OBJ-DAG: __asan_init
// OBJ-DAG: __asan_report_load4
// OBJ-DAG: asan.module_ctor

int foo(int *a) {
  return *a;
}
