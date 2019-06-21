// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-tapi-elf-v1 %s | \
// RUN: FileCheck %s

// RUN: %clang -target x86_64-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-yaml-elf-v1 %s | \
// RUN: FileCheck --check-prefix=CHECK-YAML %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c %s | llvm-nm - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-SYMBOLS %s

// CHECK: Symbols:
// CHECK-DAG:  _Z8weakFuncv: { Type: Func, Weak: true }
// CHECK-DAG:  _Z10strongFuncv: { Type: Func }

// CHECK-YAML: Symbols:
// CHECK-YAML-DAG:   - Name:            _Z8weakFuncv
// CHECK-YAML-DAG:     Type:            STT_FUNC
// CHECK-YAML-DAG:     Binding:         STB_WEAK
// CHECK-YAML-DAG:   - Name:            _Z10strongFuncv
// CHECK-YAML-DAG:     Type:            STT_FUNC
// CHECK-YAML-DAG:     Binding:         STB_GLOBAL

// CHECK-SYMBOLS-DAG: _Z10strongFuncv
// CHECK-SYMBOLS-DAG: _Z8weakFuncv
__attribute__((weak)) void weakFunc() {}
int strongFunc() { return 42; }
