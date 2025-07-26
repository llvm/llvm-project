// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -debug-info-kind=limited -dwarf-version=4 -debugger-tuning=gdb -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-RV64
// RUN: %clang_cc1 -triple riscv64 -debug-info-kind=limited -dwarf-version=4 -debugger-tuning=lldb -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-RV64
// RUN: %clang_cc1 -triple riscv64 -debug-info-kind=limited -dwarf-version=5 -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-RV64

// RUN: %clang_cc1 -triple riscv32 -debug-info-kind=limited -dwarf-version=4 -debugger-tuning=gdb -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-RV32
// RUN: %clang_cc1 -triple riscv32 -debug-info-kind=limited -dwarf-version=4 -debugger-tuning=lldb -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-RV32
// RUN: %clang_cc1 -triple riscv32 -debug-info-kind=limited -dwarf-version=5 -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-RV32

// CHECK-RV64: declare !dbg ![[FOO:[0-9]+]] void @_Z3foov() local_unnamed_addr #[[ATTR0:[0-9]+]]
// CHECK-RV32: declare !dbg ![[FOO:[0-9]+]] void @_Z3foov() local_unnamed_addr #[[ATTR0:[0-9]+]]
void foo();

// CHECK-RV64: declare !dbg ![[BAR:[0-9]+]] void @_Z3barv() local_unnamed_addr #[[ATTR0:[0-9]+]]
// CHECK-RV32: declare !dbg ![[BAR:[0-9]+]] void @_Z3barv() local_unnamed_addr #[[ATTR0:[0-9]+]]
void bar(void);

// CHECK-RV64: declare !dbg ![[BAZ:[0-9]+]] void @_Z3baziz(i32 noundef signext, ...) local_unnamed_addr #[[ATTR0:[0-9]+]]
// CHECK-RV32: declare !dbg ![[BAZ:[0-9]+]] void @_Z3baziz(i32 noundef, ...) local_unnamed_addr #[[ATTR0:[0-9]+]]
void baz(int a, ...);

int main() {
  foo();
  bar();
  baz(1);
  return 0;
}
