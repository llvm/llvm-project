// RUN: %clang_cc1 -emit-llvm -triple x86_64 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefix=GLOBALAS

/// Set !retain regardless of the target. The backend will lower !retain to
/// SHF_GNU_RETAIN on ELF and ignore the metadata for other binary formats.
// CHECK:      @c0 ={{.*}} constant i32 {{.*}}
// CHECK:      @foo.l0 = internal global i32 {{.*}}
// CHECK:      @g0 ={{.*}} global i32 {{.*}}
// CHECK-NEXT: @g1 ={{.*}} global i32 {{.*}}
// CHECK-NEXT: @g3 = internal global i32 {{.*}}
// CHECK-NEXT: @g4 = internal global i32 0, section ".data.g"{{.*}}

// CHECK:      @llvm.used = appending global [8 x ptr] [ptr @c0, ptr @foo.l0, ptr @f0, ptr @f2, ptr @g0, ptr @g1, ptr @g3, ptr @g4], section "llvm.metadata"
// CHECK:      @llvm.compiler.used = appending global [3 x ptr] [ptr @f2, ptr @g3, ptr @g4], section "llvm.metadata"
// GLOBALAS:   @llvm.used = appending addrspace(1) global [8 x ptr addrspace(1)] [ptr addrspace(1) addrspacecast (ptr addrspace(4) @c0 to ptr addrspace(1)), ptr addrspace(1) @foo.l0, ptr addrspace(1) addrspacecast (ptr @f0 to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @f2 to ptr addrspace(1)), ptr addrspace(1) @g0, ptr addrspace(1) @g1, ptr addrspace(1) @g3, ptr addrspace(1) @g4], section "llvm.metadata"
// GLOBALAS:   @llvm.compiler.used = appending addrspace(1) global [3 x ptr addrspace(1)] [ptr addrspace(1) addrspacecast (ptr @f2 to ptr addrspace(1)), ptr addrspace(1) @g3, ptr addrspace(1) @g4], section "llvm.metadata"
const int c0 __attribute__((retain)) = 42;

void foo(void) {
  static int l0 __attribute__((retain)) = 2;
}

__attribute__((retain)) int g0;
int g1 __attribute__((retain));
__attribute__((retain)) static int g2;
__attribute__((used, retain)) static int g3;
__attribute__((used, retain, section(".data.g"))) static int g4;

void __attribute__((retain)) f0(void) {}
static void __attribute__((retain)) f1(void) {}
static void __attribute__((used, retain)) f2(void) {}
