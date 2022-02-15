// RUN: %clang_cc1 -emit-llvm -triple x86_64 %s -o - | FileCheck %s

/// Set !retain regardless of the target. The backend will lower !retain to
/// SHF_GNU_RETAIN on ELF and ignore the metadata for other binary formats.
// CHECK:      @c0 ={{.*}} constant i32 {{.*}}
// CHECK:      @foo.l0 = internal global i32 {{.*}}
// CHECK:      @g0 ={{.*}} global i32 {{.*}}
// CHECK-NEXT: @g1 ={{.*}} global i32 {{.*}}
// CHECK-NEXT: @g3 = internal global i32 {{.*}}
// CHECK-NEXT: @g4 = internal global i32 0, section ".data.g"{{.*}}

// CHECK:      @llvm.used = appending global [8 x i8*] [i8* bitcast (i32* @c0 to i8*), i8* bitcast (i32* @foo.l0 to i8*), i8* bitcast (void ()* @f0 to i8*), i8* bitcast (void ()* @f2 to i8*), i8* bitcast (i32* @g0 to i8*), i8* bitcast (i32* @g1 to i8*), i8* bitcast (i32* @g3 to i8*), i8* bitcast (i32* @g4 to i8*)], section "llvm.metadata"
// CHECK:      @llvm.compiler.used = appending global [3 x i8*] [i8* bitcast (void ()* @f2 to i8*), i8* bitcast (i32* @g3 to i8*), i8* bitcast (i32* @g4 to i8*)], section "llvm.metadata"

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
