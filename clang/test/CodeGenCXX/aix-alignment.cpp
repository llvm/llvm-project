// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:     -emit-llvm -o - -x c++ %s | \
// RUN:   FileCheck %s --check-prefixes=AIX,AIX32
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:     -emit-llvm -o - %s -x c++| \
// RUN:   FileCheck %s --check-prefixes=AIX,AIX64

struct B {
  double d;
  ~B() {}
};

// AIX32: %call = call noalias noundef nonnull ptr @_Znam(i32 noundef 8)
// AIX64: %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 8)
B *allocBp() { return new B[0]; }

// AIX-LABEL: delete.notnull:
// AIX32: %0 = getelementptr inbounds i8, ptr %call, i32 -8
// AIX32: [[PTR:%.+]] = getelementptr inbounds i8, ptr %0, i32 4
// AIX64: [[PTR:%.+]] = getelementptr inbounds i8, ptr %call, i64 -8
// AIX:   %{{.+}} = load i{{[0-9]+}}, ptr [[PTR]]
void bar() { delete[] allocBp(); }

typedef struct D {
  double d;
  int i;

  ~D(){};
} D;

// AIX: define void @_Z3foo1D(ptr dead_on_unwind noalias writable sret(%struct.D) align 4 %agg.result, ptr noundef %x)
// AIX32  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.result, ptr align 4 %x, i32 16, i1 false)
// AIX64: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.result, ptr align 4 %x, i64 16, i1 false)
D foo(D x) { return x; }
