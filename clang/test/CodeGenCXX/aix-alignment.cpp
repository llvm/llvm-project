// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix \
// RUN:     -emit-llvm -o - -x c++ %s | \
// RUN:   FileCheck %s --check-prefixes=AIX,AIX32
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-unknown-aix \
// RUN:     -emit-llvm -o - %s -x c++| \
// RUN:   FileCheck %s --check-prefixes=AIX,AIX64

struct B {
  double d;
  ~B() {}
};

// AIX32: %call = call noalias noundef nonnull i8* @_Znam(i32 noundef 8)
// AIX64: %call = call noalias noundef nonnull i8* @_Znam(i64 noundef 8)
B *allocBp() { return new B[0]; }

// AIX-LABEL: delete.notnull:
// AIX32: %0 = bitcast %struct.B* %call to i8*
// AIX32: %1 = getelementptr inbounds i8, i8* %0, i32 -8
// AIX32: %2 = getelementptr inbounds i8, i8* %1, i32 4
// AIX32: %3 = bitcast i8* %2 to i32*
// AIX64: %0 = bitcast %struct.B* %call to i8*
// AIX64: %1 = getelementptr inbounds i8, i8* %0, i64 -8
// AIX64: %2 = bitcast i8* %1 to i64*
void bar() { delete[] allocBp(); }

typedef struct D {
  double d;
  int i;

  ~D(){};
} D;

// AIX: define void @_Z3foo1D(%struct.D* noalias sret(%struct.D) align 4 %agg.result, %struct.D* noundef %x)
// AIX:   %1 = bitcast %struct.D* %agg.result to i8*
// AIX:   %2 = bitcast %struct.D* %x to i8*
// AIX32  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %2, i32 16, i1 false)
// AIX64: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %1, i8* align 4 %2, i64 16, i1 false)
D foo(D x) { return x; }
