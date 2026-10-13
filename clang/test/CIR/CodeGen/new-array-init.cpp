// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

struct S {
  int a;
  int b;
};

// Value-init array new of a trivial type: () means zero-init via memset.
S *makeVar(unsigned n) { return new S[n](); }

// CIR-LABEL: cir.func{{.*}}@_Z7makeVarj
// CIR:   cir.call @_Znam(
// CIR:   cir.libc.memset {{.*}} bytes at {{.*}} to

// LLVM-LABEL: @_Z7makeVarj
// LLVM:   call {{.*}} ptr @_Znam(
// LLVM:   call void @llvm.memset.p0.i64(ptr{{.*}} %{{.*}}, i8 0, i64 %{{.*}}, i1 false)

// Constant element count: the size is folded to a constant.
S *makeConst() { return new S[4](); }

// CIR-LABEL: cir.func{{.*}}@_Z9makeConstv
// CIR:   cir.call @_Znam(
// CIR:   cir.libc.memset

// LLVM-LABEL: @_Z9makeConstv
// LLVM:   call {{.*}} ptr @_Znam(i64 noundef 32)
// LLVM:   call void @llvm.memset.p0.i64(ptr{{.*}} %{{.*}}, i8 0, i64 32, i1 false)

// No parens: default-init of a trivial type is a no-op (no memset).
S *makeNoInit(unsigned n) { return new S[n]; }

// CIR-LABEL: cir.func{{.*}}@_Z10makeNoInitj
// CIR:   cir.call @_Znam(
// CIR-NOT: cir.libc.memset

// LLVM-LABEL: @_Z10makeNoInitj
// LLVM:   call {{.*}} ptr @_Znam(
// LLVM-NOT: memset

// Braced-empty value-init goes through the InitListExpr path (also memset).
S *makeBraced(unsigned n) { return new S[n]{}; }

// CIR-LABEL: cir.func{{.*}}@_Z10makeBracedj
// CIR:   cir.call @_Znam(
// CIR:   cir.libc.memset

// LLVM-LABEL: @_Z10makeBracedj
// LLVM:   call {{.*}} ptr @_Znam(
// LLVM:   call void @llvm.memset.p0.i64(ptr{{.*}} %{{.*}}, i8 0, i64 %{{.*}}, i1 false)

// Non-zero-initializable element (pointer to data member): the null member
// representation is -1, not 0, so memset is not used; the constructor-loop
// path value-initializes each element to {0, -1}.
struct M {
  int x;
  int M::*p;
};

M *makeMember(unsigned n) { return new M[n](); }

// CIR-LABEL: cir.func{{.*}}@_Z10makeMemberj
// CIR:   cir.call @_Znam(
// CIR-NOT: cir.libc.memset
// CIR:   cir.const #cir.const_record<{#cir.int<0> : !s32i, #cir.int<-1> : !s64i}>

// LLVM-LABEL: @_Z10makeMemberj
// LLVM:   call {{.*}} ptr @_Znam(
// LLVMCIR: store %struct.M { i32 0, i64 -1 }, ptr %{{.*}}
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{.*}}, ptr align 16 @{{.*}}
