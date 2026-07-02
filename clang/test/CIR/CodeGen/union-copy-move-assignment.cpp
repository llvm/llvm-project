// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

union U {
  int a;
  float b;
};

// Odr-use both defaulted assignment operators out of line so their bodies are
// emitted under both backends.
auto get_copy = static_cast<U &(U::*)(const U &)>(&U::operator=);
auto get_move = static_cast<U &(U::*)(U &&)>(&U::operator=);

// The defaulted union copy/move assignment operators copy the object
// representation; CIR lowers that to a memcpy.  LLVM lowering uses a memcpy
// libcall; the classic backend uses the llvm.memcpy intrinsic (the divergence
// is the pre-existing builtin-memcpy lowering, not this feature).

// CIR: cir.func{{.*}}@_ZN1UaSERKS_{{.*}}cxx_assign<!rec_U, copy, trivial true>
// CIR:   cir.call @memcpy(
// CIR: cir.func{{.*}}@_ZN1UaSEOS_{{.*}}cxx_assign<!rec_U, move, trivial true>
// CIR:   cir.call @memcpy(

// LLVMCIR: define{{.*}}ptr @_ZN1UaSERKS_
// LLVMCIR:   call ptr @memcpy(ptr {{.*}}, ptr {{.*}}, i64 noundef 4)
// LLVMCIR: define{{.*}}ptr @_ZN1UaSEOS_
// LLVMCIR:   call ptr @memcpy(ptr {{.*}}, ptr {{.*}}, i64 noundef 4)

// OGCG: define{{.*}}ptr @_ZN1UaSERKS_
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 4, i1 false)
// OGCG: define{{.*}}ptr @_ZN1UaSEOS_
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}, i64 4, i1 false)
