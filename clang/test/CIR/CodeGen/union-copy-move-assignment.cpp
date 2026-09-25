// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

union U {
  int a;
  float b;
};

// Odr-use both defaulted assignment operators out of line so their bodies are
// emitted under both backends.
auto get_copy = static_cast<U &(U::*)(const U &)>(&U::operator=);
auto get_move = static_cast<U &(U::*)(U &&)>(&U::operator=);

// CIR: cir.func{{.*}}@_ZN1UaSERKS_{{.*}}cxx_assign<!rec_U, copy, trivial true>
// CIR:   cir.libc.memcpy {{.*}} align(4) to {{.*}} align(4)
// CIR: cir.func{{.*}}@_ZN1UaSEOS_{{.*}}cxx_assign<!rec_U, move, trivial true>
// CIR:   cir.libc.memcpy {{.*}} align(4) to {{.*}} align(4)

// LLVM: define linkonce_odr noundef nonnull align 4 dereferenceable(4) ptr @_ZN1UaSERKS_(ptr noundef nonnull align 4 dereferenceable(4) %{{.+}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.+}})
// LLVM:        call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.+}}, ptr align 4 %{{.+}}, i64 4, i1 false)
// LLVM: define linkonce_odr noundef nonnull align 4 dereferenceable(4) ptr @_ZN1UaSEOS_(ptr noundef nonnull align 4 dereferenceable(4) %{{.+}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.+}})
// LLVM:        call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.+}}, ptr align 4 %{{.+}}, i64 4, i1 false)
