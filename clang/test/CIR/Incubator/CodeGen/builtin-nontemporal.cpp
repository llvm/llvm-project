// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

int nontemporal_load(const int *ptr) {
  return __builtin_nontemporal_load(ptr);
}

// CIR-LABEL: @_Z16nontemporal_loadPKi
// CIR: %{{.+}} = cir.load nontemporal{{.*}}  %{{.+}} : !cir.ptr<!s32i>, !s32i

// LLVM-LABEL: @_Z16nontemporal_loadPKi
// LLVM: %{{.+}} = load i32, ptr %{{.+}}, align 4, !nontemporal !1

void nontemporal_store(int *ptr, int value) {
  __builtin_nontemporal_store(value, ptr);
}

// CIR-LABEL: @_Z17nontemporal_storePii
// CIR: cir.store nontemporal{{.*}} %{{.+}}, %{{.+}} : !s32i, !cir.ptr<!s32i>

// LLVM-LABEL: @_Z17nontemporal_storePii
// LLVM: store i32 %{{.+}}, ptr %{{.+}}, align 4, !nontemporal !1
