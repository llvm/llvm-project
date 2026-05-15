// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct WithStaticMem {
  static void StaticMem();
};

extern "C" void use(WithStaticMem m) {
  // CIR-LABEL: use(
   auto x = m.StaticMem;
  // CIR: %[[ARG_ALLOCA:.*]] = cir.alloca !rec_WithStaticMem, !cir.ptr<!rec_WithStaticMem>, ["m", init]
  // CIR: %[[X_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.func<()>>, !cir.ptr<!cir.ptr<!cir.func<()>>>, ["x", init]
  // CIR: cir.store %{{.*}}, %[[ARG_ALLOCA]] : !rec_WithStaticMem, !cir.ptr<!rec_WithStaticMem>
  // CIR: %[[GET_STATIC_FUNC:.*]] = cir.get_global @_ZN13WithStaticMem9StaticMemEv : !cir.ptr<!cir.func<()>> 
  // CIR: cir.store {{.*}}%[[GET_STATIC_FUNC]], %[[X_ALLOCA]] : !cir.ptr<!cir.func<()>>, !cir.ptr<!cir.ptr<!cir.func<()>>>

   // LLVM-LABEL: use(
   // LLVM: %[[ARG_ALLOCA:.*]] = alloca %struct.WithStaticMem
   // LLVM: %[[X_ALLOCA:.*]] = alloca ptr
   // LLVM: store ptr @_ZN13WithStaticMem9StaticMemEv, ptr %[[X_ALLOCA]]
}

