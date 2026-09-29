// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.cir.ll %s
// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  int a;

  int getA() {
    return this.a;
  }

  void setA(int v) {
    this.a = v;
  }
};

export int testGet() {
  S s;
  return s.getA();
}

// CIR-LABEL: cir.func {{.*}} @_ZN1S4getAEv
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca "this" {{.*}} : !cir.ptr<!cir.ptr<!rec_S>>
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:   %[[MEMBER:.*]] = cir.get_member %[[THIS]][0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
// CIR:   %[[VAL:.*]] = cir.load {{.*}} %[[MEMBER]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[VAL]], %[[RETVAL:.*]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return {{.*}} : !s32i

// LLVM-LABEL: define {{.*}} @_ZN1S4getAEv(
// LLVM-SAME: ptr {{.*}} %[[ARG0:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[GEP:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[VAL:.*]] = load i32, ptr %[[GEP]]
// LLVM:   ret i32 %{{.*}}

// OGCG-LABEL: define {{.*}} @_ZN1S4getAEv(
// OGCG-SAME: ptr {{.*}} %[[ARG0:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[GEP:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[VAL:.*]] = load i32, ptr %[[GEP]]
// OGCG:   ret i32 %[[VAL]]

export void testSet(int v) {
  S s;
  s.setA(v);
}

// CIR-LABEL: cir.func {{.*}} @_ZN1S4setAEi
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca "this" {{.*}} : !cir.ptr<!cir.ptr<!rec_S>>
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:   %[[MEMBER:.*]] = cir.get_member %[[THIS]][0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
// CIR:   cir.store align(1) {{.*}}, %[[MEMBER]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return

// LLVM-LABEL: define {{.*}} @_ZN1S4setAEi(
// LLVM-SAME: ptr {{.*}} %[[ARG0:.*]], i32 {{.*}} %[[ARG1:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[GEP:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[THIS]], i32 0, i32 0
// LLVM:   store i32 {{.*}}, ptr %[[GEP]]
// LLVM:   ret void

// OGCG-LABEL: define {{.*}} @_ZN1S4setAEi(
// OGCG-SAME: ptr {{.*}} %[[ARG0:.*]], i32 {{.*}} %[[ARG1:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[GEP:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[THIS]], i32 0, i32 0
// OGCG:   store i32 {{.*}}, ptr %[[GEP]]
// OGCG:   ret void
