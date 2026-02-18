// RUN: %clang_cc1 -std=c++11 -fclangir -emit-cir -o %t-cxx11.cir -triple x86_64-linux-gnu %s
// RUN: FileCheck %s --input-file=%t-cxx11.cir --check-prefixes=CIR,CIR-CXX11
// RUN: %clang_cc1 -std=c++2a -fclangir -emit-cir -o %t-cxx2a.cir -triple x86_64-linux-gnu %s
// RUN: FileCheck %s --input-file=%t-cxx2a.cir --check-prefixes=CIR,CIR-CXX2A
// RUN: %clang_cc1 -std=c++11 -fclangir -emit-llvm -o %t-cxx11-cir.ll -triple x86_64-linux-gnu %s
// RUN: FileCheck %s --input-file=%t-cxx11-cir.ll --check-prefixes=LLVM,LLVM-CXX11
// RUN: %clang_cc1 -std=c++2a -fclangir -emit-llvm -o %t-cxx2a-cir.ll -triple x86_64-linux-gnu %s
// RUN: FileCheck %s --input-file=%t-cxx2a-cir.ll --check-prefixes=LLVM,LLVM-CXX2A
// RUN: %clang_cc1 -std=c++11 -emit-llvm -o %t-cxx11.ll -triple x86_64-linux-gnu %s
// RUN: FileCheck %s --input-file=%t-cxx11.ll --check-prefixes=OGCG,OGCG-CXX11
// RUN: %clang_cc1 -std=c++2a -emit-llvm -o %t-cxx2a.ll -triple x86_64-linux-gnu %s
// RUN: FileCheck %s --input-file=%t-cxx2a.ll --check-prefixes=OGCG,OGCG-CXX2A

// CIR-DAG: cir.global "private" constant cir_private @[[F_A:.*]] = #cir.const_record<{#cir.int<1> : !s32i, #cir.const_array<[#cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i]> : !cir.array<!s32i x 3>}> : !rec_A
// LLVM-DAG: @[[F_A:.*]] = private constant {{.*}} { i32 1, [2 x i32] [i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6] }
// OGCG-DAG: @__const._Z1fi.a = private unnamed_addr constant {{.*}} { i32 1, [2 x i32] [i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6] }

// CIR-CXX11-DAG: cir.global "private" constant cir_private @_ZN7PR422765State1mE.const = #cir.const_array<[#cir.const_record<{#cir.global_view<@_ZN7PR422765State2f1Ev> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct, #cir.const_record<{#cir.global_view<@_ZN7PR422765State2f2Ev> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct]> : !cir.array<!rec_anon_struct x 2>
// LLVM-CXX11-DAG: @_ZN7PR422765State1mE.const = private constant [2 x { i64, i64 }] [{ {{.*}} @_ZN7PR422765State2f1Ev {{.*}}, i64 0 }, { {{.*}} @_ZN7PR422765State2f2Ev {{.*}}, i64 0 }]
// OGCG-CXX11-DAG: @_ZN7PR422765State1mE.const = private unnamed_addr constant [2 x { i64, i64 }] [{ {{.*}} @_ZN7PR422765State2f1Ev {{.*}}, i64 0 }, { {{.*}} @_ZN7PR422765State2f2Ev {{.*}}, i64 0 }]

// CIR-CXX2A-DAG: cir.global constant linkonce_odr comdat @_ZN7PR422765State1mE = #cir.const_array<[#cir.const_record<{#cir.global_view<@_ZN7PR422765State2f1Ev> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct, #cir.const_record<{#cir.global_view<@_ZN7PR422765State2f2Ev> : !s64i, #cir.int<0> : !s64i}> : !rec_anon_struct]> : !cir.array<!rec_anon_struct x 2>
// LLVM-CXX2A-DAG: @_ZN7PR422765State1mE = linkonce_odr constant [2 x { i64, i64 }] [{ {{.*}} @_ZN7PR422765State2f1Ev {{.*}}, i64 0 }, { {{.*}} @_ZN7PR422765State2f2Ev {{.*}}, i64 0 }], comdat
// OGCG-CXX2A-DAG: @_ZN7PR422765State1mE = linkonce_odr constant [2 x { i64, i64 }] [{ {{.*}} @_ZN7PR422765State2f1Ev {{.*}}, i64 0 }, { {{.*}} @_ZN7PR422765State2f2Ev {{.*}}, i64 0 }], comdat

// In OGCG, f1() is emitted before the lambda.
// OGCG-LABEL: define{{.*}} i32 @_Z1fi(
// OGCG:         call void {{.*}}memcpy{{.*}}({{.*}}, {{.*}} @__const._Z1fi.a
// OGCG:         call{{.*}} i32 @"_ZZ1fiENK3$_0clEiM1Ai"(ptr {{.*}}, i32 {{.*}}, i64 0)

struct A { int x, y[2]; int arr[3]; };
int f(int i) {
  constexpr A a = {1, 2, 3, 4, 5, 6};

  // CIR-LABEL: cir.func {{.*}}@_ZZ1fiENK3$_0clEiM1Ai(
  // LLVM-LABEL: define {{.*}}@"_ZZ1fiENK3$_0clEiM1Ai"(
  // OGCG-LABEL: define {{.*}}@"_ZZ1fiENK3$_0clEiM1Ai"(
    return [] (int n, int A::*p) {
    // CIR:  cir.ternary
    // LLVM: br i1
    // OGCG: br i1
    return (n >= 0
      // CIR:  %[[A:.*]] = cir.get_global @[[F_A]] : !cir.ptr<!rec_A>
      // CIR:  %[[ARR:.*]] = cir.get_member %[[A]][2] {name = "arr"} : !cir.ptr<!rec_A> -> !cir.ptr<!cir.array<!s32i x 3>>
      // CIR:  cir.get_element %[[ARR]][%{{.*}} : !s32i] : !cir.ptr<!cir.array<!s32i x 3>> -> !cir.ptr<!s32i>
      // LLVM: getelementptr [3 x i32], ptr getelementptr inbounds nuw (i8, ptr @[[F_A]], i64 12), i32 0, i64 %{{.*}}
      // OGCG: getelementptr inbounds [3 x i32], ptr getelementptr inbounds nuw ({{.*}} @__const._Z1fi.a, i32 0, i32 2), i64 0, i64 %{{.*}}
      ? a.arr[n]
      // CIR:  cir.ternary
      // LLVM: br i1
      // OGCG: br i1
      : (n == -1
        // CIR: %[[A:.*]] = cir.get_global @[[F_A]] : !cir.ptr<!rec_A>
        // CIR: %[[N:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s64i>, !s64i
        // CIR: %[[A_BYTE_PTR:.*]] = cir.cast bitcast %[[A]] : !cir.ptr<!rec_A> -> !cir.ptr<!s8i>
        // CIR: cir.ptr_stride %[[A_BYTE_PTR]], %[[N]] : (!cir.ptr<!s8i>, !s64i) -> !cir.ptr<!s8i>

        // LLVM: getelementptr i8, ptr @[[F_A]], i64 %{{.*}}
        // LLVM: load i32

        // OGCG: getelementptr inbounds i8, ptr @__const._Z1fi.a, i64 %{{.*}}
        // OGCG: load i32
        ? a.*p
        // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
        // CIR: %[[N:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s32i>, !s32i
        // CIR: %[[SUB:.*]] = cir.binop(sub, %[[TWO]], %[[N]]) nsw : !s32i
        // CIR: %[[A:.*]] = cir.get_global @[[F_A]] : !cir.ptr<!rec_A>
        // CIR: %[[Y:.*]] = cir.get_member %[[A]][1] {name = "y"} : !cir.ptr<!rec_A> -> !cir.ptr<!cir.array<!s32i x 2>>
        // CIR: cir.get_element %[[Y]][%[[SUB]] : !s32i] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>

        // LLVM: getelementptr [2 x i32], ptr getelementptr inbounds nuw ({{.*}} @[[F_A]], i64 4), i32 0, i64 %{{.*}}
        // LLVM: load i32

        // OGCG: getelementptr inbounds [2 x i32], ptr getelementptr inbounds nuw ({{.*}} @__const._Z1fi.a, i32 0, i32 1), i64 0, i64 %{{.*}}
        // OGCG: load i32
        : a.y[2 - n]));
  }(i, &A::x);
}

// With CIR, f1() is emitted after the lambda.
// CIR-LABEL: cir.func {{.*}} @_Z1fi(
// CIR:         %[[A_ADDR:.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a", init, const]
// CIR:         %[[A_INIT:.*]] = cir.get_global @[[F_A]] : !cir.ptr<!rec_A>
// CIR:         cir.copy %[[A_INIT]] to %[[A_ADDR]]
// CIR:         %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CIR:         cir.call @_ZZ1fiENK3$_0clEiM1Ai({{.*}}, {{.*}}, %[[ZERO]])

// LLVM-LABEL: define{{.*}} i32 @_Z1fi(
// LLVM:         call void @llvm.memcpy{{.*}}({{.*}}, ptr @[[F_A]]
// LLVM:         call{{.*}} i32 @"_ZZ1fiENK3$_0clEiM1Ai"(ptr %{{.*}}, i32 %{{.*}}, i64 0)

namespace PR42276 {
  class State {
    void syncDirtyObjects();
    void f1(), f2();
    using l = void (State::*)();
    static constexpr l m[]{&State::f1, &State::f2};
  };
  // CIR-CXX11-LABEL: cir.func {{.*}} @_ZN7PR422765State2f1Ev(!cir.ptr<!rec_PR422763A3AState>)
  // CIR-CXX11-LABEL: cir.func {{.*}} @_ZN7PR422765State2f2Ev(!cir.ptr<!rec_PR422763A3AState>)
  //
  // LLVM-CXX11-LABEL: declare{{.*}} @_ZN7PR422765State2f1Ev(ptr)
  // LLVM-CXX11-LABEL: declare{{.*}} @_ZN7PR422765State2f2Ev(ptr)
  //
  // OG-Codegen always generates these deferred, not only if they are non-const.
  //
  // CIR-LABEL: cir.func {{.*}} @_ZN7PR422765State16syncDirtyObjectsEv(
  // LLVM-LABEL: define{{.*}} void @_ZN7PR422765State16syncDirtyObjectsEv(
  // OGCG-LABEL: define{{.*}} void @_ZN7PR422765State16syncDirtyObjectsEv(
    void State::syncDirtyObjects() {
    for (int i = 0; i < sizeof(m) / sizeof(m[0]); ++i)
      // CIR-CXX11: %[[M:.*]] = cir.get_global @_ZN7PR422765State1mE.const : !cir.ptr<!cir.array<!rec_anon_struct x 2>>
      // CIR-CXX2A: %[[M:.*]] = cir.get_global @_ZN7PR422765State1mE : !cir.ptr<!cir.array<!rec_anon_struct x 2>>
      // CIR: %[[M_I:.*]] = cir.get_element %[[M]][%{{.*}} : !s32i] : !cir.ptr<!cir.array<!rec_anon_struct x 2>> -> !cir.ptr<!rec_anon_struct>

      // LLVM-CXX11: getelementptr [2 x { i64, i64 }], ptr @_ZN7PR422765State1mE.const, i32 0, i64 %{{.*}}
      // LLVM-CXX2A: getelementptr [2 x { i64, i64 }], ptr @_ZN7PR422765State1mE, i32 0, i64 %{{.*}}
      // OGCG-CXX11: getelementptr inbounds [2 x { i64, i64 }], ptr @_ZN7PR422765State1mE.const, i64 0, i64 %{{.*}}
      // OGCG-CXX2A: getelementptr inbounds [2 x { i64, i64 }], ptr @_ZN7PR422765State1mE, i64 0, i64 %{{.*}}
      (this->*m[i])();
  }
  // CIR-CXX2A-LABEL: cir.func {{.*}} @_ZN7PR422765State2f1Ev(!cir.ptr<!rec_PR422763A3AState>)
  // CIR-CXX2A-LABEL: cir.func {{.*}} @_ZN7PR422765State2f2Ev(!cir.ptr<!rec_PR422763A3AState>)
  //
  // LLVM-CXX2A-LABEL: declare{{.*}} @_ZN7PR422765State2f1Ev(ptr)
  // LLVM-CXX2A-LABEL: declare{{.*}} @_ZN7PR422765State2f2Ev(ptr)
  //
  // OGCG-LABEL: declare{{.*}} @_ZN7PR422765State2f1Ev(ptr{{.*}})
  // OGCG-LABEL: declare{{.*}} @_ZN7PR422765State2f2Ev(ptr{{.*}})
  //
}
