// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

template <typename T>
struct IsChar {
  constexpr operator bool() const { return false; }
};

template<>
struct IsChar<char> {
  constexpr operator bool() const { return true; }
};

template <typename T>
concept SameAsChar = (bool)IsInt<T>();

// LLVM-DAG: [[STRUCT_A:%.*]] = type { i8, double }
// CIR-DAG: ![[STRUCT_A:.*]] = !cir.record<struct "A" {!s8i, !cir.double}>
struct A {
  char i;
  double j;

  template <SameAsChar T>
  operator T() const { return i; };
};

// LLVM-DAG: [[STRUCT_B:%.*]] = type { [[STRUCT_A]], i32 }
// CIR-DAG: ![[STRUCT_B:.*]] = !cir.record<struct "B" {![[STRUCT_A]], !s32i}>
struct B {
  A a;
  int b;
};

// LLVM-DAG: [[STRUCT_C:%.*]] = type <{ [[STRUCT_B]], [[STRUCT_A]], i32, [4 x i8] }>
// CIR-DAG: ![[STRUCT_C:.*]] = !cir.record<struct "C" packed padded {![[STRUCT_B]], ![[STRUCT_A]], !s32i, !cir.array<!u8i x 4>}>
struct C : public B, public A {
  int c;
};

// LLVM-DAG: [[STRUCT_D:%.*]] = type { [[STRUCT_A]], [[STRUCT_A]], i8, [[STRUCT_A]] }
// CIR-DAG: ![[STRUCT_D:.*]] = !cir.record<struct "D" {![[STRUCT_A]], ![[STRUCT_A]], !u8i, ![[STRUCT_A]]}>
struct D {
  A a;
  A b = A{2, 2.0};
  unsigned : 2;
  A c;
};

// LLVM-DAG: [[STRUCT_E:%.*]] = type { i32, ptr }
// CIR-DAG: ![[STRUCT_E:.*]] = !cir.record<struct "E" {!s32i, !cir.ptr<!s8i>}>
struct E {
  int a;
  const char* fn = __builtin_FUNCTION();
  ~E() {};
};

// CIR-DAG: ![[STRUCT_F:.*]] = !cir.record<struct "F" padded {!u8i}>
struct F {
  F (int i = 1);
  F (const F &f) = delete;
  F (F &&f) = default;
};

// LLVM-DAG: [[STRUCT_G:%.*]] = type <{ i32, [4 x i8] }>
// CIR-DAG: ![[STRUCT_G:.*]] = !cir.record<struct "G" packed padded {!s32i, !cir.array<!u8i x 4>}>
struct G {
  int a;
  F f;
};

// LLVM-DAG: [[UNION_U:%.*]] = type { [[STRUCT_A]] }
// LLVM-DAG: [[STR:@.*]] = private {{.*}}constant [6 x i8] {{.*}}foo18{{.*}}, align 1
// CIR-DAG: ![[UNION_U:.*]] = !cir.record<union "U" {!u8i, ![[STRUCT_A]], !s8i}>
union U {
  unsigned : 1;
  A a;
  char b;
};


namespace gh61145 {
  // LLVM-DAG: [[STRUCT_VEC:%.*Vec.*]] = type { i8 }
  // CIR-DAG: ![[STRUCT_VEC:.*]] = !cir.record<struct "gh61145::Vec" padded {!u8i}>
  struct Vec {
    Vec();
    Vec(Vec&&);
    ~Vec();
  };

  // LLVM-DAG: [[STRUCT_S1:%.*]] = type { i8 }
  // CIR-DAG: ![[STRUCT_S1:.*]] = !cir.record<struct "gh61145::S1" padded {!u8i}>
  struct S1 {
    Vec v;
  };

  // LLVM-DAG: [[STRUCT_S2:%.*]] = type { i8, i8 }
  // CIR-DAG: ![[STRUCT_S2:.*]] = !cir.record<struct "gh61145::S2" padded {!u8i, !s8i}>
  struct S2 {
    Vec v;
    char c;
  };
}

namespace gh62266 {
  // LLVM-DAG: [[STRUCT_H:%.*H.*]] = type { i32, i32 }
  // CIR-DAG: ![[STRUCT_H:.*]] = !cir.record<struct "gh62266::H<2>" {!s32i, !s32i}>
  template <int J>
  struct H {
    int i;
    int j = J;
  };
}

namespace gh61567 {
  // LLVM-DAG: [[STRUCT_I:%.*I.*]] = type { i32, ptr }
  // CIR-DAG: ![[STRUCT_I:.*]] = !cir.record<struct "gh61567::I" {!s32i, !cir.ptr<!s32i>}>
  struct I {
    int a;
    int&& r = 2;
  };
}

// LLVM-DAG: [[A1:@.*a1.*]] = internal constant [[STRUCT_A]] { i8 3, double 2.000000e+00 }, align 8
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL2a1 = #cir.const_record<{#cir.int<3> : !s8i, #cir.fp<2.000000e+00> : !cir.double}> : ![[STRUCT_A]] {alignment = 8 : i64}
constexpr A a1(3.1, 2.0);
// LLVM-DAG: [[A2:@.*a2.*]] = internal constant [[STRUCT_A]] { i8 99, double 0.000000e+00 }, align 8
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL2a2 = #cir.const_record<{#cir.int<99> : !s8i, #cir.fp<0.000000e+00> : !cir.double}> : ![[STRUCT_A]] {alignment = 8 : i64}
constexpr auto a2 = static_cast<A>('c');
// LLVM-DAG: [[B1:@.*b1.*]] = internal constant [[STRUCT_B]] { [[STRUCT_A]] { i8 99, double 0.000000e+00 }, i32 0 }, align 8
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL2b1 = #cir.const_record<{#cir.const_record<{#cir.int<99> : !s8i, #cir.fp<0.000000e+00> : !cir.double}> : ![[STRUCT_A]], #cir.int<0> : !s32i}> : ![[STRUCT_B]] {alignment = 8 : i64}
constexpr B b1(A('c'));
// LLVM-DAG: [[C1:@.*c1.*]] = internal constant { [[STRUCT_A]], i32, [4 x i8], i8, double, i32 } { [[STRUCT_A]] { i8 99, double 0.000000e+00 }, i32 0, [4 x i8] {{.*}}, i8 3, double 2.000000e+00, i32 0 }, align
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL2c1 = #cir.const_record<{#cir.const_record<{#cir.int<99> : !s8i, #cir.fp<0.000000e+00> : !cir.double}> : ![[STRUCT_A]], #cir.int<0> : !s32i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 4>, #cir.int<3> : !s8i, #cir.fp<2.000000e+00> : !cir.double, #cir.int<0> : !s32i}>
constexpr C c1(b1, a1);
// LLVM-DAG: [[U1:@.*]] = internal constant {{.*}} { [[STRUCT_A]] { i8 1, double 1.000000e+00 } }, align 8
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL2u1 = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s8i, #cir.fp<1.000000e+00> : !cir.double}> : ![[STRUCT_A]]}> : !{{.*}}{alignment = 8 : i64}
constexpr U u1(A(1, 1));
// LLVM-DAG: [[D1:@.*d1.*]] = internal constant { [[STRUCT_A]], [[STRUCT_A]], [8 x i8], [[STRUCT_A]] } { [[STRUCT_A]] { i8 2, double 2.000000e+00 }, [[STRUCT_A]] { i8 2, double 2.000000e+00 }, [8 x i8] {{.*}}, [[STRUCT_A]] zeroinitializer }, align 8
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL2d1 = #cir.const_record<{#cir.const_record<{#cir.int<2> : !s8i, #cir.fp<2.000000e+00> : !cir.double}> : ![[STRUCT_A]], #cir.const_record<{#cir.int<2> : !s8i, #cir.fp<2.000000e+00> : !cir.double}> : ![[STRUCT_A]], #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 8>, #cir.zero : ![[STRUCT_A]]}>
constexpr D d1(A(2, 2));
// LLVM-DAG: [[ARR1:@.*arr1.*]] = internal constant [3 x i32] [i32 1, i32 2, i32 0], align 4
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL4arr1 = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 3> {alignment = 4 : i64}
constexpr int arr1[3](1, 2);
// LLVM-DAG: [[ARR4:@.*arr4.*]] = internal constant [1 x i32] [i32 1], align 4
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL4arr4 = #cir.const_array<[#cir.int<1> : !s32i]> : !cir.array<!s32i x 1> {alignment = 4 : i64}
constexpr int arr4[](1);
// LLVM-DAG: [[ARR5:@.*arr5.*]] = internal constant [2 x i32] [i32 2, i32 0], align 4
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL4arr5 = #cir.const_array<[#cir.int<2> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 2> {alignment = 4 : i64}
constexpr int arr5[2](2);

// LLVM: define dso_local {{.*}} @{{.*foo1.*}}
// LLVM: [[RETVAL:%.*]] = alloca [[STRUCT_A]]
// LLVM-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[RETVAL]], ptr {{.*}}[[A1]], i64 16, i1 false)
// LLVM-NEXT: [[TMP_0:%.*]] = load {{.*}}, ptr [[RETVAL]], align 8
// LLVM-NEXT: ret {{.*}}[[TMP_0]]
// CIR-LABEL: cir.func {{.*}}@_Z4foo1v()
// CIR: %[[A_ALLOCA:.*]] = cir.alloca ![[STRUCT_A]], !cir.ptr<![[STRUCT_A]]>, ["__retval"] {alignment = 8 : i64}
// CIR: %[[GET_A1:.*]] = cir.get_global @_ZL2a1 : !cir.ptr<![[STRUCT_A]]>
// CIR: cir.copy %[[GET_A1]] to %[[A_ALLOCA]] : !cir.ptr<![[STRUCT_A]]>
A foo1() {
  return a1;
}

// LLVM: define dso_local {{.*}}@{{.*foo2.*}}
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}[[B1]], i64 24, i1 false)
// CIR: cir.func {{.*}}@_Z4foo2v()
// CIR: %[[B_ALLOCA:.*]] = cir.alloca ![[STRUCT_B]], !cir.ptr<![[STRUCT_B]]>, ["__retval"] {alignment = 8 : i64}
// CIR: %[[GET_GLOB:.*]] = cir.get_global @_ZL2b1 : !cir.ptr<![[STRUCT_B]]>
// CIR: cir.copy %[[GET_GLOB]] to %[[B_ALLOCA]] : !cir.ptr<![[STRUCT_B]]>
B foo2() {
  return b1;
}

// LLVM: define dso_local {{.*}}@{{.*foo3.*}}
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}[[C1]], i64 48, i1 false)
// CIR: cir.func {{.*}}@_Z4foo3v()
// CIR: %[[C_ALLOCA:.*]] = cir.alloca ![[STRUCT_C]], !cir.ptr<![[STRUCT_C]]>, ["__retval"] {alignment = 8 : i64}
// CIR: %[[GET_GLOB:.*]] = cir.get_global @_ZL2c1
// CIR: %[[GLOB_CAST:.*]] = cir.cast bitcast %[[GET_GLOB]] : !cir.ptr<!{{.*}}> -> !cir.ptr<![[STRUCT_C]]>
// CIR: cir.copy %[[GLOB_CAST]] to %[[C_ALLOCA]] : !cir.ptr<![[STRUCT_C]]>
C foo3() {
  return c1;
}

// LLVM: define dso_local void @{{.*foo4.*}}
// LLVM-DAG: [[C2:%.*]] = alloca [[STRUCT_C]]
// LLVM-DAG: [[REF_TMP:%.*]] = alloca [[STRUCT_B]]
// LLVM-DAG: [[REF_TMP_1:%.*]] = alloca [[STRUCT_A]]
// LLVM: [[A:%.*]] = getelementptr {{.*}}[[STRUCT_B]], ptr [[REF_TMP]], i32 0, i32 0
// LLVM-NEXT: [[I:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[A]], i32 0, i32 0
// LLVM-NEXT: store i8 1, ptr [[I]], align 8
// LLVM-NEXT: [[J:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[A]], i32 0, i32 1
// LLVM-NEXT: store double 1.000000e+00, ptr [[J]], align 8
// LLVM-NEXT: [[B:%.*]] = getelementptr {{.*}}[[STRUCT_B]], ptr [[REF_TMP]], i32 0, i32 1
// LLVM-NEXT: store i32 1, ptr [[B]], align 8
// LLVM-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[C2]], ptr {{.*}}[[REF_TMP]], i64 24, i1 false)
// LLVM-NEXT: [[TMP_0:%.*]] = getelementptr {{.*}}i8, ptr [[C2]], i{{.*}} 24
// LLVM-NEXT: [[I2:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[REF_TMP_1]], i32 0, i32 0
// LLVM-NEXT: store i8 97, ptr [[I2]], align 8
// LLVM-NEXT: [[J3:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[REF_TMP_1]], i32 0, i32 1
// LLVM-NEXT: store double 0.000000e+00, ptr [[J3]], align 8
// LLVM-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[TMP_0]], ptr {{.*}}[[REF_TMP_1]], i64 16, i1 false)
// LLVM-NEXT: [[C:%.*]] = getelementptr {{.*}}[[STRUCT_C]], ptr [[C2]], i32 0, i32 2
// LLVM-NEXT: store i32 2, ptr [[C]]
// LLVM: ret void
// CIR-LABEL: cir.func {{.*}}@_Z4foo4v()
// CIR: %[[C2_ALLOCA:.*]] = cir.alloca ![[STRUCT_C]], !cir.ptr<![[STRUCT_C]]>, ["c2", init]
// CIR: %[[B_TMP:.*]] = cir.alloca ![[STRUCT_B]], !cir.ptr<![[STRUCT_B]]>, ["ref.tmp0"]
// CIR: %[[A_TMP:.*]] = cir.alloca ![[STRUCT_A]], !cir.ptr<![[STRUCT_A]]>, ["ref.tmp1"]
// CIR: %[[C_BASE:.*]] = cir.base_class_addr %[[C2_ALLOCA]] : !cir.ptr<![[STRUCT_C]]> nonnull [0] -> !cir.ptr<![[STRUCT_B]]>
// CIR: %[[GET_A:.*]] = cir.get_member %[[B_TMP]][0] {name = "a"} : !cir.ptr<![[STRUCT_B]]> -> !cir.ptr<![[STRUCT_A]]>
// CIR: %[[GET_I:.*]] = cir.get_member %[[GET_A]][0] {name = "i"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!s8i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR: cir.store{{.*}} %[[ONE:.*]], %[[GET_I]] : !s8i, !cir.ptr<!s8i>
// CIR: %[[GET_J:.*]] = cir.get_member %[[GET_A]][1] {name = "j"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!cir.double>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[ONE_F:.*]] = cir.cast int_to_float %[[ONE]] : !s32i -> !cir.double
// CIR: cir.store{{.*}} %[[ONE_F]], %[[GET_J]] : !cir.double, !cir.ptr<!cir.double>
// CIR: %[[GET_B:.*]] = cir.get_member %[[B_TMP]][1] {name = "b"} : !cir.ptr<![[STRUCT_B]]> -> !cir.ptr<!s32i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[ONE]], %[[GET_B]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.copy %[[B_TMP]] to %[[C_BASE]] : !cir.ptr<![[STRUCT_B]]>
// CIR: %[[C_BASE_A:.*]] = cir.base_class_addr %[[C2_ALLOCA]] : !cir.ptr<![[STRUCT_C]]> nonnull [24] -> !cir.ptr<![[STRUCT_A]]>
// CIR: %[[GET_I:.*]] = cir.get_member %[[A_TMP]][0] {name = "i"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!s8i>
// CIR: %[[NINETYSEVEN:.*]] = cir.const #cir.int<97> : !s8i
// CIR: cir.store{{.*}} %[[NINETYSEVEN]], %[[GET_I]] : !s8i, !cir.ptr<!s8i>
// CIR: %[[GET_J:.*]] = cir.get_member %[[A_TMP]][1] {name = "j"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!cir.double>
// CIR: %[[ZERO_F:.*]] = cir.const #cir.fp<0
// CIR: cir.store{{.*}} %[[ZERO_F]], %[[GET_J]] : !cir.double, !cir.ptr<!cir.double>
// CIR: cir.copy %[[A_TMP]] to %[[C_BASE_A]] : !cir.ptr<![[STRUCT_A]]>
// CIR: %[[GET_C:.*]] = cir.get_member %[[C2_ALLOCA]][2] {name = "c"} : !cir.ptr<![[STRUCT_C]]> -> !cir.ptr<!s32i>
// CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR: cir.store{{.*}} %[[TWO]], %[[GET_C]] : !s32i, !cir.ptr<!s32i>
void foo4() {
  C c2(B(A(1, 1), 1), A('a'), 2);
}

// LLVM: define dso_local {{.*}}@{{.*foo5.*}}
// LLVM: [[RETVAL:%.*]] = alloca [[UNION_U]]
// LLVM-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[RETVAL]], ptr {{.*}}[[U1]], i64 16, i1 false)
// CIR-LABEL: cir.func no_inline dso_local @_Z4foo5v()
// CIR:  %[[RET:.*]] = cir.alloca ![[UNION_U]], !cir.ptr<![[UNION_U]]>, ["__retval"] {alignment = 8 : i64}
// CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZL2u1 : !cir.ptr<!{{.*}}>
// CIR:  %[[GLOB_TO_U:.*]] = cir.cast bitcast %[[GET_GLOB]] : !cir.ptr<!{{.*}}> -> !cir.ptr<![[UNION_U]]>
// CIR:  cir.copy %[[GLOB_TO_U]] to %[[RET]] : !cir.ptr<![[UNION_U]]>
U foo5() {
  return u1;
}


// LLVM: define dso_local {{.*}}@{{.*foo6.*}}
// LLVM-DAG:   [[RETVAL:%.*]] = alloca [[UNION_U]]
// LLVM-DAG:   [[A:%.*]] = alloca [[STRUCT_A]]
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[RETVAL]], ptr {{.*}}[[A]], i64 16, i1 false)
// CIR-LABEL: cir.func no_inline dso_local @_Z4foo61A(
// CIR: %[[A_ALLOCA:.*]] = cir.alloca ![[STRUCT_A]], !cir.ptr<![[STRUCT_A]]>, ["a", init] {alignment = 8 : i64}
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca ![[UNION_U]], !cir.ptr<![[UNION_U]]>, ["__retval"] {alignment = 8 : i64}
// CIR: %[[GET_A:.*]] = cir.get_member %[[RET_ALLOCA:.*]][1] {name = "a"} : !cir.ptr<![[UNION_U]]> -> !cir.ptr<![[STRUCT_A]]>
// CIR: cir.copy %[[A_ALLOCA]] to %[[GET_A:.*]] : !cir.ptr<![[STRUCT_A]]>
U foo6(A a) {
  return U(a);
}

// LLVM: define dso_local void @{{.*foo7.*}}
// LLVM: [[D:%.*]] = alloca [[STRUCT_D]]
// LLVM-NEXT: [[A:%.*]] = getelementptr {{.*}}[[STRUCT_D]], ptr [[D]], i32 0, i32 0
// LLVM-NEXT: [[I:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[A]], i32 0, i32 0
// LLVM-NEXT: store i8 1, ptr [[I]], align 8
// LLVM-NEXT: [[J:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[A]], i32 0, i32 1
// LLVM-NEXT: store double 1.000000e+00, ptr [[J]], align 8
// LLVM-NEXT: [[B:%.*]] = getelementptr {{.*}}[[STRUCT_D]], ptr [[D]], i32 0, i32 1
// LLVM-NEXT: [[I1:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[B]], i32 0, i32 0
// LLVM-NEXT: store i8 11, ptr [[I1]], align 8
// LLVM-NEXT: [[J2:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[B]], i32 0, i32 1
// LLVM-NEXT: store double 1.100000e+01, ptr [[J2]], align 8
// LLVM-NEXT: [[C:%.*]] = getelementptr {{.*}}[[STRUCT_D]], ptr [[D]], i32 0, i32 3
// LLVM-NEXT: [[I3:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[C]], i32 0, i32 0
// LLVM-NEXT: store i8 111, ptr [[I3]], align 8
// LLVM-NEXT: [[J4:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[C]], i32 0, i32 1
// LLVM-NEXT: store double 1.110000e+02, ptr [[J4]], align 8
// LLVM-NEXT: ret void
// CIR-LABEL; cir.func no_inline dso_local @_Z4foo7v()
// CIR: %[[D_ALLOCA:.*]] = cir.alloca ![[STRUCT_D]], !cir.ptr<![[STRUCT_D]]>, ["d", init] {alignment = 8 : i64}
// CIR: %[[GET_A:.*]] = cir.get_member %[[D_ALLOCA]][0] {name = "a"} : !cir.ptr<![[STRUCT_D]]> -> !cir.ptr<![[STRUCT_A]]>
// CIR: %[[GET_I:.*]] = cir.get_member %[[GET_A]][0] {name = "i"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!s8i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR: cir.store align(8) %[[ONE]], %[[GET_I]] : !s8i, !cir.ptr<!s8i>
// CIR: %[[GET_J:.*]] = cir.get_member %[[GET_A]][1] {name = "j"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!cir.double>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[ONE_F:.*]] = cir.cast int_to_float %[[ONE]] : !s32i -> !cir.double
// CIR: cir.store align(8) %[[ONE_F]], %[[GET_J]] : !cir.double, !cir.ptr<!cir.double>
// CIR: %[[GET_B:.*]] = cir.get_member %[[D_ALLOCA]][1] {name = "b"} : !cir.ptr<![[STRUCT_D]]> -> !cir.ptr<![[STRUCT_A]]>
// CIR: %[[GET_I:.*]] = cir.get_member %[[GET_B]][0] {name = "i"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!s8i>
// CIR: %[[ELEVEN:.*]] = cir.const #cir.int<11> : !s8i
// CIR: cir.store align(8) %[[ELEVEN]], %[[GET_I]] : !s8i, !cir.ptr<!s8i>
// CIR: %[[GET_J:.*]] = cir.get_member %[[GET_B]][1] {name = "j"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!cir.double>
// CIR: %[[ELEVEN:.*]] = cir.const #cir.int<11> : !s32i
// CIR: %[[ELEVEN_F:.*]] = cir.cast int_to_float %[[ELEVEN]] : !s32i -> !cir.double
// CIR: cir.store align(8) %[[ELEVEN_F]], %[[GET_J]] : !cir.double, !cir.ptr<!cir.double>
// CIR: %[[GET_C:.*]] = cir.get_member %[[D_ALLOCA]][3] {name = "c"} : !cir.ptr<![[STRUCT_D]]> -> !cir.ptr<![[STRUCT_A]]>
// CIR: %[[GET_I:.*]] = cir.get_member %[[GET_C]][0] {name = "i"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!s8i>
// CIR: %[[ONE_ELEVEN:.*]] = cir.const #cir.int<111> : !s8i
// CIR: cir.store align(8) %[[ONE_ELEVEN]], %[[GET_I]] : !s8i, !cir.ptr<!s8i>
// CIR: %[[GET_J:.*]] = cir.get_member %[[GET_C]][1] {name = "j"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!cir.double>
// CIR: %[[ONE_ELEVEN:.*]] = cir.const #cir.int<111> : !s32i
// CIR: %[[ONE_ELEVEN_F:.*]] = cir.cast int_to_float %[[ONE_ELEVEN]] : !s32i -> !cir.double
// CIR: cir.store align(8) %[[ONE_ELEVEN_F]], %[[GET_J]] : !cir.double, !cir.ptr<!cir.double>
void foo7() {
  D d(A(1, 1), A(11, 11), A(111, 111));
}

// LLVM: dso_local {{.*}}@{{.*foo8.*}}(
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr {{.*}}[[D1]], i64 56, i1 false)
// CIR-LABEL: cir.func no_inline dso_local @_Z4foo8v() 
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca ![[STRUCT_D]], !cir.ptr<![[STRUCT_D]]>, ["__retval"] {alignment = 8 : i64}
// CIR: %[[GET_GLOB:.*]] = cir.get_global @_ZL2d1 :
// CIR: %[[GLOB_CAST:.*]] = cir.cast bitcast %[[GET_GLOB]] : !cir.ptr<!{{.*}}> -> !cir.ptr<![[STRUCT_D]]>
// CIR: cir.copy %[[GLOB_CAST]] to %[[RET_ALLOCA]] : !cir.ptr<![[STRUCT_D]]>
D foo8() {
  return d1;
}

// LLVM: define dso_local void @{{.*foo9.*}}
// LLVM: [[D:%.*]] = alloca [[STRUCT_D]]
// LLVM-NEXT: [[A:%.*]] = getelementptr {{.*}}[[STRUCT_D]], ptr [[D]], i32 0, i32 0
// LLVM-NEXT: [[I]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[A]], i32 0, i32 0
// LLVM-NEXT: store i8 1, ptr [[I]], align 8
// LLVM-NEXT: [[J:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[A]], i32 0, i32 1
// LLVM-NEXT: store double 1.000000e+00, ptr [[J]], align 8
// LLVM-NEXT: [[B:%.*]] = getelementptr {{.*}}[[STRUCT_D]], ptr [[D]], i32 0, i32 1
// LLVM-NEXT: [[I1:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[B]], i32 0, i32 0
// LLVM-NEXT: store i8 2, ptr [[I1]], align 8
// LLVM-NEXT: [[J2:%.*]] = getelementptr {{.*}}[[STRUCT_A]], ptr [[B]], i32 0, i32 1
// LLVM-NEXT: store double 2.000000e+00, ptr [[J2]], align 8
// LLVM-NEXT: [[C:%.*]] = getelementptr {{.*}}[[STRUCT_D]], ptr [[D]], i32 0, i32 3
// CIR-LABEL: cir.func no_inline dso_local @_Z4foo9v()
// CIR: %[[D_ALLOCA:.*]] = cir.alloca ![[STRUCT_D]], !cir.ptr<![[STRUCT_D]]>, ["d", init] {alignment = 8 : i64}
// CIR: %[[GET_A:.*]] = cir.get_member %[[D_ALLOCA]][0] {name = "a"} : !cir.ptr<![[STRUCT_D]]> -> !cir.ptr<![[STRUCT_A]]>
// CIR: %[[GET_I:.*]] = cir.get_member %[[GET_A]][0] {name = "i"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!s8i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR: cir.store align(8) %[[ONE]], %[[GET_I]] : !s8i, !cir.ptr<!s8i>
// CIR: %[[GET_J:.*]] = cir.get_member %[[GET_A]][1] {name = "j"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!cir.double>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[ONE_F:.*]] = cir.cast int_to_float %[[ONE]] : !s32i -> !cir.double
// CIR: cir.store align(8) %[[ONE_F]], %[[GET_J]] : !cir.double, !cir.ptr<!cir.double>
// CIR: %[[GET_B:.*]] = cir.get_member %[[D_ALLOCA]][1] {name = "b"} : !cir.ptr<![[STRUCT_D]]> -> !cir.ptr<![[STRUCT_A]]>
// CIR: %[[GET_I:.*]] = cir.get_member %[[GET_B]][0] {name = "i"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!s8i>
// CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s8i
// CIR: cir.store align(8) %[[TWO]], %[[GET_I]] : !s8i, !cir.ptr<!s8i>
// CIR: %[[GET_J:.*]] = cir.get_member %[[GET_B]][1] {name = "j"} : !cir.ptr<![[STRUCT_A]]> -> !cir.ptr<!cir.double>
// CIR: %[[FP_2:.*]] = cir.const #cir.fp<2
// CIR: cir.store align(8) %[[FP_2]], %[[GET_J]] : !cir.double, !cir.ptr<!cir.double>
// CIR: %[[GET_C:.*]] = cir.get_member %[[D_ALLOCA]][3] {name = "c"} : !cir.ptr<![[STRUCT_D]]> -> !cir.ptr<![[STRUCT_A]]>
// CIR: %[[ZERO:.*]] = cir.const #cir.zero : ![[STRUCT_A]]
// CIR: cir.store align(8) %[[ZERO]], %[[GET_C]] : ![[STRUCT_A]], !cir.ptr<![[STRUCT_A]]>
void foo9() {
  D d(A(1, 1));
}

// LLVM: define dso_local noundef ptr @{{.*foo10.*}}()
// FIXME: CIR lowering has an extra load here.
// CIR-LABEL: cir.func no_inline dso_local @_Z5foo10v()
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CIR: %[[GET_GLOB:.*]] = cir.get_global @_ZL4arr1 : !cir.ptr<!cir.array<!s32i x 3>>
// CIR: %[[GLOB_DECAY:.*]] = cir.cast array_to_ptrdecay %[[GET_GLOB]] : !cir.ptr<!cir.array<!s32i x 3>> -> !cir.ptr<!s32i>
// CIR: cir.store %[[GLOB_DECAY]], %[[RET_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
const int* foo10() {
  return arr1;
}

// LLVM: define dso_local void @{{.*foo11.*}}
// LLVM: [[A_ADDR:%.*]] = alloca i32
// LLVM-NEXT: [[B_ADDR:%.*]] = alloca i32
// LLVM-NEXT: [[ARR_2:%.*]] = alloca [4 x i32]
// LLVM: [[TMP_0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// FIXME: CIR does an additional GEP here to decay the type.
// LLVM: store i32 [[TMP_0]], ptr {{.*}}, align 4
// LLVM-NEXT: [[ARRINIT_ELEM:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 1
// LLVM-NEXT: [[TMP_1:%.*]] = load i32, ptr [[B_ADDR]], align 4
// FXIME: CIR calculates this via offset from the above ARRINIT_ELEM, not the 0th element.
// LLVM-NEXT: store i32 [[TMP_1]], ptr [[ARRINIT_ELEM]], align 4
// LLVM: [[ARRINIT_START:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 {{.*}}
// LLVM: [[ARRINIT_END:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 4
// LLVM-NEXT: br label [[ARRINIT_BODY:%.*]]
// LLVM: [[ARRINIT_DONE:%.*]] = icmp {{.*}} ptr
// LLVM-NEXT: br i1 [[ARRINIT_DONE]], label 
// LLVM: ret void
// CIR-LABEL: cir.func no_inline dso_local @_Z5foo11ii
// CIR: %[[A_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CIR: %[[B_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init] {alignment = 4 : i64}
// CIR: %[[ARR2_ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 4>, !cir.ptr<!cir.array<!s32i x 4>>, ["arr2", init] {alignment = 16 : i64}
// CIR: %[[ARR_ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init] {alignment = 8 : i64}
// CIR: %[[ARR2_DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARR2_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>
// CIR: %[[A_LOAD:.*]] = cir.load align(4) %[[A_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store align(4) %[[A_LOAD]], %[[ARR2_DECAY]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ARR2_DECAY]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[B_LOAD:.*]] = cir.load align(4) %[[B_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store align(4) %[[B_LOAD]], %[[ELT1]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: cir.store align(8) %[[ELT2]], %[[ARR_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !s64i
// CIR: %[[END_ITR:.*]] = cir.ptr_stride %[[ARR2_DECAY]], %[[FOUR]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: cir.do {
// CIR:   %[[ARR_ITR_LOAD:.*]] = cir.load align(8) %[[ARR_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store align(4) %[[ZERO]], %[[ARR_ITR_LOAD]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:   %[[NEXT_ITR:.*]] = cir.ptr_stride %[[ARR_ITR_LOAD]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR:   cir.store align(8) %[[NEXT_ITR]], %[[ARR_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   cir.yield
// CIR: } while {
// CIR:   %[[ARR_ITR_LOAD:.*]] = cir.load align(8) %[[ARR_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:   %[[DONE:.*]] = cir.cmp ne %[[ARR_ITR_LOAD]], %[[END_ITR]] : !cir.ptr<!s32i>
// CIR:   cir.condition(%[[DONE]])
// CIR: }
void foo11(int a, int b) {
  int arr2[4](a, b);
}

// LLVM: define dso_local void @{{.*foo12.*}}
// LLVM: [[A_ADDR:%.*]] = alloca i32
// LLVM-NEXT: [[B_ADDR:%.*]] = alloca i32
// LLVM-NEXT: [[ARR_3:%.*]] = alloca [2 x i32]
// LLVM-NEXT: store i32 [[A:%.*]], ptr [[A_ADDR]], align 4
// LLVM-NEXT: store i32 [[B:%.*]], ptr [[B_ADDR]], align 4
// LLVM: [[TMP_0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// LLVM-NEXT: store i32 [[TMP_0]], ptr {{.*}}, align 4
// LLVM-NEXT: [[ARRINIT_ELEMENT:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 1
// LLVM-NEXT: [[TMP_1:%.*]] = load i32, ptr [[B_ADDR]], align 4
// LLVM-NEXT: store i32 [[TMP_1]], ptr [[ARRINIT_ELEMENT]], align 4
// LLVM-NEXT: ret void
// CIR-LABEL: cir.func no_inline dso_local @_Z5foo12ii
// CIR: %[[A_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CIR: %[[B_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init] {alignment = 4 : i64}
// CIR: %[[ARR3_ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["arr3", init] {alignment = 4 : i64}
// CIR: %[[ARR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARR3_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
// CIR: %[[A_LOAD:.*]] = cir.load align(4) %[[A_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store align(4) %[[A_LOAD]], %[[ARR_DECAY]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELT_1:.*]] = cir.ptr_stride %[[ARR_DECAY]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[B_LOAD:.*]] = cir.load align(4) %[[B_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store align(4) %[[B_LOAD]], %[[ELT_1]] : !s32i, !cir.ptr<!s32i>
void foo12(int a, int b) {
  int arr3[](a, b);
}

// LLVM: define {{.*}}@{{.*foo13.*}}
// LLVM: [[RETVAL:%.*]] = alloca [[STRUCT_A]]
// LLVM-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[RETVAL]], ptr {{.*}}[[A2]], i64 16, i1 false)
// LLVM-NEXT: [[TMP_0:%.*]] = load {{.*}}, ptr [[RETVAL]], align 8
// LLVM-NEXT: ret {{.*}}[[TMP_0]]
// CIR-LABEL: cir.func no_inline dso_local @_Z5foo13v()
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca ![[STRUCT_A]], !cir.ptr<![[STRUCT_A]]>, ["__retval"] {alignment = 8 : i64}
// CIR; %[[GET_GLOB:.*]] = cir.get_global @_ZL2a2 : !cir.ptr<![[STRUCT_A]]>
// CIR; cir.copy %[[GET_GLOB]] to %[[RET_ALLOCA]] : !cir.ptr<![[STRUCT_A]]>
A foo13() {
  return a2;
}

// LLVM: define dso_local noundef ptr @{{.*foo14.*}}()
// LLVM: ret ptr 
// CIR-LABEL: cir.func no_inline dso_local @_Z5foo14v()
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CIR: %[[GET_GLOB:.*]] = cir.get_global @_ZL4arr4 : !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[GLOB_DECAY]] = cir.cast array_to_ptrdecay %[[GET_GLOB]] : !cir.ptr<!cir.array<!s32i x 1>> -> !cir.ptr<!s32i>
// CIR: cir.store %[[GLOB_DECAY]], %[[RET_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
const int* foo14() {
  return arr4;
}

// LLVM: define dso_local noundef ptr @{{.*foo15.*}}()
// LLVM: ret ptr 
// CIR-LABEL: cir.func {{.*}}@_Z5foo15v()
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CIR: %[[GET_GLOB:.*]] = cir.get_global @_ZL4arr5 : !cir.ptr<!cir.array<!s32i x 2>>
// CIR: %[[GLOB_DECAY:.*]] = cir.cast array_to_ptrdecay %[[GET_GLOB]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
// CIR: cir.store %[[GLOB_DECAY]], %[[RET_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
const int* foo15() {
  return arr5;
}

// LLVM: define dso_local void @{{.*foo16.*}}
// LLVM-DAG: [[ARR_6:%.*]] = alloca ptr
// LLVM-DAG: [[REF_TMP:%.*]] = alloca [1 x i32]
// LLVM: store i32 3, ptr {{.*}}, align 4
// LLVM-NEXT: store ptr [[REF_TMP]], ptr [[ARR_6]], align 8
// LLVM-NEXT: ret void
// CIR-LABEL: cir.func no_inline dso_local @_Z5foo16v()
// CIR: %[[TEMP_ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>, ["ref.tmp0"] {alignment = 4 : i64}
// CIR: %[[ARR6_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!s32i x 0>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>, ["arr6", init, const] {alignment = 8 : i64}
// CIR: %[[TEMP_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TEMP_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 1>> -> !cir.ptr<!s32i>
// CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR: cir.store align(4) %[[THREE]], %[[TEMP_DECAY]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TEMP_CAST:.*]] = cir.cast bitcast %[[TEMP_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 1>> -> !cir.ptr<!cir.array<!s32i x 0>>
// CIR: cir.store align(8) %[[TEMP_CAST]], %[[ARR6_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 0>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>
void foo16() {
  int (&&arr6)[] = static_cast<int[]>(3);
}

// LLVM: define dso_local void @{{.*foo17.*}}
// LLVM-DAG: [[ARR_7:%.*]] = alloca ptr
// LLVM-DAG: [[REF_TMP:%.*]] = alloca [2 x i32]
// LLVM: store i32 4, ptr {{.*}}, align 4
// LLVM-NEXT: [[ARRINIT_START:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 1
// LLVM: [[ARRINIT_END:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 2
// LLVM: br label
// LLVM: [[ARRINIT_DONE:%.*]] = icmp {{.*}} ptr 
// LLVM-NEXT: br i1 [[ARRINIT_DONE]], label 
// LLVM: store ptr [[REF_TMP]], ptr [[ARR_7]], align 8
// LLVM: ret void
// CIR-LABEL: cir.func no_inline dso_local @_Z5foo17v()
// CIR: %[[ARR_TEMP:.*]] = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["ref.tmp0"] {alignment = 4 : i64}
// CIR: %[[ARR7_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!s32i x 2>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 2>>>, ["arr7", init, const] {alignment = 8 : i64}
// CIR: %[[ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init] {alignment = 8 : i64}
// CIR: %[[TEMP_LOAD:.*]] = cir.cast array_to_ptrdecay %[[ARR_TEMP]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
// CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CIR: cir.store align(4) %[[FOUR]], %[[TEMP_LOAD]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELT1:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: cir.store align(8) %[[ELT1]], %[[ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
// CIR: %[[END_ITR:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[TWO]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: cir.do {
// CIR:   %[[ITR_LOAD:.*]] = cir.load align(8) %[[ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store align(4) %[[ZERO]], %[[ITR_LOAD]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:   %[[NEXT_ITR:.*]] = cir.ptr_stride %[[ITR_LOAD]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR:   cir.store align(8) %[[NEXT_ITR]], %[[ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   cir.yield
// CIR: } while {
// CIR:   %[[ITR_LOAD:.*]] = cir.load align(8) %[[ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:   %[[COND:.*]] = cir.cmp ne %[[ITR_LOAD]], %[[END_ITR]] : !cir.ptr<!s32i>
// CIR:   cir.condition(%[[COND]])
// CIR: }
// CIR: cir.store align(8) %[[ARR_TEMP]], %[[ARR7_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 2>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 2>>>
void foo17() {
  int (&&arr7)[2] = static_cast<int[2]>(4);
}

// LLVM: define dso_local void @{{.*foo18.*}}
// LLVM: [[E:%.*]] = alloca [[STRUCT_E]]
// LLVM-NEXT: [[A:%.*]] = getelementptr {{.*}}[[STRUCT_E]], ptr [[E]], i32 0, i32 0
// LLVM-NEXT: store i32 1, ptr [[A]], align 8
// LLVM-NEXT: [[FN:%.*]] = getelementptr {{.*}}[[STRUCT_E]], ptr [[E]], i32 0, i32 1
// LLVM-NEXT: store ptr [[STR]], ptr [[FN]], align 8
// LLVM: ret void
// CIR: cir.func {{.*}}@_Z5foo18v()
// CIR: %[[E_ALLOCA:.*]] = cir.alloca ![[STRUCT_E]], !cir.ptr<![[STRUCT_E]]>, ["e", init]
// CIR: %[[GET_A:.*]] = cir.get_member %[[E_ALLOCA]][0] {name = "a"} : !cir.ptr<![[STRUCT_E]]> -> !cir.ptr<!s32i>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[ONE]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[GET_FN:.*]] = cir.get_member %[[E_ALLOCA]][1] {name = "fn"} : !cir.ptr<![[STRUCT_E]]> -> !cir.ptr<!cir.ptr<!s8i>>
// CIR: %[[GET_STR:.*]] = cir.const #cir.global_view<@
// CIR: cir.store{{.*}} %[[GET_STR]], %[[GET_FN]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
void foo18() {
  E e(1);
}

// LLVM: define dso_local void @{{.*foo19.*}}
// LLVM: [[G:%.*]] = alloca [[STRUCT_G]]
// LLVM-NEXT: [[A:%.*]] = getelementptr {{.*}}[[STRUCT_G]], ptr [[G]], i32 0, i32 0
// LLVM-NEXT: store i32 2, ptr [[A]], align 4
// LLVM-NEXT: [[F:%.*]] = getelementptr {{.*}}i8, ptr [[G]], i64 4
// LLVM-NEXT: call void @{{.*F.*}}(ptr noundef nonnull align 1 dereferenceable(1) [[F]], i32 noundef 1)
// LLVM: ret void
// CIR: cir.func no_inline dso_local @_Z5foo19v() attributes {{{.*}}nothrow} {
// CIR: %[[G_ALLOCA:.*]] = cir.alloca ![[STRUCT_G]], !cir.ptr<![[STRUCT_G]]>, ["g", init]
// CIR: %[[GET_A:.*]] = cir.get_member %[[G_ALLOCA]][0] {name = "a"} : !cir.ptr<![[STRUCT_G]]> -> !cir.ptr<!s32i>
// CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR: cir.store{{.*}} %[[TWO]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[G_TO_CHAR:.*]] = cir.cast bitcast %[[G_ALLOCA]] : !cir.ptr<![[STRUCT_G]]> -> !cir.ptr<!u8i>
// CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
// CIR: %[[F_OFFSET:.*]] = cir.ptr_stride %[[G_TO_CHAR]], %[[FOUR]] : (!cir.ptr<!u8i>, !u64i) -> !cir.ptr<!u8i>
// CIR: %[[F:.*]] = cir.cast bitcast %[[F_OFFSET]] : !cir.ptr<!u8i> -> !cir.ptr<![[STRUCT_F]]>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.call @_ZN1FC1Ei(%[[F]], %[[ONE]])
void foo19() {
  G g(2);
}

namespace gh61145 {
  // a.k.a. void make1<0>()
  // LLVM: define {{.*}} void @_ZN7gh611455make1ILi0EEEvv
  // LLVM-DAG: [[V:%.*]] = alloca [[STRUCT_VEC]]
  // LLVM-DAG: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_S1]]
  // a.k.a. Vec::Vec()
  // LLVM-NEXT: call void @_ZN7gh611453VecC1Ev(ptr {{.*}}[[V]])
  // a.k.a. Vec::Vec(Vec&&)
  // LLVM: call void @_ZN7gh611453VecC1EOS0_(ptr {{.*}}[[AGG_TMP_ENSURED]], ptr {{.*}}[[V]])
  // a.k.a. S1::~S1()
  // LLVM: call void @_ZN7gh611452S1D1Ev(ptr {{.*}}[[AGG_TMP_ENSURED]])
  // a.k.a.Vec::~Vec()
  // LLVM: call void @_ZN7gh611453VecD1Ev(ptr {{.*}}[[V]])
  // LLVM: ret void
  // CIR-LABEL: cir.func {{.*}}@_ZN7gh611455make1ILi0EEEvv()
  // CIR: %[[V_ALLOCA:.*]] = cir.alloca ![[STRUCT_VEC]], !cir.ptr<![[STRUCT_VEC]]>, ["v", init] {alignment = 1 : i64}
  // CIR: %[[TMP_ALLOCA:.*]] = cir.alloca ![[STRUCT_S1]], !cir.ptr<![[STRUCT_S1]]>, ["agg.tmp.ensured"]
  // CIR: cir.call @_ZN7gh611453VecC1Ev(%[[V_ALLOCA]])
  // CIR: %[[S1_TO_VEC:.*]] = cir.cast bitcast %[[TMP_ALLOCA]] : !cir.ptr<![[STRUCT_S1]]> -> !cir.ptr<![[STRUCT_VEC]]>
  // CIR: cir.call @_ZN7gh611453VecC1EOS0_(%[[S1_TO_VEC]], %[[V_ALLOCA]]) 
  // CIR: cir.call @_ZN7gh611452S1D1Ev(
  // CIR: cir.call @_ZN7gh611453VecD1Ev(
  template <int I>
  void make1() {
    Vec v;
    S1((Vec&&) v);
  }

  // a.k.a. void make2<0>()
  // LLVM: define {{.*}} void @_ZN7gh611455make2ILi0EEEvv
  // LLVM-DAG: [[V:%.*]] = alloca [[STRUCT_VEC]]
  // LLVM-DAG: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_S2]]
  // a.k.a. Vec::Vec()
  // LLVM: call void @_ZN7gh611453VecC1Ev(ptr {{.*}}[[V]])
  // a.k.a. Vec::Vec(Vec&&)
  // LLVM: call void @_ZN7gh611453VecC1EOS0_(ptr {{.*}}[[AGG_TMP_ENSURED]], ptr {{.*}}[[V]])
  // LLVM-NEXT: [[C:%.*]] = getelementptr {{.*}}[[STRUCT_S2]], ptr [[AGG_TMP_ENSURED]], i32 0, i32
  // LLVM-NEXT: store i8 0, ptr [[C]], align 1
  // a.k.a. S2::~S2()
  // LLVM: call void @_ZN7gh611452S2D1Ev(ptr {{.*}}[[AGG_TMP_ENSURED]])
  // a.k.a. Vec::~Vec()
  // LLVM: call void @_ZN7gh611453VecD1Ev(ptr {{.*}}[[V]])
  // LLVM: ret void
  // CIR-LABEL: cir.func {{.*}}@_ZN7gh611455make2ILi0EEEvv()
  // CIR: %[[V_ALLOCA:.*]] = cir.alloca ![[STRUCT_VEC]], !cir.ptr<![[STRUCT_VEC]]>, ["v", init] {alignment = 1 : i64}
  // CIR: %[[TMP_ALLOCA:.*]] = cir.alloca ![[STRUCT_S2]], !cir.ptr<![[STRUCT_S2]]>, ["agg.tmp.ensured"]
  // CIR: cir.call @_ZN7gh611453VecC1Ev(%[[V_ALLOCA]])
  // CIR: %[[S2_TO_VEC:.*]] = cir.cast bitcast %[[TMP_ALLOCA]] : !cir.ptr<![[STRUCT_S2]]> -> !cir.ptr<![[STRUCT_VEC]]>
  // CIR: cir.call @_ZN7gh611453VecC1EOS0_(%[[S2_TO_VEC]], %[[V_ALLOCA]])
  // CIR: %[[GET_C:.*]] = cir.get_member %[[TMP_ALLOCA]][1] {name = "c"} : !cir.ptr<![[STRUCT_S2]]> -> !cir.ptr<!s8i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_C]] : !s8i, !cir.ptr<!s8i>
  template <int I>
  void make2() {
    Vec v;
    S2((Vec&&) v, 0);
  }

  void foo() {
    make1<0>();
    make2<0>();
  }
}

namespace gh62266 {
  // LLVM: define {{.*}} void {{.*foo20.*}}
  // LLVM: [[H:%.*]] = alloca [[STRUCT_H]]
  // LLVM-NEXT: [[I:%.*]] = getelementptr {{.*}}[[STRUCT_H]], ptr [[H]], i32 0, i32 0
  // LLVM-NEXT: store i32 1, ptr [[I]], align 4
  // LLVM-NEXT: [[J:%.*]] = getelementptr {{.*}}[[STRUCT_H]], ptr [[H]], i32 0, i32 1
  // LLVM-NEXT: store i32 2, ptr [[J]], align 4
  // LLVM-NEXT: ret void
  // CIR-LABEL: cir.func {{.*}}@_ZN7gh622665foo20Ev()
  // CIR: %[[TMP_ALLOCA:.*]] = cir.alloca ![[STRUCT_H]], !cir.ptr<![[STRUCT_H]]>, ["h", init]
  // CIR: %[[GET_I:.*]] = cir.get_member %[[TMP_ALLOCA]][0] {name = "i"} : !cir.ptr<![[STRUCT_H]]> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store{{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_J:.*]] = cir.get_member %[[TMP_ALLOCA]][1] {name = "j"} : !cir.ptr<![[STRUCT_H]]> -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store{{.*}} %[[TWO]], %[[GET_J]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.return
  void foo20() {
    H<2> h(1);
  }
}

namespace gh61567 {
  int foo20();

  // LLVM: define {{.*}} void @{{.*foo21.*}} {
  // LLVM: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_I]]
  // LLVM-NEXT: [[REF_TMP:%.*]] = alloca i32
  // LLVM: [[A:%.*]] = getelementptr {{.*}}[[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // LLVM-NEXT: store i32 0, ptr [[A]], align 8
  // LLVM-NEXT: [[R:%.*]] = getelementptr {{.*}}[[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 1
  // LLVM-NEXT: store i32 1, ptr [[REF_TMP]], align 4
  // LLVM-NEXT: store ptr [[REF_TMP]], ptr [[R]], align 8
  // LLVM: ret void
  // CIR-LABEL: cir.func {{.*}}@_ZN7gh615675foo21Ev()
  // CIR: %[[TMP_ALLOCA:.*]] = cir.alloca ![[STRUCT_I]], !cir.ptr<![[STRUCT_I]]>, ["agg.tmp.ensured"]
  // CIR: %[[INT_TMP:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0"]
  // CIR: %[[GET_A:.*]] = cir.get_member %[[TMP_ALLOCA]][0] {name = "a"} : !cir.ptr<![[STRUCT_I]]> -> !cir.ptr<!s32i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_R:.*]] = cir.get_member %[[TMP_ALLOCA]][1] {name = "r"} : !cir.ptr<![[STRUCT_I]]> -> !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store{{.*}} %[[ONE]], %[[INT_TMP]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[INT_TMP]], %[[GET_R]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  void foo21() {
    I(0, 1);
  }

  // LLVM: define {{.*}} void @{{.*foo22.*}} {
  // LLVM: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_I]]
  // LLVM-NEXT: [[REF_TMP:%.*]] = alloca i32
  // LLVM: [[A:%.*]] = getelementptr {{.*}}[[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // LLVM-NEXT: store i32 0, ptr [[A]], align 8
  // LLVM-NEXT: [[R:%.*]] = getelementptr {{.*}}[[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 1
  // LLVM-NEXT: [[CALL:%.*]] = call noundef i32 @{{.*foo20.*}}
  // LLVM-NEXT: store i32 [[CALL]], ptr [[REF_TMP]], align 4
  // LLVM-NEXT: store ptr [[REF_TMP]], ptr [[R]], align 8
  // LLVM: ret void
  // CIR-LABEL: cir.func {{.*}}@_ZN7gh615675foo22Ev()
  // CIR: %[[TMP_ALLOCA:.*]] = cir.alloca ![[STRUCT_I]], !cir.ptr<![[STRUCT_I]]>, ["agg.tmp.ensured"]
  // CIR: %[[INT_TMP:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0"]
  // CIR: %[[GET_A:.*]] = cir.get_member %[[TMP_ALLOCA]][0] {name = "a"} : !cir.ptr<![[STRUCT_I]]> -> !cir.ptr<!s32i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_R:.*]] = cir.get_member %[[TMP_ALLOCA]][1] {name = "r"} : !cir.ptr<![[STRUCT_I]]> -> !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[GET_CALL:.*]] = cir.call @_ZN7gh615675foo20Ev() : () -> (!s32i {llvm.noundef})
  // CIR: cir.store{{.*}} %[[GET_CALL]], %[[INT_TMP]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[INT_TMP]], %[[GET_R]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  void foo22() {
    I(0, foo20());
  }

  // LLVM: define {{.*}} void @{{.*foo23.*}}
  // LLVM: [[I_ADDR:%.*]] = alloca i32
  // LLVM-NEXT: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_I]]
  // LLVM: [[A:%.*]] = getelementptr {{.*}}[[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // LLVM-NEXT: store i32 0, ptr [[A]], align 8
  // LLVM-NEXT: [[R:%.*]] = getelementptr {{.*}}[[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 1
  // LLVM-NEXT: store ptr [[I_ADDR]], ptr [[R]], align 8
  // LLVM-NEXT: ret void
  // CIR-LABEL: cir.func no_inline dso_local @_ZN7gh615675foo23Ei(%arg0: !s32i {llvm.noundef}
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
  // CIR: %[[TMP_ALLOCA:.*]] = cir.alloca ![[STRUCT_I]], !cir.ptr<![[STRUCT_I]]>, ["agg.tmp.ensured"]
  // CIR: %[[GET_A:.*]] = cir.get_member %[[TMP_ALLOCA]][0] {name = "a"} : !cir.ptr<![[STRUCT_I]]> -> !cir.ptr<!s32i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_R:.*]] = cir.get_member %[[TMP_ALLOCA]][1] {name = "r"} : !cir.ptr<![[STRUCT_I]]> -> !cir.ptr<!cir.ptr<!s32i>>
  // CIR: cir.store{{.*}} %[[I_ALLOCA]], %[[GET_R]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  void foo23(int i) {
    I(0, static_cast<int&&>(i));
  }

  // LLVM: define {{.*}} void @{{.*foo24.*}} {
  // LLVM: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_I]]
  // LLVM-NEXT: [[REF_TMP:%.*]] = alloca i32
  // LLVM-NEXT: [[A:%.*]] = getelementptr {{.*}}[[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // LLVM-NEXT: store i32 0, ptr [[A]], align 8
  // LLVM-NEXT: [[R:%.*]] = getelementptr {{.*}}[[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 1
  // LLVM-NEXT: store i32 2, ptr [[REF_TMP]], align 4
  // LLVM-NEXT: store ptr [[REF_TMP]], ptr [[R]], align 8
  // LLVM-NEXT: ret void
  // CIR-LABEL: cir.func {{.*}}@_ZN7gh615675foo24Ev()
  // CIR: %[[TMP_ALLOCA:.*]] = cir.alloca ![[STRUCT_I]], !cir.ptr<![[STRUCT_I]]>, ["agg.tmp.ensured"]
  // CIR: %[[INT_TMP:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0"]
  // CIR: %[[GET_A:.*]] = cir.get_member %[[TMP_ALLOCA:.*]][0] {name = "a"} : !cir.ptr<![[STRUCT_I]]> -> !cir.ptr<!s32i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_R:.*]] = cir.get_member %[[TMP_ALLOCA:.*]][1] {name = "r"} : !cir.ptr<![[STRUCT_I]]> -> !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store{{.*}} %[[TWO]], %[[INT_TMP]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[INT_TMP]], %[[GET_R]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  void foo24() {
    I(0);
  }
}

namespace gh68198 {
  // LLVM: define {{.*}} void @{{.*foo25.*}} {
  // LLVM: [[ARR_8:%.*]] = alloca ptr
  // LLVM-NEXT: [[CALL_PTR:%.*]] = call {{.*}}nonnull ptr @_Znam(i64 noundef 8)
  // LLVM-NEXT: store i32 1, ptr [[CALL_PTR]], align 4
  // LLVM-NEXT: [[ARRAY_EXP_NEXT:%.*]] = getelementptr {{.*}}i32, ptr [[CALL_PTR]], i64 1
  // LLVM-NEXT: store i32 2, ptr [[ARRAY_EXP_NEXT]], align 4
  // LLVM-NEXT: [[ARRAY_EXP_NEXT1:%.*]] = getelementptr {{.*}}i32, ptr [[ARRAY_EXP_NEXT]], i64 1
  // LLVM-NEXT: store ptr [[CALL_PTR]], ptr [[ARR_8]], align 8
  // LLVM-NEXT: ret void
  // CIR-LABEL: cir.{{.*}}@_ZN7gh681985foo25Ev()
  // CIR: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arr8", init] {alignment = 8 : i64}
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<8> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_TO_ARR:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store{{.*}} %[[ONE]], %[[ALLOC_TO_ARR]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_TO_ARR]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store{{.*}} %[[TWO]], %[[ELT1]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[ALLOC_TO_ARR]], %[[ARR_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  void foo25() {
    int* arr8 = new int[](1, 2);
  }

  // LLVM: define {{.*}} void @{{.*foo26.*}} {
  // LLVM: [[ARR_10:%.*]] = alloca ptr
  // LLVM-NEXT: [[CALL_PTR]] = call {{.*}}nonnull ptr @_Znam(i64 noundef 16)
  // LLVM: [[ARRAYINIT_ELEMENT:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 1
  // LLVM-NEXT: store i32 2, ptr [[ARRAYINIT_ELEMENT]], align 4
  // LLVM-NEXT: [[ARRAY_EXP_NEXT:%.*]] = getelementptr {{.*}}[2 x i32], ptr {{.*}}, i64 1
  // LLVM: store i32 3, ptr {{.*}}, align 4
  // LLVM-NEXT: [[ARRAYINIT_ELEMENT2:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 1
  // LLVM-NEXT: store i32 4, ptr [[ARRAYINIT_ELEMENT2]], align 4
  // LLVM-NEXT: [[ARRAY_EXP_NEXT3:%.*]] = getelementptr {{.*}}[2 x i32], ptr [[ARRAY_EXP_NEXT]], i64 1
  // LLVM-NEXT: store ptr [[CALL_PTR]], ptr [[ARR_10]], align 8
  // LLVM-NEXT: ret void
  // CIR-LABEL: cir.{{.*}}@_ZN7gh681985foo26Ev()
  // CIR: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["arr9", init] {alignment = 8 : i64}
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<16> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_TO_ARR:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!cir.array<!s32i x 2>>
  // CIR: %[[ELT0_0:.*]] = cir.cast array_to_ptrdecay %[[ALLOC_TO_ARR]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store{{.*}} %[[ONE]], %[[ELT0_0]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
  // CIR: %[[ELT0_1:.*]] = cir.ptr_stride %[[ELT0_0]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store{{.*}} %[[TWO]], %[[ELT0_1]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_TO_ARR]], %[[ONE]] : (!cir.ptr<!cir.array<!s32i x 2>>, !s32i) -> !cir.ptr<!cir.array<!s32i x 2>>
  // CIR: %[[ELT1_0:.*]] = cir.cast array_to_ptrdecay %[[ELT1]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store{{.*}} %[[THREE]], %[[ELT1_0]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
  // CIR: %[[ELT1_1:.*]] = cir.ptr_stride %[[ELT1_0]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
  // CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: cir.store{{.*}} %[[FOUR]], %[[ELT1_1]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!cir.array<!s32i x 2>>, !s32i) -> !cir.ptr<!cir.array<!s32i x 2>>
  // CIR: %[[ARR_TO_VOID:.*]] = cir.cast bitcast %[[ALLOC_TO_ARR]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!void>
  // CIR: cir.store{{.*}} %[[ARR_TO_VOID]], %[[ARR_ALLOCA]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  void foo26() {
    void* arr9 = new int[][2]({1, 2}, {3, 4});
  }

  // LLVM: define {{.*}} void @{{.*foo27.*}} {
  // LLVM: [[ARR_10:%.*]] = alloca ptr
  // LLVM-NEXT: [[CALL_PTR]] = call {{.*}}nonnull {{.*}}@_Znam(i64 noundef 32)
  // LLVM: store i32 5, ptr {{.*}}, align 4
  // LLVM-NEXT: [[ARRAYINIT_ELEMENT:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 1
  // LLVM-NEXT: store i32 6, ptr [[ARRAYINIT_ELEMENT]], align 4
  // LLVM-NEXT: [[ARRAY_EXP_NEXT:%.*]] = getelementptr {{.*}}[2 x i32], ptr {{.*}}, i64 1
  // LLVM: store i32 7, ptr {{.*}}, align 4
  // LLVM-NEXT: [[ARRAYINIT_ELEMENT2:%.*]] = getelementptr {{.*}}i32, ptr {{.*}}, i64 1
  // LLVM-NEXT: store i32 8, ptr [[ARRAYINIT_ELEMENT2]], align 4
  // LLVM-NEXT: [[ARRAY_EXP_NEXT3:%.*]] = getelementptr {{.*}}[2 x i32], ptr [[ARRAY_EXP_NEXT]], i64 1
  // LLVM-NEXT: call void @llvm.memset.p0.i64(ptr {{.*}}[[ARRAY_EXP_NEXT3]], i8 0, i64 16, i1 false)
  // LLVM-NEXT: store ptr [[CALL_PTR]], ptr [[ARR_10]], align 8
  // LLVM-NEXT: ret void
  // CIR-LABEL: cir.{{.*}}@_ZN7gh681985foo27Ev()
  // CIR: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["arr10", init] {alignment = 8 : i64}
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<32> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_TO_ARR:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!cir.array<!s32i x 2>>
  // CIR: %[[ELT0_0:.*]] = cir.cast array_to_ptrdecay %[[ALLOC_TO_ARR]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
  // CIR: %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
  // CIR: cir.store{{.*}} %[[FIVE]], %[[ELT0_0]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
  // CIR: %[[ELT0_1:.*]] = cir.ptr_stride %[[ELT0_0]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
  // CIR: %[[SIX:.*]] = cir.const #cir.int<6> : !s32i
  // CIR: cir.store{{.*}} %[[SIX]], %[[ELT0_1]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_TO_ARR]], %[[ONE]] : (!cir.ptr<!cir.array<!s32i x 2>>, !s32i) -> !cir.ptr<!cir.array<!s32i x 2>>
  // CIR: %[[ELT1_0:.*]] = cir.cast array_to_ptrdecay %[[ELT1]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
  // CIR: %[[SEVEN:.*]] = cir.const #cir.int<7> : !s32i
  // CIR: cir.store{{.*}} %[[SEVEN]], %[[ELT1_0]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
  // CIR: %[[ELT1_1:.*]] = cir.ptr_stride %[[ELT1_0]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
  // CIR: %[[EIGHT:.*]] = cir.const #cir.int<8> : !s32i
  // CIR: cir.store{{.*}} %[[EIGHT]], %[[ELT1_1]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!cir.array<!s32i x 2>>, !s32i) -> !cir.ptr<!cir.array<!s32i x 2>>
  // CIR: %[[INIT_SIZE:.*]] = cir.const #cir.int<16> : !u64i
  // CIR: %[[ALLOC_DIFF:.*]] = cir.sub %[[SIZE]], %[[INIT_SIZE]] : !u64i
  // CIR: %[[ELT2_DECAY:.*]] = cir.cast bitcast %[[ELT2]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!void>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: cir.libc.memset %[[ALLOC_DIFF]] bytes at %[[ELT2_DECAY]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i
  // CIR: %[[ARR_TO_VOID:.*]] = cir.cast bitcast %[[ALLOC_TO_ARR]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!void>
  // CIR: cir.store{{.*}} %[[ARR_TO_VOID]], %[[ARR_ALLOCA]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  void foo27() {
    void* arr10 = new int[4][2]({5, 6}, {7, 8});
  }
}
