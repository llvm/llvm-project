// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

using i10 = signed _BitInt(10);
using u10 = unsigned _BitInt(10);

unsigned _BitInt(1) GlobSize1 = 0;
// CHECK: cir.global external @GlobSize1 = #cir.int<0> : !cir.int<u, 1>

i10 test_signed(i10 arg) {
  return arg;
}

// CHECK: cir.func @_Z11test_signedDB10_(%arg0: !cir.int<s, 10> loc({{.*}}) -> !cir.int<s, 10>
// CHECK: }

u10 test_unsigned(u10 arg) {
  return arg;
}

// CHECK: cir.func @_Z13test_unsignedDU10_(%arg0: !cir.int<u, 10> loc({{.*}}) -> !cir.int<u, 10>
// CHECK: }

i10 test_init() {
  return 42;
}

//      CHECK: cir.func @_Z9test_initv() -> !cir.int<s, 10>
//      CHECK:   %[[#LITERAL:]] = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK-NEXT:   %{{.+}} = cir.cast(integral, %[[#LITERAL]] : !s32i), !cir.int<s, 10>
//      CHECK: }

void test_init_for_mem() {
  i10 x = 42;
}

//      CHECK: cir.func @_Z17test_init_for_memv()
//      CHECK:   %[[#LITERAL:]] = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK-NEXT:   %[[#INIT:]] = cir.cast(integral, %[[#LITERAL]] : !s32i), !cir.int<s, 10>
// CHECK-NEXT:   cir.store %[[#INIT]], %{{.+}} : !cir.int<s, 10>, cir.ptr <!cir.int<s, 10>>
//      CHECK: }

i10 test_arith(i10 lhs, i10 rhs) {
  return lhs + rhs;
}

//      CHECK: cir.func @_Z10test_arithDB10_S_(%arg0: !cir.int<s, 10> loc({{.+}}), %arg1: !cir.int<s, 10> loc({{.+}})) -> !cir.int<s, 10>
//      CHECK:   %[[#LHS:]] = cir.load %{{.+}} : cir.ptr <!cir.int<s, 10>>, !cir.int<s, 10>
// CHECK-NEXT:   %[[#RHS:]] = cir.load %{{.+}} : cir.ptr <!cir.int<s, 10>>, !cir.int<s, 10>
// CHECK-NEXT:   %{{.+}} = cir.binop(add, %[[#LHS]], %[[#RHS]]) : !cir.int<s, 10>
//      CHECK: }

void Size1ExtIntParam(unsigned _BitInt(1) A) {
  unsigned _BitInt(1) B[5];
  B[2] = A;
}

//      CHECK: cir.func @_Z16Size1ExtIntParamDU1_
//      CHECK:   %[[#A:]] = cir.load %{{.+}} : cir.ptr <!cir.int<u, 1>>, !cir.int<u, 1>
// CHECK-NEXT:   %[[#IDX:]] = cir.const(#cir.int<2> : !s32i) : !s32i
// CHECK-NEXT:   %[[#ARRAY:]] = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!cir.int<u, 1> x 5>>), !cir.ptr<!cir.int<u, 1>>
// CHECK-NEXT:   %[[#PTR:]] = cir.ptr_stride(%[[#ARRAY]] : !cir.ptr<!cir.int<u, 1>>, %[[#IDX]] : !s32i), !cir.ptr<!cir.int<u, 1>>
// CHECK-NEXT:   cir.store %[[#A]], %[[#PTR]] : !cir.int<u, 1>, cir.ptr <!cir.int<u, 1>>
//      CHECK: }

struct S {
  _BitInt(17) A;
  _BitInt(10) B;
  _BitInt(17) C;
};

void OffsetOfTest(void) {
  int A = __builtin_offsetof(struct S,A);
  int B = __builtin_offsetof(struct S,B);
  int C = __builtin_offsetof(struct S,C);
}

// CHECK: cir.func @_Z12OffsetOfTestv()
// CHECK:   %{{.+}} = cir.const(#cir.int<0> : !u64i) : !u64i
// CHECK:   %{{.+}} = cir.const(#cir.int<4> : !u64i) : !u64i
// CHECK:   %{{.+}} = cir.const(#cir.int<8> : !u64i) : !u64i
// CHECK: }

_BitInt(2) ParamPassing(_BitInt(15) a, _BitInt(31) b) {}

// CHECK: cir.func @_Z12ParamPassingDB15_DB31_(%arg0: !cir.int<s, 15> loc({{.+}}), %arg1: !cir.int<s, 31> loc({{.+}})) -> !cir.int<s, 2>
