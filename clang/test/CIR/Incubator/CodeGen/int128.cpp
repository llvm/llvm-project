// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

// TODO: remove the -fno-clangir-call-conv-lowering flag when ABI lowering for
//       int128 is supported.

// CHECK-LABEL: @_Z5test1n
// LLVM-LABEL: @_Z5test1n
__int128 test1(__int128 x) {
  return x;
  // CHECK: cir.return %{{.+}} : !s128i
  // LLVM: ret i128 %{{.+}}
}

// CHECK-LABEL: @_Z5test2o
// LLVM-LABEL: @_Z5test2o
unsigned __int128 test2(unsigned __int128 x) {
  return x;
  // CHECK: cir.return %{{.+}} : !u128i
  // LLVM: ret i128 %{{.+}}
}

// CHECK-LABEL: @_Z11unary_arithn
// LLVM-LABEL: @_Z11unary_arithn
__int128 unary_arith(__int128 x) {
  return ++x;
  // CHECK: %{{.+}} = cir.unary(inc, %{{.+}}) nsw : !s128i, !s128i
  // LLVM: %{{.+}} = add nsw i128 %{{.+}}, 1
}

// CHECK-LABEL: @_Z12binary_arithnn
// LLVM-LABEL: @_Z12binary_arithnn
__int128 binary_arith(__int128 x, __int128 y) {
  return x + y;
  // CHECK: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) nsw : !s128i
  // LLVM: %{{.+}} = add nsw i128 %{{.+}}, %{{.+}}
}

volatile int int_var;
volatile double double_var;

// CHECK-LABEL: @_Z19integral_conversionn
// LLVM-LABEL: @_Z19integral_conversionn
__int128 integral_conversion(__int128 x) {
  int_var = x;
  // CHECK: %[[#VAL:]] = cir.cast integral %{{.+}} : !s128i -> !s32i
  // LLVM: %{{.+}} = trunc i128 %{{.+}} to i32

  return int_var;
  // CHECK: %{{.+}} = cir.cast integral %{{.+}} : !s32i -> !s128i
  // LLVM: %{{.+}} = sext i32 %{{.+}} to i128
}

// CHECK-LABEL: @_Z16float_conversionn
// LLVM-LABEL: @_Z16float_conversionn
__int128 float_conversion(__int128 x) {
  double_var = x;
  // CHECK: %[[#VAL:]] = cir.cast int_to_float %{{.+}} : !s128i -> !cir.double
  // LLVM: %{{.+}} = sitofp i128 %{{.+}} to double

  return double_var;
  // CHECK: %{{.+}} = cir.cast float_to_int %{{.+}} : !cir.double -> !s128i
  // LLVM: %{{.+}} = fptosi double %{{.+}} to i128
}

// CHECK-LABEL: @_Z18boolean_conversionn
// LLVM-LABEL: @_Z18boolean_conversionn
bool boolean_conversion(__int128 x) {
  return x;
  // CHECK: %{{.+}} = cir.cast int_to_bool %{{.+}} : !s128i -> !cir.bool
  // LLVM: %{{.+}} = icmp ne i128 %{{.+}}, 0
}
