// Test that bind_c derived type results are not moved to a hidden argument
// by the abstract-result pass. They will be dealt with according to the C
// struct returning ABI for the target in the target-rewrite pass.
// RUN: fir-opt %s --abstract-result | FileCheck %s

!t = !fir.type<t{i:f32, j: i32, k: f32}>
!cptr = !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>

func.func private @foo() -> !t attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

func.func @test_call(%x: !fir.ref<!t>) {
  %0 = fir.call @foo() proc_attrs<bind_c> : () -> !t
  fir.save_result %0 to %x : !t, !fir.ref<!t>
  return
}

func.func @test_addr_of() -> (() -> !t) {
  %0 = fir.address_of(@foo) : () -> !t
  return %0 : () -> !t
}

func.func @test_dispatch(%x: !fir.ref<!t>, %y : !fir.class<!fir.type<somet>>) {
  %0 = fir.dispatch "bar"(%y : !fir.class<!fir.type<somet>>) (%y : !fir.class<!fir.type<somet>>) -> !t proc_attrs<bind_c> {pass_arg_pos = 0 : i32}
  fir.save_result %0 to %x : !t, !fir.ref<!t>
  return
}


func.func private @return_cptr() -> !cptr attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
func.func @test_return_cptr(%x: !fir.ref<!cptr>) {
  %0 = fir.call @return_cptr() proc_attrs<bind_c> : () -> !cptr
  fir.save_result %0 to %x : !cptr, !fir.ref<!cptr>
  return
}


// CHECK-LABEL:   func.func @test_call(
// CHECK-SAME:                         %[[VAL_0:.*]]: !fir.ref<!fir.type<t{i:f32,j:i32,k:f32}>>) {
// CHECK:           %[[VAL_1:.*]] = fir.call @foo() proc_attrs<bind_c> : () -> !fir.type<t{i:f32,j:i32,k:f32}>
// CHECK:           fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<!fir.type<t{i:f32,j:i32,k:f32}>>
// CHECK:           return
// CHECK:         }
// CHECK-LABEL:   func.func @test_addr_of() -> (() -> !fir.type<t{i:f32,j:i32,k:f32}>) {
// CHECK:           %[[VAL_0:.*]] = fir.address_of(@foo) : () -> !fir.type<t{i:f32,j:i32,k:f32}>
// CHECK:           return %[[VAL_0]] : () -> !fir.type<t{i:f32,j:i32,k:f32}>
// CHECK:         }
// CHECK-LABEL:   func.func @test_dispatch(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.type<t{i:f32,j:i32,k:f32}>>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !fir.class<!fir.type<somet>>) {
// CHECK:           %[[VAL_2:.*]] = fir.dispatch "bar"(%[[VAL_1]] : !fir.class<!fir.type<somet>>) (%[[VAL_1]] : !fir.class<!fir.type<somet>>) -> !fir.type<t{i:f32,j:i32,k:f32}> proc_attrs <bind_c> {pass_arg_pos = 0 : i32}
// CHECK:           fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.type<t{i:f32,j:i32,k:f32}>>
// CHECK:           return
// CHECK:         }
// CHECK-LABEL:   func.func @test_return_cptr(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) {
// CHECK:           %[[VAL_1:.*]] = fir.call @return_cptr() : () -> !fir.ref<none>
// CHECK:           %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<none>) -> i64
// CHECK:           fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ref<i64>
