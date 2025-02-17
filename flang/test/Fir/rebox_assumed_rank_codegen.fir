// Test fir.rebox_assumed_rank lowering to runtime calls in fir-assumed-rank-op pass.
// RUN: fir-opt -o - --fir-assumed-rank-op %s | FileCheck %s

func.func @test_simple(%arg0: !fir.box<!fir.array<*:f32>> ) {
  %1 = fir.rebox_assumed_rank %arg0 lbs ones : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<*:f32>>
  fir.call @somefunc(%1) : (!fir.box<!fir.array<*:f32>>) -> ()
  return
}
func.func @test_simple_zeroes(%arg0: !fir.box<!fir.array<*:f32>> ) {
  %1 = fir.rebox_assumed_rank %arg0 lbs zeroes : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<*:f32>>
  fir.call @somefunc(%1) : (!fir.box<!fir.array<*:f32>>) -> ()
  return
}
func.func @test_simple_preserve(%arg0: !fir.box<!fir.array<*:f32>> ) {
  %1 = fir.rebox_assumed_rank %arg0 lbs preserve : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<*:f32>>
  fir.call @somefunc(%1) : (!fir.box<!fir.array<*:f32>>) -> ()
  return
}
func.func @test_allocatable(%arg0: !fir.box<!fir.array<*:f32>> ) {
  %1 = fir.rebox_assumed_rank %arg0 lbs preserve : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.heap<!fir.array<*:f32>>>
  fir.call @somefuncalloc(%1) : (!fir.box<!fir.heap<!fir.array<*:f32>>>) -> ()
  return
}
func.func @test_pointer(%arg0: !fir.box<!fir.array<*:f32>> ) {
  %1 = fir.rebox_assumed_rank %arg0 lbs preserve : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.ptr<!fir.array<*:f32>>>
  fir.call @somefuncpointer(%1) : (!fir.box<!fir.ptr<!fir.array<*:f32>>>) -> ()
  return
}
!t1= !fir.type<t1{i:i32}>
!t2= !fir.type<t2{t1:!t1, x:f32}>
func.func @test_new_dtype(%arg0: !fir.box<!fir.array<*:!t2>> ) {
  %1 = fir.rebox_assumed_rank %arg0 lbs ones : (!fir.box<!fir.array<*:!t2>>) -> !fir.box<!fir.array<*:!t1>>
  fir.call @somefunct1(%1) : (!fir.box<!fir.array<*:!t1>>) -> ()
  return
}

!sometype = !fir.type<sometype{i:i32}>
func.func @test_poly_to_nonepoly(%arg0: !fir.class<!fir.array<*:!sometype>>) {
  %1 = fir.rebox_assumed_rank %arg0 lbs ones : (!fir.class<!fir.array<*:!sometype>>) -> !fir.box<!fir.array<*:!sometype>>
  fir.call @takes_assumed_rank_t(%1) : (!fir.box<!fir.array<*:!sometype>>) -> ()
  return
}

func.func private @somefunc(!fir.box<!fir.array<*:f32>>)
func.func private @somefuncalloc(!fir.box<!fir.heap<!fir.array<*:f32>>>)
func.func private @somefuncpointer(!fir.box<!fir.ptr<!fir.array<*:f32>>>)
func.func private @somefunct1(!fir.box<!fir.array<*:!t1>>)
func.func private @takes_assumed_rank_t(!fir.box<!fir.array<*:!sometype>>)

// CHECK-LABEL:   func.func @test_simple(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>
// CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranACopyAndUpdateDescriptor(%[[VAL_5]], %[[VAL_6]], %[[VAL_4]], %[[VAL_2]], %[[VAL_1]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<none>, i8, i32) -> ()
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>) -> !fir.box<!fir.array<*:f32>>
// CHECK:           fir.call @somefunc(%[[VAL_9]]) : (!fir.box<!fir.array<*:f32>>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @test_simple_zeroes(
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           fir.call @_FortranACopyAndUpdateDescriptor(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_1]])

// CHECK-LABEL:   func.func @test_simple_preserve(
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           fir.call @_FortranACopyAndUpdateDescriptor(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_1]])

// CHECK-LABEL:   func.func @test_allocatable(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i8
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>
// CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranACopyAndUpdateDescriptor(%[[VAL_5]], %[[VAL_6]], %[[VAL_4]], %[[VAL_2]], %[[VAL_1]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<none>, i8, i32) -> ()
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>) -> !fir.box<!fir.heap<!fir.array<*:f32>>>
// CHECK:           fir.call @somefuncalloc(%[[VAL_9]]) : (!fir.box<!fir.heap<!fir.array<*:f32>>>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @test_pointer(
// CHECK-SAME:                            %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i8
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>
// CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranACopyAndUpdateDescriptor(%[[VAL_5]], %[[VAL_6]], %[[VAL_4]], %[[VAL_2]], %[[VAL_1]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<none>, i8, i32) -> ()
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>>) -> !fir.box<!fir.ptr<!fir.array<*:f32>>>
// CHECK:           fir.call @somefuncpointer(%[[VAL_9]]) : (!fir.box<!fir.ptr<!fir.array<*:f32>>>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @test_new_dtype(
// CHECK-SAME:                              %[[VAL_0:.*]]: !fir.box<!fir.array<*:!fir.type<t2{t1:!fir.type<t1{i:i32}>,x:f32}>>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?x!fir.type<t1{i:i32}>>>
// CHECK:           %[[VAL_4:.*]] = fir.type_desc !fir.type<t1{i:i32}>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?x!fir.type<t1{i:i32}>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:!fir.type<t2{t1:!fir.type<t1{i:i32}>,x:f32}>>>) -> !fir.box<none>
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.tdesc<!fir.type<t1{i:i32}>>) -> !fir.ref<none>
// CHECK:           fir.call @_FortranACopyAndUpdateDescriptor(%[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_2]], %[[VAL_1]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<none>, i8, i32) -> ()
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?x!fir.type<t1{i:i32}>>>>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?x!fir.type<t1{i:i32}>>>) -> !fir.box<!fir.array<*:!fir.type<t1{i:i32}>>>
// CHECK:           fir.call @somefunct1(%[[VAL_10]]) : (!fir.box<!fir.array<*:!fir.type<t1{i:i32}>>>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @test_poly_to_nonepoly(
// CHECK:           %[[VAL_4:.*]] = fir.type_desc !fir.type<sometype{i:i32}>
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.tdesc<!fir.type<sometype{i:i32}>>) -> !fir.ref<none>
// CHECK:           fir.call @_FortranACopyAndUpdateDescriptor(%{{.*}}, %{{.*}}, %[[VAL_7]],

// CHECK:         func.func private @_FortranACopyAndUpdateDescriptor(!fir.ref<!fir.box<none>> {llvm.nocapture}, !fir.box<none> {llvm.nocapture}, !fir.ref<none>, i8, i32) attributes {fir.runtime}
