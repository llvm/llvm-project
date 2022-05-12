// RUN: tco %s | FileCheck %s

// UNSUPPORTED: system-windows

// CHECK-LABEL: define i32* @alloca_scalar_nonchar()
// CHECK: alloca i32, i64 1
func.func @alloca_scalar_nonchar() -> !fir.ref<i32> {
  %1 = fir.alloca i32
  return %1 : !fir.ref<i32>
}

// CHECK-LABEL: define i32* @alloca_scalars_nonchar()
// CHECK: alloca i32, i64 100
func.func @alloca_scalars_nonchar() -> !fir.ref<i32> {
  %0 = arith.constant 100 : index
  %1 = fir.alloca i32, %0
  return %1 : !fir.ref<i32>
}

// CHECK-LABEL: define i32* @allocmem_scalar_nonchar(
// CHECK: call i8* @malloc(i64 4)
func.func @allocmem_scalar_nonchar() -> !fir.heap<i32> {
  %1 = fir.allocmem i32
  return %1 : !fir.heap<i32>
}

// CHECK-LABEL: define i32* @allocmem_scalars_nonchar(
// CHECK: call i8* @malloc(i64 400)
func.func @allocmem_scalars_nonchar() -> !fir.heap<i32> {
  %0 = arith.constant 100 : index
  %1 = fir.allocmem i32, %0
  return %1 : !fir.heap<i32>
}

// CHECK-LABEL: define [10 x i8]* @alloca_scalar_char(
// CHECK: alloca [10 x i8], i64 1
func.func @alloca_scalar_char() -> !fir.ref<!fir.char<1,10>> {
  %1 = fir.alloca !fir.char<1,10>
  return %1 : !fir.ref<!fir.char<1,10>>
}

// CHECK-LABEL: define [10 x i16]* @alloca_scalar_char_kind(
// CHECK: alloca [10 x i16], i64 1
func.func @alloca_scalar_char_kind() -> !fir.ref<!fir.char<2,10>> {
  %1 = fir.alloca !fir.char<2,10>
  return %1 : !fir.ref<!fir.char<2,10>>
}

// CHECK-LABEL: define [10 x i8]* @allocmem_scalar_char(
// CHECK: call i8* @malloc(i64 ptrtoint ([10 x i8]* getelementptr ([10 x i8], [10 x i8]* null, i64 1) to i64))
func.func @allocmem_scalar_char() -> !fir.heap<!fir.char<1,10>> {
  %1 = fir.allocmem !fir.char<1,10>
  return %1 : !fir.heap<!fir.char<1,10>>
}

// CHECK-LABEL: define [10 x i16]* @allocmem_scalar_char_kind(
// CHECK: call i8* @malloc(i64 ptrtoint ([10 x i16]* getelementptr ([10 x i16], [10 x i16]* null, i64 1) to i64))
func.func @allocmem_scalar_char_kind() -> !fir.heap<!fir.char<2,10>> {
  %1 = fir.allocmem !fir.char<2,10>
  return %1 : !fir.heap<!fir.char<2,10>>
}

// CHECK-LABEL: define i8* @alloca_scalar_dynchar(
// CHECK-SAME: i32 %[[len:.*]])
// CHECK: %[[mul1:.*]] = sext i32 %[[len]] to i64
// CHECK: alloca i8, i64 %[[mul1]]
func.func @alloca_scalar_dynchar(%l : i32) -> !fir.ref<!fir.char<1,?>> {
  %1 = fir.alloca !fir.char<1,?>(%l : i32)
  return %1 : !fir.ref<!fir.char<1,?>>
}

// CHECK-LABEL: define i16* @alloca_scalar_dynchar_kind(
// CHECK-SAME: i32 %[[len:.*]])
// CHECK: %[[mul1:.*]] = sext i32 %[[len]] to i64
// CHECK: alloca i16, i64 %[[mul1]]
func.func @alloca_scalar_dynchar_kind(%l : i32) -> !fir.ref<!fir.char<2,?>> {
  %1 = fir.alloca !fir.char<2,?>(%l : i32)
  return %1 : !fir.ref<!fir.char<2,?>>
}

// CHECK-LABEL: define i8* @allocmem_scalar_dynchar(
// CHECK-SAME: i32 %[[len:.*]])
// CHECK: %[[mul1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[mul2:.*]] = mul i64 1, %[[mul1]]
// CHECK: call i8* @malloc(i64 %[[mul2]])
func.func @allocmem_scalar_dynchar(%l : i32) -> !fir.heap<!fir.char<1,?>> {
  %1 = fir.allocmem !fir.char<1,?>(%l : i32)
  return %1 : !fir.heap<!fir.char<1,?>>
}

// CHECK-LABEL: define i16* @allocmem_scalar_dynchar_kind(
// CHECK-SAME: i32 %[[len:.*]])
// CHECK: %[[mul1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[mul2:.*]] = mul i64 2, %[[mul1]]
// CHECK: call i8* @malloc(i64 %[[mul2]])
func.func @allocmem_scalar_dynchar_kind(%l : i32) -> !fir.heap<!fir.char<2,?>>{
  %1 = fir.allocmem !fir.char<2,?>(%l : i32)
  return %1 : !fir.heap<!fir.char<2,?>>
}

// CHECK-LABEL: define i32** @alloca_ptr_to_dynarray_nonchar(
// CHECK: %1 = alloca i32*, i64 1
func.func @alloca_ptr_to_dynarray_nonchar() -> !fir.ref<!fir.ptr<!fir.array<?xi32>>> {
  %1 = fir.alloca !fir.ptr<!fir.array<?xi32>>
  return %1 : !fir.ref<!fir.ptr<!fir.array<?xi32>>>
}

// CHECK-LABEL: define [3 x [3 x i32]]* @alloca_array_of_nonchar(
// CHECK: alloca [3 x [3 x i32]], i64 1
func.func @alloca_array_of_nonchar() -> !fir.ref<!fir.array<3x3xi32>> {
  %1 = fir.alloca !fir.array<3x3xi32>
  return %1 : !fir.ref<!fir.array<3x3xi32>>
}

// CHECK-LABEL: define [3 x [3 x [10 x i8]]]* @alloca_array_of_char(
// CHECK: alloca [3 x [3 x [10 x i8]]], i64 1
func.func @alloca_array_of_char() -> !fir.ref<!fir.array<3x3x!fir.char<1,10>>> {
  %1 = fir.alloca !fir.array<3x3x!fir.char<1,10>>
  return %1 : !fir.ref<!fir.array<3x3x!fir.char<1,10>>>
}

// CHECK-LABEL: define i8* @alloca_array_of_dynchar(
// CHECK-SAME: i32 %[[len:.*]])
// CHECK: %[[mul1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[mul2:.*]] = mul i64 %[[mul1]], 9
// CHECK: alloca i8, i64 %[[mul2]]
func.func @alloca_array_of_dynchar(%l: i32) -> !fir.ref<!fir.array<3x3x!fir.char<1,?>>> {
  %1 = fir.alloca !fir.array<3x3x!fir.char<1,?>>(%l : i32)
  return %1 : !fir.ref<!fir.array<3x3x!fir.char<1,?>>>
}

// CHECK-LABEL: define [3 x [3 x i32]]* @allocmem_array_of_nonchar(
// CHECK: call i8* @malloc(i64 ptrtoint ([3 x [3 x i32]]* getelementptr ([3 x [3 x i32]], [3 x [3 x i32]]* null, i64 1) to i64))
func.func @allocmem_array_of_nonchar() -> !fir.heap<!fir.array<3x3xi32>> {
  %1 = fir.allocmem !fir.array<3x3xi32>
  return %1 : !fir.heap<!fir.array<3x3xi32>>
}

// CHECK-LABEL: define [3 x [3 x [10 x i8]]]* @allocmem_array_of_char(
// CHECK: call i8* @malloc(i64 ptrtoint ([3 x [3 x [10 x i8]]]* getelementptr ([3 x [3 x [10 x i8]]], [3 x [3 x [10 x i8]]]* null, i64 1) to i64))
func.func @allocmem_array_of_char() -> !fir.heap<!fir.array<3x3x!fir.char<1,10>>> {
  %1 = fir.allocmem !fir.array<3x3x!fir.char<1,10>>
  return %1 : !fir.heap<!fir.array<3x3x!fir.char<1,10>>>
}

// CHECK-LABEL: define i8* @allocmem_array_of_dynchar(
// CHECK-SAME: i32 %[[len:.*]])
// CHECK: %[[mul1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[mul2:.*]] = mul i64 9, %[[mul1]]
// CHECK: call i8* @malloc(i64 %[[mul2]])
func.func @allocmem_array_of_dynchar(%l: i32) -> !fir.heap<!fir.array<3x3x!fir.char<1,?>>> {
  %1 = fir.allocmem !fir.array<3x3x!fir.char<1,?>>(%l : i32)
  return %1 : !fir.heap<!fir.array<3x3x!fir.char<1,?>>>
}

// CHECK-LABEL: define [3 x i32]* @alloca_dynarray_of_nonchar(
// CHECK-SAME: i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = mul i64 1, %[[extent]]
// CHECK: alloca [3 x i32], i64 %[[prod1]]
func.func @alloca_dynarray_of_nonchar(%e: index) -> !fir.ref<!fir.array<3x?xi32>> {
  %1 = fir.alloca !fir.array<3x?xi32>, %e
  return %1 : !fir.ref<!fir.array<3x?xi32>>
}

// CHECK-LABEL: define i32* @alloca_dynarray_of_nonchar2(
// CHECK-SAME: i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = mul i64 1, %[[extent]]
// CHECK: %[[prod2:.*]] = mul i64 %[[prod1]], %[[extent]]
// CHECK: alloca i32, i64 %[[prod2]]
func.func @alloca_dynarray_of_nonchar2(%e: index) -> !fir.ref<!fir.array<?x?xi32>> {
  %1 = fir.alloca !fir.array<?x?xi32>, %e, %e
  return %1 : !fir.ref<!fir.array<?x?xi32>>
}

// CHECK-LABEL: define [3 x i32]* @allocmem_dynarray_of_nonchar(
// CHECK-SAME: i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = mul i64 ptrtoint ([3 x i32]* getelementptr ([3 x i32], [3 x i32]* null, i64 1) to i64), %[[extent]]
// CHECK: call i8* @malloc(i64 %[[prod1]])
func.func @allocmem_dynarray_of_nonchar(%e: index) -> !fir.heap<!fir.array<3x?xi32>> {
  %1 = fir.allocmem !fir.array<3x?xi32>, %e
  return %1 : !fir.heap<!fir.array<3x?xi32>>
}

// CHECK-LABEL: define i32* @allocmem_dynarray_of_nonchar2(
// CHECK-SAME: i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = mul i64 4, %[[extent]]
// CHECK: %[[prod2:.*]] = mul i64 %[[prod1]], %[[extent]]
// CHECK: call i8* @malloc(i64 %[[prod2]])
func.func @allocmem_dynarray_of_nonchar2(%e: index) -> !fir.heap<!fir.array<?x?xi32>> {
  %1 = fir.allocmem !fir.array<?x?xi32>, %e, %e
  return %1 : !fir.heap<!fir.array<?x?xi32>>
}

// CHECK-LABEL: define [3 x [10 x i16]]* @alloca_dynarray_of_char(
// CHECK-SAME: i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = mul i64 1, %[[extent]]
// CHECK: alloca [3 x [10 x i16]], i64 %[[prod1]]
func.func @alloca_dynarray_of_char(%e : index) -> !fir.ref<!fir.array<3x?x!fir.char<2,10>>> {
  %1 = fir.alloca !fir.array<3x?x!fir.char<2,10>>, %e
  return %1 : !fir.ref<!fir.array<3x?x!fir.char<2,10>>>
}

// CHECK-LABEL: define [10 x i16]* @alloca_dynarray_of_char2(
// CHECK-SAME: i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = mul i64 1, %[[extent]]
// CHECK: %[[prod2:.*]] = mul i64 %[[prod1]], %[[extent]]
// CHECK: alloca [10 x i16], i64 %[[prod2]]
func.func @alloca_dynarray_of_char2(%e : index) -> !fir.ref<!fir.array<?x?x!fir.char<2,10>>> {
  %1 = fir.alloca !fir.array<?x?x!fir.char<2,10>>, %e, %e
  return %1 : !fir.ref<!fir.array<?x?x!fir.char<2,10>>>
}

// CHECK-LABEL: define [3 x [10 x i16]]* @allocmem_dynarray_of_char(
// CHECK-SAME: i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = mul i64 ptrtoint ([3 x [10 x i16]]* getelementptr ([3 x [10 x i16]], [3 x [10 x i16]]* null, i64 1) to i64), %[[extent]]
// CHECK: call i8* @malloc(i64 %[[prod1]])
func.func @allocmem_dynarray_of_char(%e : index) -> !fir.heap<!fir.array<3x?x!fir.char<2,10>>> {
  %1 = fir.allocmem !fir.array<3x?x!fir.char<2,10>>, %e
  return %1 : !fir.heap<!fir.array<3x?x!fir.char<2,10>>>
}

// CHECK-LABEL: define [10 x i16]* @allocmem_dynarray_of_char2(
// CHECK-SAME: i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = mul i64 ptrtoint ([10 x i16]* getelementptr ([10 x i16], [10 x i16]* null, i64 1) to i64), %[[extent]]
// CHECK: %[[prod2:.*]] = mul i64 %[[prod1]], %[[extent]]
// CHECK: call i8* @malloc(i64 %[[prod2]])
func.func @allocmem_dynarray_of_char2(%e : index) -> !fir.heap<!fir.array<?x?x!fir.char<2,10>>> {
  %1 = fir.allocmem !fir.array<?x?x!fir.char<2,10>>, %e, %e
  return %1 : !fir.heap<!fir.array<?x?x!fir.char<2,10>>>
}

// CHECK-LABEL: define i16* @alloca_dynarray_of_dynchar(
// CHECK-SAME: i32 %[[len:.*]], i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[prod2:.*]] = mul i64 %[[prod1]], 3
// CHECK: %[[prod3:.*]] = mul i64 %[[prod2]], %[[extent]]
// CHECK: alloca i16, i64 %[[prod3]]
func.func @alloca_dynarray_of_dynchar(%l: i32, %e : index) -> !fir.ref<!fir.array<3x?x!fir.char<2,?>>> {
  %1 = fir.alloca !fir.array<3x?x!fir.char<2,?>>(%l : i32), %e
  return %1 : !fir.ref<!fir.array<3x?x!fir.char<2,?>>>
}

// CHECK-LABEL: define i16* @alloca_dynarray_of_dynchar2(
// CHECK-SAME: i32 %[[len:.*]], i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[prod2:.*]] = mul i64 %[[prod1]], %[[extent]]
// CHECK: %[[prod3:.*]] = mul i64 %[[prod2]], %[[extent]]
// CHECK: alloca i16, i64 %[[prod3]]
func.func @alloca_dynarray_of_dynchar2(%l: i32, %e : index) -> !fir.ref<!fir.array<?x?x!fir.char<2,?>>> {
  %1 = fir.alloca !fir.array<?x?x!fir.char<2,?>>(%l : i32), %e, %e
  return %1 : !fir.ref<!fir.array<?x?x!fir.char<2,?>>>
}

// CHECK-LABEL: define i16* @allocmem_dynarray_of_dynchar(
// CHECK-SAME: i32 %[[len:.*]], i64 %[[extent:.*]])
// CHECK: %[[prod1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[prod2:.*]] = mul i64 6, %[[prod1]]
// CHECK: %[[prod3:.*]] = mul i64 %[[prod2]], %[[extent]]
// CHECK: call i8* @malloc(i64 %[[prod3]])
func.func @allocmem_dynarray_of_dynchar(%l: i32, %e : index) -> !fir.heap<!fir.array<3x?x!fir.char<2,?>>> {
  %1 = fir.allocmem !fir.array<3x?x!fir.char<2,?>>(%l : i32), %e
  return %1 : !fir.heap<!fir.array<3x?x!fir.char<2,?>>>
}

// CHECK-LABEL: define i16* @allocmem_dynarray_of_dynchar2(
// CHECK-SAME: i32 %[[len:.*]], i64 %[[extent:.*]])
// CHECK: %[[a:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[prod1:.*]] = mul i64 2, %[[a]]
// CHECK: %[[prod2:.*]] = mul i64 %[[prod1]], %[[extent]]
// CHECK: %[[prod3:.*]] = mul i64 %[[prod2]], %[[extent]]
// CHECK: call i8* @malloc(i64 %[[prod3]])
func.func @allocmem_dynarray_of_dynchar2(%l: i32, %e : index) -> !fir.heap<!fir.array<?x?x!fir.char<2,?>>> {
  %1 = fir.allocmem !fir.array<?x?x!fir.char<2,?>>(%l : i32), %e, %e
  return %1 : !fir.heap<!fir.array<?x?x!fir.char<2,?>>>
}

// CHECK-LABEL: define i32* @alloca_array_with_holes_nonchar(
// CHECK-SAME: i64 %[[a:.*]], i64 %[[b:.*]])
// CHECK: %[[prod1:.*]] = mul i64 60, %[[a]]
// CHECK: %[[prod2:.*]] = mul i64 %[[prod1]], %[[b]]
// CHECK: alloca i32, i64 %[[prod2]]
func.func @alloca_array_with_holes_nonchar(%0 : index, %1 : index) -> !fir.ref<!fir.array<4x?x3x?x5xi32>> {
  %a = fir.alloca !fir.array<4x?x3x?x5xi32>, %0, %1
  return %a : !fir.ref<!fir.array<4x?x3x?x5xi32>>
}

// CHECK-LABEL: define [10 x i16]* @alloca_array_with_holes_char(
// CHECK-SAME: i64 %[[e:.*]])
// CHECK: %[[mul:.*]] = mul i64 12, %[[e]]
// CHECK: alloca [10 x i16], i64 %[[mul]]
func.func @alloca_array_with_holes_char(%e: index) -> !fir.ref<!fir.array<3x?x4x!fir.char<2,10>>> {
  %1 = fir.alloca !fir.array<3x?x4x!fir.char<2,10>>, %e
  return %1 : !fir.ref<!fir.array<3x?x4x!fir.char<2,10>>>
}

// CHECK-LABEL: define i16* @alloca_array_with_holes_dynchar(
// CHECK-SAME: i64 %[[len:.*]], i64 %[[extent:.*]])
// CHECK: %[[a:.*]] = mul i64 %[[len]], 12
// CHECK: %[[b:.*]] = mul i64 %[[a]], %[[extent]]
// CHECK: alloca i16, i64 %[[b]]
func.func @alloca_array_with_holes_dynchar(%arg0: index, %arg1: index) -> !fir.ref<!fir.array<3x?x4x!fir.char<2,?>>> {
  %1 = fir.alloca !fir.array<3x?x4x!fir.char<2,?>>(%arg0 : index), %arg1
  return %1 : !fir.ref<!fir.array<3x?x4x!fir.char<2,?>>>
}

// CHECK-LABEL: define i32* @allocmem_array_with_holes_nonchar(
// CHECK-SAME: i64 %[[e1:.*]], i64 %[[e2:.*]])
// CHECK: %[[a:.*]] = mul i64 240, %[[e1]]
// CHECK: %[[b:.*]] = mul i64 %3, %[[e2]]
// CHECK: call i8* @malloc(i64 %[[b]])
func.func @allocmem_array_with_holes_nonchar(%0 : index, %1 : index) -> !fir.heap<!fir.array<4x?x3x?x5xi32>> {
  %a = fir.allocmem !fir.array<4x?x3x?x5xi32>, %0, %1
  return %a : !fir.heap<!fir.array<4x?x3x?x5xi32>>
}

// CHECK-LABEL: define [10 x i16]* @allocmem_array_with_holes_char(
// CHECK-SAME: i64 %[[e:.*]])
// CHECK: %[[mul:.*]] = mul i64 mul (i64 ptrtoint ([10 x i16]* getelementptr ([10 x i16], [10 x i16]* null, i64 1) to i64), i64 12), %[[e]]
// CHECK: call i8* @malloc(i64 %[[mul]])
func.func @allocmem_array_with_holes_char(%e: index) -> !fir.heap<!fir.array<3x?x4x!fir.char<2,10>>> {
  %1 = fir.allocmem !fir.array<3x?x4x!fir.char<2,10>>, %e
  return %1 : !fir.heap<!fir.array<3x?x4x!fir.char<2,10>>>
}

// CHECK-LABEL: define i16* @allocmem_array_with_holes_dynchar(
// CHECK-SAME: i64 %[[len:.*]], i64 %[[extent:.*]])
// CHECK: %[[a:.*]] = mul i64 24, %[[len]]
// CHECK: %[[b:.*]] = mul i64 %[[a]], %[[extent]]
// CHECK: call i8* @malloc(i64 %[[b]])
func.func @allocmem_array_with_holes_dynchar(%arg0: index, %arg1: index) -> !fir.heap<!fir.array<3x?x4x!fir.char<2,?>>> {
  %1 = fir.allocmem !fir.array<3x?x4x!fir.char<2,?>>(%arg0 : index), %arg1
  return %1 : !fir.heap<!fir.array<3x?x4x!fir.char<2,?>>>
}

