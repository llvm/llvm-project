// RUN: %flang_fc1 -emit-llvm  %s -o - | FileCheck %s

// UNSUPPORTED: system-windows
// Disabled on 32-bit targets due to the additional `trunc` opcodes required
// UNSUPPORTED: target-x86
// UNSUPPORTED: target=sparc-{{.*}}
// UNSUPPORTED: target=sparcel-{{.*}}

// CHECK-LABEL: define void @omp_target_allocmem_scalar_nonchar() {
// CHECK-NEXT:    [[TMP1:%.*]] = call ptr @omp_target_alloc(i64 4, i32 0)
// CHECK-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[TMP1]] to i64
// CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP3]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_scalar_nonchar() -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, i32
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_scalars_nonchar() {
// CHECK-NEXT:    [[TMP1:%.*]] = call ptr @omp_target_alloc(i64 400, i32 0)
// CHECK-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[TMP1]] to i64
// CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP3]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_scalars_nonchar() -> () {
  %device = arith.constant 0 : i32
  %0 = arith.constant 100 : index
  %1 = omp.target_allocmem %device : i32, i32, %0
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_scalar_char() {
// CHECK-NEXT:    [[TMP1:%.*]] = call ptr @omp_target_alloc(i64 10, i32 0)
// CHECK-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[TMP1]] to i64
// CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP3]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_scalar_char() -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.char<1,10>
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_scalar_char_kind() {
// CHECK-NEXT:    [[TMP1:%.*]] = call ptr @omp_target_alloc(i64 20, i32 0)
// CHECK-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[TMP1]] to i64
// CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP3]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_scalar_char_kind() -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.char<2,10>
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_scalar_dynchar(
// CHECK-SAME: i32 [[TMP0:%.*]]) {
// CHECK-NEXT:    [[TMP2:%.*]] = sext i32 [[TMP0]] to i64
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 1, [[TMP2]]
// CHECK-NEXT:    [[TMP4:%.*]] = mul i64 1, [[TMP3]]
// CHECK-NEXT:    [[TMP5:%.*]] = call ptr @omp_target_alloc(i64 [[TMP4]], i32 0)
// CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
// CHECK-NEXT:    [[TMP7:%.*]] = inttoptr i64 [[TMP6]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP7]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_scalar_dynchar(%l : i32) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.char<1,?>(%l : i32)
  omp.target_freemem %device, %1 : i32, i64
  return
}


// CHECK-LABEL: define void @omp_target_allocmem_scalar_dynchar_kind(
// CHECK-SAME: i32 [[TMP0:%.*]]) {
// CHECK-NEXT:    [[TMP2:%.*]] = sext i32 [[TMP0]] to i64
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 2, [[TMP2]]
// CHECK-NEXT:    [[TMP4:%.*]] = mul i64 1, [[TMP3]]
// CHECK-NEXT:    [[TMP5:%.*]] = call ptr @omp_target_alloc(i64 [[TMP4]], i32 0)
// CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
// CHECK-NEXT:    [[TMP7:%.*]] = inttoptr i64 [[TMP6]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP7]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_scalar_dynchar_kind(%l : i32) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.char<2,?>(%l : i32)
  omp.target_freemem %device, %1 : i32, i64
  return
}


// CHECK-LABEL: define void @omp_target_allocmem_array_of_nonchar() {
// CHECK-NEXT:    [[TMP1:%.*]] = call ptr @omp_target_alloc(i64 36, i32 0)
// CHECK-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[TMP1]] to i64
// CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP3]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_array_of_nonchar() -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x3xi32>
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_array_of_char() {
// CHECK-NEXT:    [[TMP1:%.*]] = call ptr @omp_target_alloc(i64 90, i32 0)
// CHECK-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[TMP1]] to i64
// CHECK-NEXT:    [[TMP3:%.*]] = inttoptr i64 [[TMP2]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP3]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_array_of_char() -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x3x!fir.char<1,10>>
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_array_of_dynchar(
// CHECK-SAME: i32 [[TMP0:%.*]]) {
// CHECK-NEXT:    [[TMP2:%.*]] = sext i32 [[TMP0]] to i64
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 9, [[TMP2]]
// CHECK-NEXT:    [[TMP4:%.*]] = mul i64 1, [[TMP3]]
// CHECK-NEXT:    [[TMP5:%.*]] = call ptr @omp_target_alloc(i64 [[TMP4]], i32 0)
// CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
// CHECK-NEXT:    [[TMP7:%.*]] = inttoptr i64 [[TMP6]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP7]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_array_of_dynchar(%l: i32) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x3x!fir.char<1,?>>(%l : i32)
  omp.target_freemem %device, %1 : i32, i64
  return
}


// CHECK-LABEL: define void @omp_target_allocmem_dynarray_of_nonchar(
// CHECK-SAME: i64 [[TMP0:%.*]]) {
// CHECK-NEXT:    [[TMP2:%.*]] = mul i64 12, [[TMP0]]
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 1, [[TMP2]]
// CHECK-NEXT:    [[TMP4:%.*]] = call ptr @omp_target_alloc(i64 [[TMP3]], i32 0)
// CHECK-NEXT:    [[TMP5:%.*]] = ptrtoint ptr [[TMP4]] to i64
// CHECK-NEXT:    [[TMP6:%.*]] = inttoptr i64 [[TMP5]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP6]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_dynarray_of_nonchar(%e: index) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x?xi32>, %e
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_dynarray_of_nonchar2(
// CHECK-SAME: i64 [[TMP0:%.*]]) {
// CHECK-NEXT:    [[TMP2:%.*]] = mul i64 4, [[TMP0]]
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], [[TMP0]]
// CHECK-NEXT:    [[TMP4:%.*]] = mul i64 1, [[TMP3]]
// CHECK-NEXT:    [[TMP5:%.*]] = call ptr @omp_target_alloc(i64 [[TMP4]], i32 0)
// CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
// CHECK-NEXT:    [[TMP7:%.*]] = inttoptr i64 [[TMP6]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP7]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_dynarray_of_nonchar2(%e: index) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<?x?xi32>, %e, %e
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_dynarray_of_char(
// CHECK-SAME: i64 [[TMP0:%.*]]) {
// CHECK-NEXT:    [[TMP2:%.*]] = mul i64 60, [[TMP0]]
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 1, [[TMP2]]
// CHECK-NEXT:    [[TMP4:%.*]] = call ptr @omp_target_alloc(i64 [[TMP3]], i32 0)
// CHECK-NEXT:    [[TMP5:%.*]] = ptrtoint ptr [[TMP4]] to i64
// CHECK-NEXT:    [[TMP6:%.*]] = inttoptr i64 [[TMP5]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP6]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_dynarray_of_char(%e : index) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x?x!fir.char<2,10>>, %e
  omp.target_freemem %device, %1 : i32, i64
  return
}


// CHECK-LABEL: define void @omp_target_allocmem_dynarray_of_char2(
// CHECK-SAME: i64 [[TMP0:%.*]]) {
// CHECK-NEXT:    [[TMP2:%.*]] = mul i64 20, [[TMP0]]
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], [[TMP0]]
// CHECK-NEXT:    [[TMP4:%.*]] = mul i64 1, [[TMP3]]
// CHECK-NEXT:    [[TMP5:%.*]] = call ptr @omp_target_alloc(i64 [[TMP4]], i32 0)
// CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
// CHECK-NEXT:    [[TMP7:%.*]] = inttoptr i64 [[TMP6]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP7]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_dynarray_of_char2(%e : index) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<?x?x!fir.char<2,10>>, %e, %e
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_dynarray_of_dynchar(
// CHECK-SAME: i32 [[TMP0:%.*]], i64 [[TMP1:%.*]]) {
// CHECK-NEXT:    [[TMP3:%.*]] = sext i32 [[TMP0]] to i64
// CHECK-NEXT:    [[TMP4:%.*]] = mul i64 6, [[TMP3]]
// CHECK-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], [[TMP1]]
// CHECK-NEXT:    [[TMP6:%.*]] = mul i64 1, [[TMP5]]
// CHECK-NEXT:    [[TMP7:%.*]] = call ptr @omp_target_alloc(i64 [[TMP6]], i32 0)
// CHECK-NEXT:    [[TMP8:%.*]] = ptrtoint ptr [[TMP7]] to i64
// CHECK-NEXT:    [[TMP9:%.*]] = inttoptr i64 [[TMP8]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP9]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_dynarray_of_dynchar(%l: i32, %e : index) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x?x!fir.char<2,?>>(%l : i32), %e
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_dynarray_of_dynchar2(
// CHECK-SAME: i32 [[TMP0:%.*]], i64 [[TMP1:%.*]]) {
// CHECK-NEXT:    [[TMP3:%.*]] = sext i32 [[TMP0]] to i64
// CHECK-NEXT:    [[TMP4:%.*]] = mul i64 2, [[TMP3]]
// CHECK-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], [[TMP1]]
// CHECK-NEXT:    [[TMP6:%.*]] = mul i64 [[TMP5]], [[TMP1]]
// CHECK-NEXT:    [[TMP7:%.*]] = mul i64 1, [[TMP6]]
// CHECK-NEXT:    [[TMP8:%.*]] = call ptr @omp_target_alloc(i64 [[TMP7]], i32 0)
// CHECK-NEXT:    [[TMP9:%.*]] = ptrtoint ptr [[TMP8]] to i64
// CHECK-NEXT:    [[TMP10:%.*]] = inttoptr i64 [[TMP9]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP10]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_dynarray_of_dynchar2(%l: i32, %e : index) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<?x?x!fir.char<2,?>>(%l : i32), %e, %e
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_array_with_holes_nonchar(
// CHECK-SAME: i64 [[TMP0:%.*]], i64 [[TMP1:%.*]]) {
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 240, [[TMP0]]
// CHECK-NEXT:    [[TMP4:%.*]] = mul i64 [[TMP3]], [[TMP1]]
// CHECK-NEXT:    [[TMP5:%.*]] = mul i64 1, [[TMP4]]
// CHECK-NEXT:    [[TMP6:%.*]] = call ptr @omp_target_alloc(i64 [[TMP5]], i32 0)
// CHECK-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP6]] to i64
// CHECK-NEXT:    [[TMP8:%.*]] = inttoptr i64 [[TMP7]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP8]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_array_with_holes_nonchar(%0 : index, %1 : index) -> () {
  %device = arith.constant 0 : i32
  %2 = omp.target_allocmem %device : i32, !fir.array<4x?x3x?x5xi32>, %0, %1
  omp.target_freemem %device, %2 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_array_with_holes_char(
// CHECK-SAME: i64 [[TMP0:%.*]]) {
// CHECK-NEXT:    [[TMP2:%.*]] = mul i64 240, [[TMP0]]
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 1, [[TMP2]]
// CHECK-NEXT:    [[TMP4:%.*]] = call ptr @omp_target_alloc(i64 [[TMP3]], i32 0)
// CHECK-NEXT:    [[TMP5:%.*]] = ptrtoint ptr [[TMP4]] to i64
// CHECK-NEXT:    [[TMP6:%.*]] = inttoptr i64 [[TMP5]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP6]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_array_with_holes_char(%e: index) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x?x4x!fir.char<2,10>>, %e
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_array_with_holes_dynchar(
// CHECK-SAME: i64 [[TMP0:%.*]], i64 [[TMP1:%.*]]) {
// CHECK-NEXT:    [[TMP3:%.*]] = mul i64 24, [[TMP0]]
// CHECK-NEXT:    [[TMP4:%.*]] = mul i64 [[TMP3]], [[TMP1]]
// CHECK-NEXT:    [[TMP5:%.*]] = mul i64 1, [[TMP4]]
// CHECK-NEXT:    [[TMP6:%.*]] = call ptr @omp_target_alloc(i64 [[TMP5]], i32 0)
// CHECK-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP6]] to i64
// CHECK-NEXT:    [[TMP8:%.*]] = inttoptr i64 [[TMP7]] to ptr
// CHECK-NEXT:    call void @omp_target_free(ptr [[TMP8]], i32 0)
// CHECK-NEXT:    ret void
func.func @omp_target_allocmem_array_with_holes_dynchar(%arg0: index, %arg1: index) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x?x4x!fir.char<2,?>>(%arg0 : index), %arg1
  omp.target_freemem %device, %1 : i32, i64
  return
}
