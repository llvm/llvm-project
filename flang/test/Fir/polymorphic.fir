// RUN: tco %s | FileCheck %s

// Test code gen for unlimited polymorphic type descriptor.

func.func @_QMpolymorphic_testPtest_allocate_unlimited_polymorphic_non_derived() {
  %0 = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = "u", uniq_name = "_QMpolymorphic_testFtest_allocate_unlimited_polymorphic_non_derivedEu"}
  %1 = fir.zero_bits !fir.ptr<none>
  %2 = fir.embox %1 : (!fir.ptr<none>) -> !fir.class<!fir.ptr<none>>
  fir.store %2 to %0 : !fir.ref<!fir.class<!fir.ptr<none>>>
  return
}

// CHECK-LABEL: define void @_QMpolymorphic_testPtest_allocate_unlimited_polymorphic_non_derived() {
// CHECK:   %[[MEM:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
// CHECK:   %[[DESC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, i64 1
// CHECK:   store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr null, i64 0, i32 20180515, i8 0, i8 -1, i8 1, i8 1, ptr null, [1 x i64] undef }, ptr %[[MEM]]
// CHECK:   %[[LOADED:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[MEM]], align 8
// CHECK:   store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOADED]], ptr %[[DESC]]
// CHECK:   ret void
// CHECK: }


// Test rebox of unlimited polymoprhic descriptor

func.func @_QMpolymorphic_testPtest_rebox() {
  %0 = fir.address_of(@_QFEx) : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
  %c-1_i32 = arith.constant -1 : i32
  %9 = fir.address_of(@_QQcl.2E2F64756D6D792E66393000) : !fir.ref<!fir.char<1,12>>
  %10 = fir.convert %9 : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
  %c8_i32 = arith.constant 8 : i32
  %11 = fir.call @_FortranAioBeginExternalListOutput(%c-1_i32, %10, %c8_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  %12 = fir.load %0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
  %c0_1 = arith.constant 0 : index
  %13:3 = fir.box_dims %12, %c0_1 : (!fir.class<!fir.ptr<!fir.array<?xnone>>>, index) -> (index, index, index)
  %14 = fir.shift %13#0 : (index) -> !fir.shift<1>
  %15 = fir.rebox %12(%14) : (!fir.class<!fir.ptr<!fir.array<?xnone>>>, !fir.shift<1>) -> !fir.class<!fir.array<?xnone>>
  %16 = fir.convert %15 : (!fir.class<!fir.array<?xnone>>) -> !fir.box<none>
  %17 = fir.call @_FortranAioOutputDescriptor(%11, %16) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
  %18 = fir.call @_FortranAioEndIoStatement(%11) fastmath<contract> : (!fir.ref<i8>) -> i32
  return
}

// CHECK-LABEL: @_QMpolymorphic_testPtest_rebox
// CHECK: %[[ELE_SIZE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
// CHECK: %[[ELE_SIZE:.*]] = load i64, ptr %[[ELE_SIZE_GEP]]
// CHECK: %[[TYPE_CODE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 4
// CHECK: %[[TYPE_CODE:.*]] = load i32, ptr %[[TYPE_CODE_GEP]]
// CHECK: %{{.*}} = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } undef, i64 %[[ELE_SIZE]], 1
// CHECK: %[[TYPE_CODE_I8:.*]] = trunc i32 %[[TYPE_CODE]] to i8
// CHECK: %{{.*}} = insertvalue { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %{{.*}}, i8 %[[TYPE_CODE_I8]], 4

// Test emboxing to a unlimited polymorphic descriptor

func.func @_QMpolymorphic_testPtest_embox() {
  %0 = fir.address_of(@_QFEx) : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
  %1 = fir.address_of(@_QFEy) : !fir.ref<!fir.array<1xi32>>
  %c1 = arith.constant 1 : index
  %2 = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
  %3 = fir.shape %c1 : (index) -> !fir.shape<1>
  %4 = fir.embox %1(%3) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
  fir.store %4 to %0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
  return
}

// CHECK-LABEL: @_QMpolymorphic_testPtest_embox()
// CHECK: %[[ALLOCA_DESC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } { ptr @_QFEy, i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i32 20180515, i8 1, i8 9, {{.*}}, ptr %[[ALLOCA_DESC]]
// CHECK: %[[LOADED_DESC:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ALLOCA_DESC]], align 8
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[LOADED_DESC]], ptr @_QFEx, align 8


fir.global internal @_QFEx : !fir.class<!fir.ptr<!fir.array<?xnone>>> {
  %0 = fir.zero_bits !fir.ptr<!fir.array<?xnone>>
  %c0 = arith.constant 0 : index
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<?xnone>>, !fir.shape<1>) -> !fir.class<!fir.ptr<!fir.array<?xnone>>>
  fir.has_value %2 : !fir.class<!fir.ptr<!fir.array<?xnone>>>
}

fir.global internal @_QFEy target : !fir.array<1xi32> {
  %0 = fir.undefined !fir.array<1xi32>
  fir.has_value %0 : !fir.array<1xi32>
}

func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
func.func private @_FortranAioOutputDescriptor(!fir.ref<i8>, !fir.box<none>) -> i1 attributes {fir.io, fir.runtime}
func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}
fir.global linkonce @_QQcl.2E2F64756D6D792E66393000 constant : !fir.char<1,12> {
  %0 = fir.string_lit "./dummy.f90\00"(12) : !fir.char<1,12>
  fir.has_value %0 : !fir.char<1,12>
}
