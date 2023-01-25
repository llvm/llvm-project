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

// Test emboxing of an array element from an unlimited polymorphic array.

func.func @_QMunlimitedPsub1(%arg0: !fir.class<!fir.array<?xnone>> {fir.bindc_name = "a"}) {
  %c1_i64 = arith.constant 1 : i64
  %c1_i64_0 = arith.constant 1 : i64
  %0 = arith.subi %c1_i64, %c1_i64_0 : i64
  %1 = fir.coordinate_of %arg0, %0 : (!fir.class<!fir.array<?xnone>>, i64) -> !fir.ref<none>
  %3 = fir.embox %1 source_box %arg0 : (!fir.ref<none>, !fir.class<!fir.array<?xnone>>) -> !fir.class<none>
  fir.select_type %3 : !fir.class<none> [#fir.type_is<i32>, ^bb1, unit, ^bb2]
^bb1:
  %4 = fir.box_addr %3 : (!fir.class<none>) -> !fir.ref<i32>
  %c10_i32 = arith.constant 10 : i32
  fir.store %c10_i32 to %4 : !fir.ref<i32>
  cf.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  return
}

// CHECK-LABEL: define void @_QMunlimitedPsub1(
// CHECK-SAME: ptr %[[ARRAY:.*]]) {
// CHECK: %[[BOX:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
// CHECK: %{{.}} = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ARRAY]], i32 0, i32 7, i32 0, i32 2
// CHECK: %[[TYPE_DESC_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ARRAY]], i32 0, i32 8
// CHECK: %[[TYPE_DESC:.*]] = load ptr, ptr %[[TYPE_DESC_GEP]]
// CHECK: %[[ELE_SIZE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ARRAY]], i32 0, i32 1
// CHECK: %[[ELE_SIZE:.*]] = load i64, ptr %[[ELE_SIZE_GEP]]
// CHECK: %[[TYPE_CODE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ARRAY]], i32 0, i32 4
// CHECK: %[[TYPE_CODE:.*]] = load i32, ptr %[[TYPE_CODE_GEP]]
// CHECK: %{{.*}} = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } undef, i64 %[[ELE_SIZE]], 1
// CHECK: %[[TYPE_CODE_TRUNC:.*]] = trunc i32 %[[TYPE_CODE]] to i8
// CHECK: %{{.*}} = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %{{.*}}, i8 %[[TYPE_CODE_TRUNC]], 4
// CHECK: %{{.*}} = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %{{.*}}, ptr %[[TYPE_DESC]], 7
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %24, ptr %[[BOX]]


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

fir.global linkonce_odr @_QMmod1E.dt.t.2 constant target : !fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,__padding0:!fir.array<6xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,__padding0:!fir.array<4xi8>}> {
}

fir.global @_QMmod1Ea target : !fir.type<_QMmod1TtK2{v:!fir.array<2xi32>}> {
  %0 = fir.undefined !fir.type<_QMmod1TtK2{v:!fir.array<2xi32>}>
  %1 = fir.field_index v, !fir.type<_QMmod1TtK2{v:!fir.array<2xi32>}>
  %2 = fir.undefined !fir.array<2xi32>
  %c17_i32 = arith.constant 17 : i32
  %3 = fir.insert_value %2, %c17_i32, [0 : index] : (!fir.array<2xi32>, i32) -> !fir.array<2xi32>
  %c-17_i32 = arith.constant -17 : i32
  %4 = fir.insert_value %3, %c-17_i32, [1 : index] : (!fir.array<2xi32>, i32) -> !fir.array<2xi32>
  %c2 = arith.constant 2 : index
  %5 = fir.insert_value %0, %4, ["v", !fir.type<_QMmod1TtK2{v:!fir.array<2xi32>}>] : (!fir.type<_QMmod1TtK2{v:!fir.array<2xi32>}>, !fir.array<2xi32>) -> !fir.type<_QMmod1TtK2{v:!fir.array<2xi32>}>
  fir.has_value %5 : !fir.type<_QMmod1TtK2{v:!fir.array<2xi32>}>
}
func.func @_QMmod1Psub1(%arg0: !fir.ref<!fir.class<!fir.ptr<none>>> {fir.bindc_name = "target"}) {
  return
}
func.func @_QQmain() {
  %0 = fir.alloca !fir.class<!fir.ptr<none>>
  %1 = fir.address_of(@_QMmod1Ea) : !fir.ref<!fir.type<_QMmod1TtK2{v:!fir.array<2xi32>}>>
  %2 = fir.embox %1 : (!fir.ref<!fir.type<_QMmod1TtK2{v:!fir.array<2xi32>}>>) -> !fir.class<!fir.ptr<none>>
  fir.store %2 to %0 : !fir.ref<!fir.class<!fir.ptr<none>>>
  fir.call @_QMmod1Psub1(%0) fastmath<contract> : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> ()
  return
}

// CHECK-LABEL: define void @_QQmain() {
// CHECK: %[[CLASS_NONE:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
// CHECK: %[[DESC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, i64 1
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr @_QMmod1Ea, i64 ptrtoint (ptr getelementptr (%_QMmod1TtK2, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 42, i8 1, i8 1, ptr @_QMmod1E.dt.t.2, [1 x i64] undef }, ptr %[[CLASS_NONE]], align 8
// CHECK: %[[LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[CLASS_NONE]]
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD]], ptr %[[DESC]]
// CHECK: call void @_QMmod1Psub1(ptr %[[DESC]])

fir.global @_QMmod2Ep : !fir.class<!fir.ptr<none>> {
  %0 = fir.zero_bits !fir.ptr<none>
  %1 = fir.embox %0 : (!fir.ptr<none>) -> !fir.class<!fir.ptr<none>>
  fir.has_value %1 : !fir.class<!fir.ptr<none>>
}
func.func @_QMmod2Pinitp(%arg0: !fir.ref<!fir.class<!fir.ptr<none>>> {fir.bindc_name = "target"}) {
  %0 = fir.address_of(@_QMmod2Ep) : !fir.ref<!fir.class<!fir.ptr<none>>>
  %1 = fir.load %arg0 : !fir.ref<!fir.class<!fir.ptr<none>>>
  %2 = fir.convert %0 : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> !fir.ref<!fir.box<none>>
  %3 = fir.convert %1 : (!fir.class<!fir.ptr<none>>) -> !fir.box<none>
  %4 = fir.call @_FortranAPointerAssociate(%2, %3) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>) -> none
  return
}
func.func private @_FortranAPointerAssociate(!fir.ref<!fir.box<none>>, !fir.box<none>) -> none attributes {fir.runtime}

// CHECK-LABEL: define void @_QMmod2Pinitp(
// CHECK-SAME: ptr %[[ARG0:.*]]) {
// CHECK: %[[ALLOCA_CLASS_NONE:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
// CHECK: %[[LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ARG0]]
// CHECK: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD]], ptr %[[ALLOCA_CLASS_NONE]]
// CHECK: %{{.*}} = call {} @_FortranAPointerAssociate(ptr @_QMmod2Ep, ptr %[[ALLOCA_CLASS_NONE]])
// CHECK: ret void


