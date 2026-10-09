// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QQmain() attributes {fir.bindc_name = "EVENT_TEST"} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %0 = fir.alloca !fir.array<0xi64>
    %1 = fir.alloca !fir.array<1xi64>
    %2 = fir.dummy_scope : !fir.dscope
    %3 = fir.address_of(@_QFEdata_ready) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
    %4 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %4 : !fir.ref<i64>
    %5 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
    %6 = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
    mif.alloc_coarray %3 lcobounds %5 ucobounds %6 {uniq_name = "_QFEdata_ready"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
    %7 = fir.declare %3 {uniq_name = "_QFEdata_ready"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
    %8 = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFEme"}
    %9 = fir.declare %8 {uniq_name = "_QFEme"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %10 = mif.this_image : () -> i32
    fir.store %10 to %9 : !fir.ref<i32>
    %11 = fir.load %9 : !fir.ref<i32>
    %12 = arith.cmpi eq, %11, %c2_i32 : i32
    fir.if %12 {
      %13 = fir.load %7 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
      %14 = fir.box_addr %13 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
      %15 = fir.convert %14 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
      %16 = fir.embox %15 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>, corank:1>
      mif.event_post %16[%c1_i64] : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>, corank:1>, i64) -> ()
    } else {
      %13 = fir.load %9 : !fir.ref<i32>
      %14 = arith.cmpi eq, %13, %c1_i32 : i32
      fir.if %14 {
        %15 = fir.load %7 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
        %16 = fir.box_addr %15 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
        mif.event_wait %16 until_count %c1_i32 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, i32) -> ()
      }
    }
    return
}

// CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:  %[[VAL_1:.*]] = fir.alloca i64
// CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.array<1xi64>
// CHECK:  %[[VAL_3:.*]] = fir.alloca i32
// CHECK:  %[[VAL_4:.*]] = fir.alloca i64
// CHECK:  %[[VAL_5:.*]] = fir.alloca i32
// CHECK:  %[[VAL_6:.*]] = fir.alloca i64
// CHECK:  %[[VAL_7:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{{.*}}>
// CHECK:  %[[VAL_8:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
// CHECK:  %[[VAL_9:.*]] = fir.alloca !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
// CHECK:  %c1_i32 = arith.constant 1 : i32
// CHECK:  %c2_i32 = arith.constant 2 : i32
// CHECK:  %c0 = arith.constant 0 : index
// CHECK:  %c1_i64 = arith.constant 1 : i64
// CHECK:  %[[VAL_10:.*]] = fir.alloca !fir.array<0xi64>
// CHECK:  %[[VAL_11:.*]] = fir.alloca !fir.array<1xi64>
// CHECK:  %[[VAL_12:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:  %[[VAL_13:.*]] = fir.address_of(@_QFEdata_ready) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK:  %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_11]], %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK:  fir.store %c1_i64 to %[[VAL_14]] : !fir.ref<i64>
// CHECK:  %[[VAL_15:.*]] = fir.embox %[[VAL_11]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK:  %[[VAL_16:.*]] = fir.embox %[[VAL_10]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
// CHECK:  %[[VAL_17:.*]] = fir.zero_bits (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:  %[[VAL_18:.*]] = fir.emboxproc %[[VAL_17]] : ((!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()) -> !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
// CHECK:  fir.store %[[VAL_18]] to %[[VAL_9]] : !fir.ref<!fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>>
// CHECK:  %[[VAL_19:.*]] = fir.load %[[VAL_13]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK:  %[[VAL_20:.*]] = fir.box_elesize %[[VAL_19]] : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>) -> i64
// CHECK:  fir.store %[[VAL_20]] to %[[VAL_6]] : !fir.ref<i64>
// CHECK:  %[[VAL_21:.*]] = fir.absent !fir.ref<i32>
// CHECK:  %[[VAL_22:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:  %[[VAL_23:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:  %[[VAL_25:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>>) -> !fir.ref<none>
// CHECK:  %[[VAL_26:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<none>
// CHECK:  fir.call @_QMprifPprif_allocate_coarray(%[[VAL_23]], %[[VAL_24]], %[[VAL_6]], %[[VAL_25]], %[[VAL_26]], %[[VAL_7]], %[[VAL_21]], %[[VAL_22]], %[[VAL_22]]) : (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>, !fir.ref<i64>, !fir.ref<none>, !fir.ref<none>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:  %[[VAL_27:.*]] = fir.address_of(@_QFEdata_ready_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
// CHECK:  fir.copy %[[VAL_8]] to %[[VAL_27]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
// CHECK:  %[[VAL_28:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:  %[[VAL_29:.*]] = fir.coordinate_of %[[VAL_7]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK:  %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.ref<i64>
// CHECK:  %[[VAL_31:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_32:.*]] = fir.convert %[[VAL_30]] : (i64) -> !fir.llvm_ptr<i8>
// CHECK:  fir.call @_FortranAAllocatableSetBaseAddr(%[[VAL_31]], %[[VAL_32]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK:  %[[VAL_33:.*]] = fir.address_of(@_QQclXc6b86460de27409454bc24fe92413d46) : !fir.ref<!fir.char<1,80>>
// CHECK:  %c80 = arith.constant 80 : index
// CHECK:  %c16_i32 = arith.constant 16 : i32
// CHECK:  %[[VAL_34:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>) -> !fir.box<none>
// CHECK:  %[[VAL_35:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,80>>) -> !fir.ref<i8>
// CHECK:  fir.call @_FortranAInitialize(%[[VAL_34]], %[[VAL_35]], %c16_i32) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:  %[[VAL_36:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:  %[[VAL_37:.*]] = fir.absent !fir.ref<i32>
// CHECK:  fir.call @_QMprifPprif_sync_all(%[[VAL_37]], %[[VAL_36]], %[[VAL_36]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:  %[[VAL_38:.*]] = fir.declare %[[VAL_13]] {uniq_name = "_QFEdata_ready"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK:  %[[VAL_39:.*]] = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFEme"}
// CHECK:  %[[VAL_40:.*]] = fir.declare %[[VAL_39]] {uniq_name = "_QFEme"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:  %[[VAL_41:.*]] = fir.absent !fir.ref<none>
// CHECK:  fir.call @_QMprifPprif_this_image_no_coarray(%[[VAL_41]], %[[VAL_5]]) : (!fir.ref<none>, !fir.ref<i32>) -> ()
// CHECK:  %[[VAL_42:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
// CHECK:  fir.store %[[VAL_42]] to %[[VAL_40]] : !fir.ref<i32>
// CHECK:  %[[VAL_43:.*]] = fir.load %[[VAL_40]] : !fir.ref<i32>
// CHECK:  %[[VAL_44:.*]] = arith.cmpi eq, %[[VAL_43]], %c2_i32 : i32
// CHECK:  fir.if %[[VAL_44]] {
// CHECK:    %[[VAL_45:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK:    %[[VAL_46:.*]] = fir.box_addr %[[VAL_45]] : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK:    %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK:    %[[VAL_48:.*]] = fir.embox %[[VAL_47]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>, corank:1>
// CHECK:    %[[VAL_49:.*]] = fir.absent !fir.ref<i32>
// CHECK:    %[[VAL_50:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:    %c0_i64 = arith.constant 0 : i64
// CHECK:    fir.store %c0_i64 to %[[VAL_4]] : !fir.ref<i64>
// CHECK:    %[[VAL_51:.*]] = fir.address_of(@_QFEdata_ready_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
// CHECK:    %c0_0 = arith.constant 0 : index
// CHECK:    %[[VAL_52:.*]] = fir.coordinate_of %[[VAL_2]], %c0_0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK:    fir.store %c1_i64 to %[[VAL_52]] : !fir.ref<i64>
// CHECK:    %[[VAL_53:.*]] = fir.embox %[[VAL_2]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK:    %[[VAL_54:.*]] = fir.absent !fir.ref<i32>
// CHECK:    %[[VAL_55:.*]] = fir.convert %[[VAL_51]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<none>
// CHECK:    %[[VAL_56:.*]] = fir.convert %[[VAL_53]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:    fir.call @_QMprifPprif_initial_team_index(%[[VAL_55]], %[[VAL_56]], %[[VAL_3]], %[[VAL_54]]) : (!fir.ref<none>, !fir.box<!fir.array<?xi64>>, !fir.ref<i32>, !fir.ref<i32>) -> ()
// CHECK:    %[[VAL_57:.*]] = fir.convert %[[VAL_51]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<none>
// CHECK:    fir.call @_QMprifPprif_event_post(%[[VAL_3]], %[[VAL_57]], %[[VAL_4]], %[[VAL_49]], %[[VAL_50]], %[[VAL_50]]) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:  } else {
// CHECK:    %[[VAL_45:.*]] = fir.load %[[VAL_40]] : !fir.ref<i32>
// CHECK:    %[[VAL_46:.*]] = arith.cmpi eq, %[[VAL_45]], %c1_i32 : i32
// CHECK:    fir.if %[[VAL_46]] {
// CHECK:      %[[VAL_47:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK:      %[[VAL_48:.*]] = fir.box_addr %[[VAL_47]] : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK:      %[[VAL_49:.*]] = fir.convert %c1_i32 : (i32) -> i64
// CHECK:      fir.store %[[VAL_49]] to %[[VAL_1]] : !fir.ref<i64>
// CHECK:      %[[VAL_50:.*]] = fir.absent !fir.ref<i32>
// CHECK:      %[[VAL_51:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:      %[[VAL_52:.*]] = fir.convert %[[VAL_48]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK:      %[[VAL_53:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:      %[[VAL_54:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK:      fir.store %[[VAL_52]] to %[[VAL_54]] : !fir.ref<i64>
// CHECK:      fir.call @_QMprifPprif_event_wait(%[[VAL_0]], %[[VAL_1]], %[[VAL_50]], %[[VAL_51]], %[[VAL_51]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:    }

