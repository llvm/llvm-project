// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QQmain() attributes {fir.bindc_name = "P"} {
  %0 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  %1 = fir.alloca !fir.array<1xi64>
  %2 = fir.alloca i32
  %3 = fir.alloca i64
  %4 = fir.alloca i64
  %5 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  %6 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
  %7 = fir.alloca !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
  %8 = fir.alloca !fir.array<0xi64>
  %9 = fir.alloca !fir.array<1xi64>
  %10 = fir.dummy_scope : !fir.dscope
  %11 = fir.address_of(@_QFEe) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %12:2 = hlfir.declare %11 uniq_name("_QFEe") fortran_attrs<allocatable> : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>)
  %13 = fir.absent !fir.box<none>
  %14 = fir.type_desc !fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>
  %c0_i32 = arith.constant 0 : i32
  %c0_i32_0 = arith.constant 0 : i32
  %15 = fir.convert %12#0 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> !fir.ref<!fir.box<none>>
  %16 = fir.convert %14 : (!fir.tdesc<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.ref<none>
  fir.call @_FortranAAllocatableInitDerivedForAllocate(%15, %16, %c0_i32, %c0_i32_0) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %17 = fir.coordinate_of %9, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64 to %17 : !fir.ref<i64>
  %18 = fir.embox %9 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %19 = fir.embox %8 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  %20 = fir.zero_bits (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()
  %21 = fir.emboxproc %20 : ((!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()) -> !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
  fir.store %21 to %7 : !fir.ref<!fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>>
  %22 = fir.load %12#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %23 = fir.box_elesize %22 : (!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> i64
  fir.store %23 to %4 : !fir.ref<i64>
  %24 = fir.absent !fir.ref<i32>
  %25 = fir.absent !fir.box<!fir.char<1,?>>
  %26 = fir.convert %18 : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
  %27 = fir.convert %19 : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
  %28 = fir.convert %7 : (!fir.ref<!fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>>) -> !fir.ref<none>
  %29 = fir.convert %6 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<none>
  %30 = fir.convert %13 : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
  fir.call @_QMprifPprif_allocate_coarray(%26, %27, %4, %28, %29, %5, %24, %30, %25) : (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>, !fir.ref<i64>, !fir.ref<none>, !fir.ref<none>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  %31 = fir.address_of(@_QFEe_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
  fir.copy %6 to %31 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
  %32 = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  %33 = fir.coordinate_of %5, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
  %34 = fir.load %33 : !fir.ref<i64>
  %35 = fir.convert %12#0 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> !fir.ref<!fir.box<none>>
  %36 = fir.convert %34 : (i64) -> !fir.llvm_ptr<i8>
  fir.call @_FortranAAllocatableSetBaseAddr(%35, %36) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
  %37 = fir.address_of(@_QQclX746573742E6D6C697200) : !fir.ref<!fir.char<1,10>>
  %c10 = arith.constant 10 : index
  %c21_i32 = arith.constant 21 : i32
  %38 = fir.convert %12#0 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> !fir.box<none>
  %39 = fir.convert %37 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  fir.call @_FortranAInitialize(%38, %39, %c21_i32) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
  %40 = fir.absent !fir.box<!fir.char<1,?>>
  %41 = fir.absent !fir.ref<i32>
  %42 = fir.convert %13 : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
  fir.call @_QMprifPprif_sync_all(%41, %42, %40) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  %43 = fir.load %12#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %c1_i64_1 = arith.constant 1 : i64
  %44 = fir.absent !fir.ref<i32>
  %45 = fir.absent !fir.box<!fir.char<1,?>>
  %c0_i64 = arith.constant 0 : i64
  fir.store %c0_i64 to %3 : !fir.ref<i64>
  %46 = fir.address_of(@_QFEe_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
  %c0_2 = arith.constant 0 : index
  %47 = fir.coordinate_of %1, %c0_2 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64_1 to %47 : !fir.ref<i64>
  %48 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %49 = fir.absent !fir.ref<i32>
  %50 = fir.convert %46 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<none>
  %51 = fir.convert %48 : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
  fir.call @_QMprifPprif_initial_team_index(%50, %51, %2, %49) : (!fir.ref<none>, !fir.box<!fir.array<?xi64>>, !fir.ref<i32>, !fir.ref<i32>) -> ()
  %52 = fir.convert %46 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<none>
  fir.call @_QMprifPprif_event_post(%2, %52, %3, %44, %45, %45) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  %53 = fir.load %12#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %54 = fir.absent !fir.ref<i64>
  %55 = fir.absent !fir.ref<i32>
  %56 = fir.absent !fir.box<!fir.char<1,?>>
  %57 = fir.box_addr %53 : (!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %58 = fir.convert %57 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i64
  %59 = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  %60 = fir.coordinate_of %0, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
  fir.store %58 to %60 : !fir.ref<i64>
  fir.call @_QMprifPprif_event_wait(%0, %54, %55, %56, %56) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  return
}

// CHECK: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
// CHECK: %[[VAL_2:.*]] = fir.alloca i32
// CHECK: %[[VAL_3:.*]] = fir.alloca i64
// CHECK: %[[VAL_4:.*]] = fir.alloca i64
// CHECK: %[[VAL_5:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_6:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{{.*}}>
// CHECK: %[[VAL_7:.*]] = fir.alloca !fir.boxproc<{{.*}}>
// CHECK: %[[VAL_8:.*]] = fir.alloca !fir.array<0xi64>
// CHECK: %[[VAL_9:.*]] = fir.alloca !fir.array<1xi64>
// CHECK: %[[VAL_10:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_11:.*]] = fir.address_of(@_QFEe) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK: %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] uniq_name("_QFEe") fortran_attrs<allocatable> : {{.*}}
// CHECK: %[[VAL_13:.*]] = fir.absent !fir.box<none>
// CHECK: %[[VAL_14:.*]] = fir.type_desc !fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>
// CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[C0_I32_0:.*]] = arith.constant 0 : i32
// CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_12]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : {{.*}} -> !fir.ref<none>
// CHECK: fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[VAL_15]], %[[VAL_16]], %[[C0_I32]], %[[C0_I32_0]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
// CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_17:.*]] = fir.coordinate_of %[[VAL_9]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK: fir.store %[[C1_I64]] to %[[VAL_17]] : !fir.ref<i64>
// CHECK: %[[VAL_18:.*]] = fir.embox %[[VAL_9]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK: %[[VAL_19:.*]] = fir.embox %[[VAL_8]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
// CHECK: %[[VAL_20:.*]] = fir.zero_bits {{.*}} -> ()
// CHECK: %[[VAL_21:.*]] = fir.emboxproc %[[VAL_20]] : {{.*}}
// CHECK: fir.store %[[VAL_21]] to %[[VAL_7]] : {{.*}}
// CHECK: %[[VAL_22:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK: %[[VAL_23:.*]] = fir.box_elesize %[[VAL_22]] : ({{.*}}) -> i64
// CHECK: fir.store %[[VAL_23]] to %[[VAL_4]] : !fir.ref<i64>
// CHECK: %[[VAL_24:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_25:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_26:.*]] = fir.convert %[[VAL_18]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: %[[VAL_27:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: %[[VAL_28:.*]] = fir.convert %[[VAL_7]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_6]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_13]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_allocate_coarray(%[[VAL_26]], %[[VAL_27]], %[[VAL_4]], %[[VAL_28]], %[[VAL_29]], %[[VAL_5]], %[[VAL_24]], %[[VAL_30]], %[[VAL_25]]) : {{.*}} -> ()
// CHECK: %[[VAL_31:.*]] = fir.address_of(@_QFEe_coarray_handle) : {{.*}}
// CHECK: fir.copy %[[VAL_6]] to %[[VAL_31]] : {{.*}}
// CHECK: %[[VAL_32:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_33:.*]] = fir.coordinate_of %[[VAL_5]], __address : {{.*}} -> !fir.ref<i64>
// CHECK: %[[VAL_34:.*]] = fir.load %[[VAL_33]] : !fir.ref<i64>
// CHECK: %[[VAL_35:.*]] = fir.convert %[[VAL_12]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_36:.*]] = fir.convert %[[VAL_34]] : (i64) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranAAllocatableSetBaseAddr(%[[VAL_35]], %[[VAL_36]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK: %[[VAL_37:.*]] = fir.address_of(@_QQclX746573742E6D6C697200) : !fir.ref<!fir.char<1,10>>
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK: %[[C21_I32:.*]] = arith.constant 21 : i32
// CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_12]]#0 : {{.*}} -> !fir.box<none>
// CHECK: %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
// CHECK: fir.call @_FortranAInitialize(%[[VAL_38]], %[[VAL_39]], %[[C21_I32]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK: %[[VAL_40:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_41:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_42:.*]] = fir.convert %[[VAL_13]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_all(%[[VAL_41]], %[[VAL_42]], %[[VAL_40]]) : {{.*}} -> ()
// CHECK: %[[VAL_43:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK: %[[C1_I64_1:.*]] = arith.constant 1 : i64
// CHECK: %[[VAL_44:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_45:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK: fir.store %[[C0_I64]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK: %[[VAL_46:.*]] = fir.address_of(@_QFEe_coarray_handle) : {{.*}}
// CHECK: %[[C0_2:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_47:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0_2]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK: fir.store %[[C1_I64_1]] to %[[VAL_47]] : !fir.ref<i64>
// CHECK: %[[VAL_48:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK: %[[VAL_49:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_50:.*]] = fir.convert %[[VAL_46]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_51:.*]] = fir.convert %[[VAL_48]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: fir.call @_QMprifPprif_initial_team_index(%[[VAL_50]], %[[VAL_51]], %[[VAL_2]], %[[VAL_49]]) : {{.*}} -> ()
// CHECK: %[[VAL_52:.*]] = fir.convert %[[VAL_46]] : {{.*}} -> !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_event_post(%[[VAL_2]], %[[VAL_52]], %[[VAL_3]], %[[VAL_44]], %[[VAL_45]], %[[VAL_45]]) : {{.*}} -> ()
// CHECK: %[[VAL_53:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK: %[[VAL_54:.*]] = fir.absent !fir.ref<i64>
// CHECK: %[[VAL_55:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_56:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_57:.*]] = fir.box_addr %[[VAL_53]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_59:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_60:.*]] = fir.coordinate_of %[[VAL_0]], __address : {{.*}} -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_58]] to %[[VAL_60]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_wait(%[[VAL_0]], %[[VAL_54]], %[[VAL_55]], %[[VAL_56]], %[[VAL_56]]) : {{.*}} -> ()
// CHECK: return
