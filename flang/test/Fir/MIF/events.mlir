// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QQmain() attributes {fir.bindc_name = "EVENT_TEST"} {
    %0 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
    %1 = fir.alloca i64
    %2 = fir.alloca !fir.array<1xi64>
    %3 = fir.alloca i32
    %4 = fir.alloca i64
    %5 = fir.alloca i32
    %6 = fir.alloca i64
    %7 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
    %8 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>
    %9 = fir.alloca !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
    %10 = fir.alloca !fir.array<0xi64>
    %11 = fir.alloca !fir.array<1xi64>
    %12 = fir.dummy_scope : !fir.dscope
    %13 = fir.address_of(@_QFEdata_ready) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %14 = fir.coordinate_of %11, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %14 : !fir.ref<i64>
    %15 = fir.embox %11 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
    %16 = fir.embox %10 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
    %17 = fir.zero_bits (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()
    %18 = fir.emboxproc %17 : ((!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()) -> !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
    fir.store %18 to %9 : !fir.ref<!fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>>
    %19 = fir.load %13 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
    %20 = fir.box_elesize %19 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> i64
    fir.store %20 to %6 : !fir.ref<i64>
    %21 = fir.absent !fir.ref<i32>
    %22 = fir.absent !fir.box<!fir.char<1,?>>
    %23 = fir.convert %15 : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
    %24 = fir.convert %16 : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
    %25 = fir.convert %9 : (!fir.ref<!fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>>) -> !fir.ref<none>
    %26 = fir.convert %8 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<none>
    fir.call @_QMprifPprif_allocate_coarray(%23, %24, %6, %25, %26, %7, %21, %22, %22) : (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>, !fir.ref<i64>, !fir.ref<none>, !fir.ref<none>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
    %27 = fir.address_of(@_QFEdata_ready_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
    fir.copy %8 to %27 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
    %28 = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
    %29 = fir.coordinate_of %7, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
    %30 = fir.load %29 : !fir.ref<i64>
    %31 = fir.convert %13 : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> !fir.ref<!fir.box<none>>
    %32 = fir.convert %30 : (i64) -> !fir.llvm_ptr<i8>
    fir.call @_FortranAAllocatableSetBaseAddr(%31, %32) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
    %33 = fir.address_of(@_QQclX746573742E6D6C697200) : !fir.ref<!fir.char<1,10>>
    %c10 = arith.constant 10 : index
    %c13_i32 = arith.constant 13 : i32
    %34 = fir.convert %13 : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> !fir.box<none>
    %35 = fir.convert %33 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
    fir.call @_FortranAInitialize(%34, %35, %c13_i32) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
    %36:2 = hlfir.declare %13 {uniq_name = "_QFEdata_ready"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>)
    %37 = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFEme"}
    %38:2 = hlfir.declare %37 {uniq_name = "_QFEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %39 = fir.absent !fir.ref<none>
    fir.call @_QMprifPprif_this_image_no_coarray(%39, %5) : (!fir.ref<none>, !fir.ref<i32>) -> ()
    %40 = fir.load %5 : !fir.ref<i32>
    hlfir.assign %40 to %38#0 : i32, !fir.ref<i32>
    %41 = fir.load %38#0 : !fir.ref<i32>
    %c2_i32 = arith.constant 2 : i32
    %42 = arith.cmpi eq, %41, %c2_i32 : i32
    fir.if %42 {
      %43 = fir.load %36#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
      %44 = fir.box_addr %43 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
      %45 = hlfir.designate %44   : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
      %46 = fir.embox %45 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>, corank:1>
      %c1_i64_0 = arith.constant 1 : i64
      %47 = fir.absent !fir.ref<i32>
      %48 = fir.absent !fir.box<!fir.char<1,?>>
      %c0_i64 = arith.constant 0 : i64
      fir.store %c0_i64 to %4 : !fir.ref<i64>
      %49 = fir.address_of(@_QFEdata_ready_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
      %c0_1 = arith.constant 0 : index
      %50 = fir.coordinate_of %2, %c0_1 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
      fir.store %c1_i64_0 to %50 : !fir.ref<i64>
      %51 = fir.embox %2 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
      %52 = fir.absent !fir.ref<i32>
      %53 = fir.convert %49 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<none>
      %54 = fir.convert %51 : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
      fir.call @_QMprifPprif_initial_team_index(%53, %54, %3, %52) : (!fir.ref<none>, !fir.box<!fir.array<?xi64>>, !fir.ref<i32>, !fir.ref<i32>) -> ()
      %55 = fir.convert %49 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<none>
      fir.call @_QMprifPprif_event_post(%3, %55, %4, %47, %48, %48) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
    } else {
      %43 = fir.load %38#0 : !fir.ref<i32>
      %c1_i32 = arith.constant 1 : i32
      %44 = arith.cmpi eq, %43, %c1_i32 : i32
      fir.if %44 {
        %c1_i32_0 = arith.constant 1 : i32
        %45 = fir.load %36#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
        %46 = fir.box_addr %45 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
        %47 = fir.convert %c1_i32_0 : (i32) -> i64
        fir.store %47 to %1 : !fir.ref<i64>
        %48 = fir.absent !fir.ref<i32>
        %49 = fir.absent !fir.box<!fir.char<1,?>>
        %50 = fir.convert %46 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i64
        %51 = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
        %52 = fir.coordinate_of %0, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
        fir.store %50 to %52 : !fir.ref<i64>
        fir.call @_QMprifPprif_event_wait(%0, %1, %48, %49, %49) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
      }
    }
    return
}

// CHECK: fir.call @_QMprifPprif_initial_team_index(
// CHECK: fir.call @_QMprifPprif_event_post(%[[IMAGE_INDEX:.*]], %[[EVENT_PTR:.*]], %[[OFFSET:.*]], %[[STAT:.*]], %[[ERRMSG:.*]], %[[ERRMSG2:.*]]) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()


// CHECK: fir.call @_QMprifPprif_event_wait(%[[EVENT_PTR:.*]], %[[UNTIL_COUNT:.*]], %[[STAT:.*]], %[[ERRMSG:.*]], %[[ERRMSG2:.*]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
