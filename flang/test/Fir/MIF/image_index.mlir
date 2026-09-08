// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (git@github.com:SiPearl/llvm-project.git d31a4730513391710d91c5ad33bb8ea3d68db3cb)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
// CHECK-LABEL: func.func @_QQmain
  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
   %0 = fir.alloca !fir.array<2xi64>
    %1 = fir.alloca !fir.array<3xi64>
    %2 = fir.dummy_scope : !fir.dscope
    %3 = fir.address_of(@_QFEa) : !fir.ref<i32>
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %4 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %4 : !fir.ref<i64>
    %c3_i64 = arith.constant 3 : i64
    %c1 = arith.constant 1 : index
    %5 = fir.coordinate_of %1, %c1 : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
    fir.store %c3_i64 to %5 : !fir.ref<i64>
    %c1_i64_0 = arith.constant 1 : i64
    %c2 = arith.constant 2 : index
    %6 = fir.coordinate_of %1, %c2 : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64_0 to %6 : !fir.ref<i64>
    %7 = fir.embox %1 : (!fir.ref<!fir.array<3xi64>>) -> !fir.box<!fir.array<3xi64>>
    %c1_i64_1 = arith.constant 1 : i64
    %c0_2 = arith.constant 0 : index
    %8 = fir.coordinate_of %0, %c0_2 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64_1 to %8 : !fir.ref<i64>
    %c3_i64_3 = arith.constant 3 : i64
    %c1_4 = arith.constant 1 : index
    %9 = fir.coordinate_of %0, %c1_4 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c3_i64_3 to %9 : !fir.ref<i64>
    %10 = fir.embox %0 : (!fir.ref<!fir.array<2xi64>>) -> !fir.box<!fir.array<2xi64>>
    mif.alloc_coarray %3 lcobounds %7 ucobounds %10 {uniq_name = "_QFEa"} : (!fir.ref<i32>, !fir.box<!fir.array<3xi64>>, !fir.box<!fir.array<2xi64>>) -> ()
    %11:2 = hlfir.declare %3 {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %12 = fir.alloca i32 {bindc_name = "idx", uniq_name = "_QFEidx"}
    %13:2 = hlfir.declare %12 {uniq_name = "_QFEidx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %14 = fir.address_of(@_QFEsub) : !fir.ref<!fir.array<3xi32>>
    %c3 = arith.constant 3 : index
    %15 = fir.shape %c3 : (index) -> !fir.shape<1>
    %16:2 = hlfir.declare %14(%15) {uniq_name = "_QFEsub"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
    %17 = fir.address_of(@_QFEsub2) : !fir.ref<!fir.array<3xi64>>
    %c3_5 = arith.constant 3 : index
    %18 = fir.shape %c3_5 : (index) -> !fir.shape<1>
    %19:2 = hlfir.declare %17(%18) {uniq_name = "_QFEsub2"} : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)
    %20 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team", uniq_name = "_QFEteam"}
    %21:2 = hlfir.declare %20 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %22 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    fir.copy %22 to %21#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %23 = fir.alloca i32 {bindc_name = "team_number", uniq_name = "_QFEteam_number"}
    %24:2 = hlfir.declare %23 {uniq_name = "_QFEteam_number"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %25 = fir.embox %11#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:3>
    %26 = fir.shape %c3 : (index) -> !fir.shape<1>
    %27 = fir.embox %16#0(%26) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
    %28 = mif.image_index coarray %25 sub %27 : (!fir.box<i32, corank:3>, !fir.box<!fir.array<3xi32>>) -> i32
    hlfir.assign %28 to %13#0 : i32, !fir.ref<i32>
    %29 = fir.embox %11#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:3>
    %30 = fir.shape %c3_5 : (index) -> !fir.shape<1>
    %31 = fir.embox %19#0(%30) : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi64>>
    %32 = mif.image_index coarray %29 sub %31 : (!fir.box<i32, corank:3>, !fir.box<!fir.array<3xi64>>) -> i32
    hlfir.assign %32 to %13#0 : i32, !fir.ref<i32>
    %33 = fir.embox %11#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:3>
    %34 = fir.shape %c3 : (index) -> !fir.shape<1>
    %35 = fir.embox %16#0(%34) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
    %36 = mif.image_index coarray %33 sub %35 team %21#0 : (!fir.box<i32, corank:3>, !fir.box<!fir.array<3xi32>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> i32
    hlfir.assign %36 to %13#0 : i32, !fir.ref<i32>
    %37 = fir.embox %11#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:3>
    %38 = fir.shape %c3 : (index) -> !fir.shape<1>
    %39 = fir.embox %16#0(%38) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
    %40 = fir.load %24#0 : !fir.ref<i32>
    %41 = mif.image_index coarray %37 sub %39 team_number %40 : (!fir.box<i32, corank:3>, !fir.box<!fir.array<3xi32>>, i32) -> i32
    hlfir.assign %41 to %13#0 : i32, !fir.ref<i32>
    return
  }
}

// CHECK: fir.call @_QMprifPprif_image_index(
// CHECK: fir.call @_QMprifPprif_image_index(
// CHECK: fir.call @_QMprifPprif_image_index_with_team(
// CHECK: fir.call @_QMprifPprif_image_index_with_team_number(
