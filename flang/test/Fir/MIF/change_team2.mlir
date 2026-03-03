// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (git@github.com:SiPearl/llvm-project.git 5acb5b14086f4f61f007a6fc14a86e930bd1d247)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST_NESTED_TEAMS"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFEme"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.alloca i32 {bindc_name = "ni", uniq_name = "_QFEni"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFEni"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team_global", uniq_name = "_QFEteam_global"}
    %6:2 = hlfir.declare %5 {uniq_name = "_QFEteam_global"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %7 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    fir.copy %7 to %6#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %8 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team_sub", uniq_name = "_QFEteam_sub"}
    %9:2 = hlfir.declare %8 {uniq_name = "_QFEteam_sub"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %10 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    fir.copy %10 to %9#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %11 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
    %12:2 = hlfir.declare %11 {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %13 = mif.this_image : () -> i32
    hlfir.assign %13 to %2#0 : i32, !fir.ref<i32>
    %14 = mif.num_images : () -> i32
    hlfir.assign %14 to %4#0 : i32, !fir.ref<i32>
    %c1_i32 = arith.constant 1 : i32
    %15 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    mif.form_team team_number %c1_i32 team_var %15 : (i32, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> ()
    %c2_i32 = arith.constant 2 : i32
    %16 = fir.load %2#0 : !fir.ref<i32>
    %17 = arith.remsi %16, %c2_i32 : i32
    %c1_i32_0 = arith.constant 1 : i32
    %18 = arith.addi %17, %c1_i32_0 : i32
    %19 = fir.embox %9#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    mif.form_team team_number %18 team_var %19 : (i32, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> ()
    %20 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    mif.change_team %20 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) {
      %c6_i32 = arith.constant 6 : i32
      %21 = fir.address_of(@_QQclX662561532848a47f9eb7ab919b187d63) : !fir.ref<!fir.char<1,49>>
      %22 = fir.convert %21 : (!fir.ref<!fir.char<1,49>>) -> !fir.ref<i8>
      %c15_i32 = arith.constant 15 : i32
      %23 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %22, %c15_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
      %24 = fir.address_of(@_QQclX44616E73206C61207465616D20676C6F62616C65) : !fir.ref<!fir.char<1,20>>
      %c20 = arith.constant 20 : index
      %25:2 = hlfir.declare %24 typeparams %c20 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX44616E73206C61207465616D20676C6F62616C65"} : (!fir.ref<!fir.char<1,20>>, index) -> (!fir.ref<!fir.char<1,20>>, !fir.ref<!fir.char<1,20>>)
      %26 = fir.convert %25#0 : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<i8>
      %27 = fir.convert %c20 : (index) -> i64
      %28 = fir.call @_FortranAioOutputAscii(%23, %26, %27) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
      %29 = fir.call @_FortranAioEndIoStatement(%23) fastmath<contract> : (!fir.ref<i8>) -> i32
      %30 = fir.load %2#0 : !fir.ref<i32>
      %c2_i32_1 = arith.constant 2 : i32
      %31 = arith.cmpi eq, %30, %c2_i32_1 : i32
      fir.if %31 {
        %33 = fir.load %4#0 : !fir.ref<i32>
        %c0_i32 = arith.constant 0 : i32
        %34 = arith.addi %33, %c0_i32 : i32
        hlfir.assign %34 to %12#0 : i32, !fir.ref<i32>
      } else {
        %c2_i32_2 = arith.constant 2 : i32
        %33 = fir.load %2#0 : !fir.ref<i32>
        %34 = arith.muli %c2_i32_2, %33 : i32
        hlfir.assign %34 to %12#0 : i32, !fir.ref<i32>
      }
      %32 = fir.embox %9#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
      mif.change_team %32 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) {
        %c6_i32_2 = arith.constant 6 : i32
        %33 = fir.address_of(@_QQclX662561532848a47f9eb7ab919b187d63) : !fir.ref<!fir.char<1,49>>
        %34 = fir.convert %33 : (!fir.ref<!fir.char<1,49>>) -> !fir.ref<i8>
        %c23_i32 = arith.constant 23 : i32
        %35 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32_2, %34, %c23_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
        %36 = fir.address_of(@_QQclX327beca4c668cfd5c48dd70a28c39c6d) : !fir.ref<!fir.char<1,36>>
        %c36 = arith.constant 36 : index
        %37:2 = hlfir.declare %36 typeparams %c36 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX327beca4c668cfd5c48dd70a28c39c6d"} : (!fir.ref<!fir.char<1,36>>, index) -> (!fir.ref<!fir.char<1,36>>, !fir.ref<!fir.char<1,36>>)
        %38 = fir.convert %37#0 : (!fir.ref<!fir.char<1,36>>) -> !fir.ref<i8>
        %39 = fir.convert %c36 : (index) -> i64
        %40 = fir.call @_FortranAioOutputAscii(%35, %38, %39) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
        %41 = fir.call @_FortranAioEndIoStatement(%35) fastmath<contract> : (!fir.ref<i8>) -> i32
        mif.end_team : () -> ()
      }
      mif.end_team : () -> ()
    }
    return
  }
}
// CHECK: fir.call @_QMprifPprif_change_team(%[[VAL_3:.*]], %[[VAL_1:.*]], %[[VAL_2:.*]], %[[VAL_2:.*]]) : (!fir.box<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// Just to show that we can handle operations with body inside change_team
// CHECK: fir.if %[[COND:.*]] {
// CHECK: } else {
// CHECK: }

// We can handle also another change_team operation inside.
// CHECK: %[[VAL_4:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_5:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_6:.*]] = fir.convert %[[TEAM2:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_change_team(%[[VAL_6]], %[[VAL_4]], %[[VAL_5]], %[[VAL_5]]) : (!fir.box<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_7:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_8:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_end_team(%[[VAL_7]], %[[VAL_8]], %[[VAL_8]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_9:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_10:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_end_team(%[[VAL_9]], %[[VAL_10]], %[[VAL_10]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
