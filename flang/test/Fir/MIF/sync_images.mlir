// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:SiPearl/llvm-project.git 666e4313ebc03587f27774139ad8f780bac15c3e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST_SYNC_IMAGES"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QFEerror_message) : !fir.ref<!fir.char<1,128>>
    %c128 = arith.constant 128 : index
    %2:2 = hlfir.declare %1 typeparams %c128 {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
    %3 = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFEme"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5 = fir.alloca i32 {bindc_name = "sync_status", uniq_name = "_QFEsync_status"}
    %6:2 = hlfir.declare %5 {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %7 = fir.embox %2#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
    mif.sync_images stat %6#0 errmsg %7 : (!fir.ref<i32>, !fir.box<!fir.char<1,128>>) -> ()
    %8 = fir.embox %2#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
    %9 = fir.embox %4#0 : (!fir.ref<i32>) -> !fir.box<i32>
    mif.sync_images image_set %9 stat %6#0 errmsg %8 : (!fir.box<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,128>>) -> ()
    %10 = fir.embox %2#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
    %11 = fir.address_of(@_QQro.1xi4.0) : !fir.ref<!fir.array<1xi32>>
    %c1 = arith.constant 1 : index
    %12 = fir.shape %c1 : (index) -> !fir.shape<1>
    %13:2 = hlfir.declare %11(%12) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.1xi4.0"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
    %14 = fir.shape %c1 : (index) -> !fir.shape<1>
    %15 = fir.embox %13#0(%14) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
    mif.sync_images image_set %15 stat %6#0 errmsg %10 : (!fir.box<!fir.array<1xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,128>>) -> ()
    mif.sync_images : () -> ()
    %16 = fir.embox %4#0 : (!fir.ref<i32>) -> !fir.box<i32>
    mif.sync_images image_set %16 : (!fir.box<i32>) -> ()
    %17 = fir.address_of(@_QQro.1xi4.0) : !fir.ref<!fir.array<1xi32>>
    %c1_0 = arith.constant 1 : index
    %18 = fir.shape %c1_0 : (index) -> !fir.shape<1>
    %19:2 = hlfir.declare %17(%18) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.1xi4.0"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
    %20 = fir.shape %c1_0 : (index) -> !fir.shape<1>
    %21 = fir.embox %19#0(%20) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
    mif.sync_images image_set %21 : (!fir.box<!fir.array<1xi32>>) -> ()
    return
  }
  fir.global internal @_QFEerror_message : !fir.char<1,128> {
    %0 = fir.zero_bits !fir.char<1,128>
    fir.has_value %0 : !fir.char<1,128>
  }
}
  
  // CHECK: %[[ERRMSG:.*]]:2 = hlfir.declare %[[E:.*]] typeparams %[[C_128:.*]] {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
  // CHECK: %[[ME:.*]]:2 = hlfir.declare %[[M:.*]] {uniq_name = "_QFEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  // CHECK: %[[STAT:.*]]:2 = hlfir.declare %[[S:.*]] {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  // CHECK: %[[VAL_1:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  // CHECK: %[[VAL_2:.*]] = fir.absent !fir.box<!fir.array<?xi32>> 
  // CHECK: %[[VAL_3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[VAL_4:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
  // CHECK: fir.call @_QMprifPprif_sync_images(%[[VAL_2]], %[[STAT]]#0, %[[VAL_4]], %[[VAL_3]]) : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

  // CHECK: %[[VAL_5:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  // CHECK: %[[VAL_6:.*]] = fir.embox %[[ME]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  // CHECK: %[[VAL_7:.*]] = fir.rebox %[[VAL_6]](%[[SHAPE:.*]]) : (!fir.box<i32>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  // CHECK: %[[VAL_8:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>> 
  // CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
  // CHECK: fir.call @_QMprifPprif_sync_images(%[[VAL_9]], %[[STAT]]#0, %[[VAL_10]], %[[VAL_8]]) : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

  // CHECK: %[[VAL_11:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  // CHECK: %[[VAL_12:.*]] = fir.embox %[[IMG_SET:.*]]#0(%[[SHAPE_1:.*]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  // CHECK: %[[VAL_13:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>>
  // CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
  // CHECK: fir.call @_QMprifPprif_sync_images(%[[VAL_14]], %[[STAT]]#0, %[[VAL_15]], %[[VAL_13]]) : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  
  // CHECK: %[[VAL_16:.*]] = fir.absent !fir.box<!fir.array<?xi32>> 
  // CHECK: %[[VAL_17:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[VAL_18:.*]] = fir.absent !fir.ref<i32>
  // CHECK: fir.call @_QMprifPprif_sync_images(%[[VAL_16]], %[[VAL_18]], %[[VAL_17]], %[[VAL_17]]) : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  
  // CHECK: %[[VAL_19:.*]] = fir.embox %[[ME]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  // CHECK: %[[VAL_20:.*]] = fir.rebox %[[VAL_19]](%[[SHAPE_2:.*]]) : (!fir.box<i32>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  // CHECK: %[[VAL_21:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[VAL_22:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>> 
  // CHECK: fir.call @_QMprifPprif_sync_images(%[[VAL_23]], %[[VAL_22]], %[[VAL_21]], %[[VAL_21]]) : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  
  // CHECK: %[[VAL_24:.*]] = fir.embox %[[IMG_SET:.*]]#0(%[[SHAPE_3:.*]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  // CHECK: %[[VAL_25:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  // CHECK: %[[VAL_26:.*]] = fir.absent !fir.ref<i32>
  // CHECK: %[[VAL_27:.*]] = fir.convert %[[VAL_24]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>>
  // CHECK: fir.call @_QMprifPprif_sync_images(%[[VAL_27]], %[[VAL_26]], %[[VAL_25]], %[[VAL_25]]) : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
