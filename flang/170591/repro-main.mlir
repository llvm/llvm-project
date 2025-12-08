module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (https://github.com/eugeneepshteyn/llvm-project.git b70be3dc14c1f54eaae33418ced28a3473ab7d70)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QPkohb_exit(%arg0: !fir.ref<i32> {fir.bindc_name = "status"}) -> i32 {
    %c6_i32 = arith.constant 6 : i32
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFkohb_exitEi"}
    %2 = fir.declare %1 {uniq_name = "_QFkohb_exitEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.alloca i32 {bindc_name = "kohb_exit", uniq_name = "_QFkohb_exitEkohb_exit"}
    %4 = fir.declare %3 {uniq_name = "_QFkohb_exitEkohb_exit"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %5 = fir.declare %arg0 dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFkohb_exitEstatus"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %6 = fir.load %5 : !fir.ref<i32>
    %7 = arith.cmpi sgt, %6, %c0_i32 : i32
    fir.if %7 {
      %14 = fir.load %5 : !fir.ref<i32>
      fir.call @_FortranAExit(%14) fastmath<contract> : (i32) -> ()
    }
    fir.store %c0_i32 to %4 : !fir.ref<i32>
    %8 = fir.convert %c1_i32 : (i32) -> index
    %9 = fir.load %5 : !fir.ref<i32>
    %10 = fir.convert %9 : (i32) -> index
    %11 = fir.convert %8 : (index) -> i32
    %12 = fir.do_loop %arg1 = %8 to %10 step %c1 iter_args(%arg2 = %11) -> (i32) {
      fir.store %arg2 to %2 : !fir.ref<i32>
      %14 = fir.address_of(@_QQclX28412C493029) : !fir.ref<!fir.char<1,6>>
      %15 = fir.declare %14 typeparams %c6 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX28412C493029"} : (!fir.ref<!fir.char<1,6>>, index) -> !fir.ref<!fir.char<1,6>>
      %16 = fir.convert %15 : (!fir.ref<!fir.char<1,6>>) -> !fir.ref<i8>
      %17 = fir.convert %c6 : (index) -> i64
      %18 = fir.zero_bits !fir.box<none>
      %19 = fir.address_of(@_QQclXa224fc2bcd1182277fdd6e03fbf16dbd) : !fir.ref<!fir.char<1,76>>
      %20 = fir.convert %19 : (!fir.ref<!fir.char<1,76>>) -> !fir.ref<i8>
      %21 = fir.call @_FortranAioBeginExternalFormattedOutput(%16, %17, %18, %c6_i32, %20, %c6_i32) fastmath<contract> : (!fir.ref<i8>, i64, !fir.box<none>, i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
      %22 = fir.address_of(@_QQclX4B4F48622023) : !fir.ref<!fir.char<1,6>>
      %23 = fir.declare %22 typeparams %c6 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX4B4F48622023"} : (!fir.ref<!fir.char<1,6>>, index) -> !fir.ref<!fir.char<1,6>>
      %24 = fir.convert %23 : (!fir.ref<!fir.char<1,6>>) -> !fir.ref<i8>
      %25 = fir.convert %c6 : (index) -> i64
      %26 = fir.call @_FortranAioOutputAscii(%21, %24, %25) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
      %27 = fir.load %2 : !fir.ref<i32>
      %28 = fir.call @_FortranAioOutputInteger32(%21, %27) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
      %29 = fir.call @_FortranAioEndIoStatement(%21) fastmath<contract> : (!fir.ref<i8>) -> i32
      %30 = fir.convert %c1 : (index) -> i32
      %31 = fir.load %2 : !fir.ref<i32>
      %32 = arith.addi %31, %30 overflow<nsw> : i32
      fir.result %32 : i32
    }
    fir.store %12 to %2 : !fir.ref<i32>
    %13 = fir.load %4 : !fir.ref<i32>
    return %13 : i32
  }
  func.func private @_FortranAExit(i32) attributes {fir.runtime}
  func.func private @_FortranAioBeginExternalFormattedOutput(!fir.ref<i8>, i64, !fir.box<none>, i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
  fir.global linkonce @_QQclX28412C493029 constant : !fir.char<1,6> {
    %0 = fir.string_lit "(A,I0)"(6) : !fir.char<1,6>
    fir.has_value %0 : !fir.char<1,6>
  }
  fir.global linkonce @_QQclXa224fc2bcd1182277fdd6e03fbf16dbd constant : !fir.char<1,76> {
    %0 = fir.string_lit "/home/eepshteyn/src/flang-upstream/llvm-project/flang/170591/repro-main.f90\00"(76) : !fir.char<1,76>
    fir.has_value %0 : !fir.char<1,76>
  }
  func.func private @_FortranAioOutputAscii(!fir.ref<i8>, !fir.ref<i8>, i64) -> i1 attributes {fir.io, fir.runtime}
  fir.global linkonce @_QQclX4B4F48622023 constant : !fir.char<1,6> {
    %0 = fir.string_lit "KOHb #"(6) : !fir.char<1,6>
    fir.has_value %0 : !fir.char<1,6>
  }
  func.func private @_FortranAioOutputInteger32(!fir.ref<i8>, i32) -> i1 attributes {fir.io, fir.runtime}
  func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}
}
