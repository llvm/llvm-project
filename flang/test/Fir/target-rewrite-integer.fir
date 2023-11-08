// RUN: fir-opt --split-input-file --target-rewrite="target=i386-unknown-linux-gnu" %s | FileCheck %s --check-prefixes=I32,ALL
// RUN: fir-opt --split-input-file --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefixes=X64,ALL
// RUN: fir-opt --split-input-file --target-rewrite="target=aarch64-unknown-linux-gnu" %s | FileCheck %s --check-prefixes=AARCH64,ALL
// RUN: fir-opt --split-input-file --target-rewrite="target=powerpc64le-unknown-linux-gnu" %s | FileCheck %s --check-prefixes=PPC,ALL
// RUN: fir-opt --split-input-file --target-rewrite="target=sparc64-unknown-linux-gnu" %s | FileCheck %s --check-prefixes=SPARCV9,ALL
// RUN: fir-opt --split-input-file --target-rewrite="target=sparcv9-sun-solaris2.11" %s | FileCheck %s --check-prefixes=SPARCV9,ALL
// RUN: fir-opt --split-input-file --target-rewrite="target=loongarch64-unknown-linux-gnu" %s | FileCheck %s --check-prefixes=LOONGARCH64,ALL

// -----

// subroutine test_i1(x)
//   logical x
//   print *, x
// end subroutine test_i1

// ALL-LABEL: @_QPtest_i1
// I32: func.func{{.*}}@_FortranAioOutputLogical({{.*}}i1 {llvm.zeroext}) -> (i1 {llvm.zeroext})
// X64: func.func{{.*}}@_FortranAioOutputLogical({{.*}}i1 {llvm.zeroext}) -> (i1 {llvm.zeroext})
// AARCH64: func.func{{.*}}@_FortranAioOutputLogical({{.*}}i1 {llvm.zeroext}) -> (i1 {llvm.zeroext})
// PPC: func.func{{.*}}@_FortranAioOutputLogical({{.*}}i1 {llvm.zeroext}) -> (i1 {llvm.zeroext})
// SPARCV9: func.func{{.*}}@_FortranAioOutputLogical({{.*}}i1 {llvm.zeroext}) -> (i1 {llvm.zeroext})
// LOONGARCH64: func.func{{.*}}@_FortranAioOutputLogical({{.*}}i1 {llvm.zeroext}) -> (i1 {llvm.zeroext})
func.func @_QPtest_i1(%arg0: !fir.ref<!fir.logical<4>> {fir.bindc_name = "x"}) {
  %c3_i32 = arith.constant 3 : i32
  %c-1_i32 = arith.constant -1 : i32
  %0 = fir.address_of(@_QQclX2E2F746573742E66393000) : !fir.ref<!fir.char<1,11>>
  %1 = fir.convert %0 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
  %2 = fir.call @_FortranAioBeginExternalListOutput(%c-1_i32, %1, %c3_i32) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  %3 = fir.load %arg0 : !fir.ref<!fir.logical<4>>
  %4 = fir.convert %3 : (!fir.logical<4>) -> i1
  %5 = fir.call @_FortranAioOutputLogical(%2, %4) : (!fir.ref<i8>, i1) -> i1
  %6 = fir.call @_FortranAioEndIoStatement(%2) : (!fir.ref<i8>) -> i32
  return
}
func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
fir.global linkonce @_QQclX2E2F746573742E66393000 constant : !fir.char<1,11> {
  %0 = fir.string_lit "./test.f90\00"(11) : !fir.char<1,11>
  fir.has_value %0 : !fir.char<1,11>
}
func.func private @_FortranAioOutputLogical(!fir.ref<i8>, i1) -> i1 attributes {fir.io, fir.runtime}
func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}

// -----

// Manually created test with 'si1' argument/return type.
// Flang does not use 'si1' type currently.

// ALL-LABEL: @_QPtest_si1
// I32: func.func{{.*}}@_SomeFunc_si1(si1 {llvm.signext}) -> (si1 {llvm.signext})
// X64: func.func{{.*}}@_SomeFunc_si1(si1 {llvm.signext}) -> (si1 {llvm.signext})
// AARCH64: func.func{{.*}}@_SomeFunc_si1(si1 {llvm.signext}) -> (si1 {llvm.signext})
// PPC: func.func{{.*}}@_SomeFunc_si1(si1 {llvm.signext}) -> (si1 {llvm.signext})
// SPARCV9: func.func{{.*}}@_SomeFunc_si1(si1 {llvm.signext}) -> (si1 {llvm.signext})
// LOONGARCH64: func.func{{.*}}@_SomeFunc_si1(si1 {llvm.signext}) -> (si1 {llvm.signext})
func.func @_QPtest_si1(%arg0: !fir.ref<!fir.logical<4>> {fir.bindc_name = "x"}) {
  %0 = fir.load %arg0 : !fir.ref<!fir.logical<4>>
  %1 = fir.convert %0 : (!fir.logical<4>) -> si1
  %2 = fir.call @_SomeFunc_si1(%1) : (si1) -> si1
  return
}
func.func private @_SomeFunc_si1(si1) -> si1 attributes {fir.runtime}

// -----

// Manually created test with 'ui1' argument/return type.
// Flang does not use 'ui1' type currently.

// ALL-LABEL: @_QPtest_ui1
// I32: func.func{{.*}}@_SomeFunc_ui1(ui1 {llvm.zeroext}) -> (ui1 {llvm.zeroext})
// X64: func.func{{.*}}@_SomeFunc_ui1(ui1 {llvm.zeroext}) -> (ui1 {llvm.zeroext})
// AARCH64: func.func{{.*}}@_SomeFunc_ui1(ui1 {llvm.zeroext}) -> (ui1 {llvm.zeroext})
// PPC: func.func{{.*}}@_SomeFunc_ui1(ui1 {llvm.zeroext}) -> (ui1 {llvm.zeroext})
// SPARCV9: func.func{{.*}}@_SomeFunc_ui1(ui1 {llvm.zeroext}) -> (ui1 {llvm.zeroext})
// LOONGARCH64: func.func{{.*}}@_SomeFunc_ui1(ui1 {llvm.zeroext}) -> (ui1 {llvm.zeroext})
func.func @_QPtest_ui1(%arg0: !fir.ref<!fir.logical<4>> {fir.bindc_name = "x"}) {
  %0 = fir.load %arg0 : !fir.ref<!fir.logical<4>>
  %1 = fir.convert %0 : (!fir.logical<4>) -> ui1
  %2 = fir.call @_SomeFunc_ui1(%1) : (ui1) -> ui1
  return
}
func.func private @_SomeFunc_ui1(ui1) -> ui1 attributes {fir.runtime}

// -----

// subroutine test(x, y)
//   use iso_c_binding
//   interface
//      integer(c_int8_t) function cfun8(x) bind(C)
//        integer(c_int8_t),value :: x
//      end function cfun8
//      integer(c_int16_t) function cfun16(x) bind(C)
//        integer(c_int16_t),value :: x
//      end function cfun16
//   end interface
//   integer(c_int8_t) :: x
//   integer(c_int16_t) :: y
//   x = cfun8(x)
//   y = cfun16(y)
// end subroutine test

// ALL-LABEL: @_QPtest_bindc
// ALL: func.func private @cfun8(i8 {llvm.signext}) -> (i8 {llvm.signext}) attributes {fir.bindc_name = "cfun8"}
// ALL: func.func private @cfun16(i16 {llvm.signext}) -> (i16 {llvm.signext}) attributes {fir.bindc_name = "cfun16"}
func.func @_QPtest_bindc(%arg0: !fir.ref<i8> {fir.bindc_name = "x"}, %arg1: !fir.ref<i16> {fir.bindc_name = "y"}) {
  %0 = fir.load %arg0 : !fir.ref<i8>
  %1 = fir.call @cfun8(%0) fastmath<contract> : (i8) -> i8
  fir.store %1 to %arg0 : !fir.ref<i8>
  %2 = fir.load %arg1 : !fir.ref<i16>
  %3 = fir.call @cfun16(%2) fastmath<contract> : (i16) -> i16
  fir.store %3 to %arg1 : !fir.ref<i16>
  return
}
func.func private @cfun8(i8) -> i8 attributes {fir.bindc_name = "cfun8"}
func.func private @cfun16(i16) -> i16 attributes {fir.bindc_name = "cfun16"}
