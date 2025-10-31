// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QPerror_stop_test() {
  %0 = fir.dummy_scope : !fir.dscope
  mif.error_stop : () -> ()
  fir.unreachable
}
func.func @_QPerror_stop_code1() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "int_code", uniq_name = "_QFerror_stop_code1Eint_code"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFerror_stop_code1Eint_code"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.load %2#0 : !fir.ref<i32>
  mif.error_stop code %3 : (i32) -> ()
  fir.unreachable
}
func.func @_QPerror_stop_code2() {
  %0 = fir.dummy_scope : !fir.dscope
  %c26_i32 = arith.constant 26 : i32
  %1 = hlfir.no_reassoc %c26_i32 : i32
  mif.error_stop code %1 : (i32) -> ()
  fir.unreachable
}
func.func @_QPerror_stop_code_char1() {
  %0 = fir.dummy_scope : !fir.dscope
  %c128 = arith.constant 128 : index
  %1 = fir.alloca !fir.char<1,128> {bindc_name = "char_code", uniq_name = "_QFerror_stop_code_char1Echar_code"}
  %2:2 = hlfir.declare %1 typeparams %c128 {uniq_name = "_QFerror_stop_code_char1Echar_code"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
  %3 = fir.emboxchar %2#0, %c128 : (!fir.ref<!fir.char<1,128>>, index) -> !fir.boxchar<1>
  mif.error_stop code %3 : (!fir.boxchar<1>) -> ()
  fir.unreachable
}
func.func @_QPerror_stop_code_char2() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QQclX63) : !fir.ref<!fir.char<1>>
  %c1 = arith.constant 1 : index
  %2:2 = hlfir.declare %1 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX63"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
  %3 = fir.emboxchar %2#0, %c1 : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
  mif.error_stop code %3 : (!fir.boxchar<1>) -> ()
  fir.unreachable
}
func.func @_QPerror_stop_code_char3() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QQclX70726F6772616D206661696C6564) : !fir.ref<!fir.char<1,14>>
  %c14 = arith.constant 14 : index
  %2:2 = hlfir.declare %1 typeparams %c14 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX70726F6772616D206661696C6564"} : (!fir.ref<!fir.char<1,14>>, index) -> (!fir.ref<!fir.char<1,14>>, !fir.ref<!fir.char<1,14>>)
  %3 = hlfir.as_expr %2#0 : (!fir.ref<!fir.char<1,14>>) -> !hlfir.expr<!fir.char<1,14>>
  %4:3 = hlfir.associate %3 typeparams %c14 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,14>>, index) -> (!fir.ref<!fir.char<1,14>>, !fir.ref<!fir.char<1,14>>, i1)
  %5 = fir.emboxchar %4#0, %c14 : (!fir.ref<!fir.char<1,14>>, index) -> !fir.boxchar<1>
  mif.error_stop code %5 : (!fir.boxchar<1>) -> ()
  fir.unreachable
}
func.func @_QPerror_stop_code_quiet1() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.logical<4> {bindc_name = "bool", uniq_name = "_QFerror_stop_code_quiet1Ebool"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFerror_stop_code_quiet1Ebool"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %3 = fir.alloca i32 {bindc_name = "int_code", uniq_name = "_QFerror_stop_code_quiet1Eint_code"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFerror_stop_code_quiet1Eint_code"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.load %4#0 : !fir.ref<i32>
  %6 = fir.load %2#0 : !fir.ref<!fir.logical<4>>
  mif.error_stop code %5 quiet %6 : (i32, !fir.logical<4>) -> ()
  fir.unreachable
}
func.func @_QPerror_stop_code_quiet2() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "int_code", uniq_name = "_QFerror_stop_code_quiet2Eint_code"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFerror_stop_code_quiet2Eint_code"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.load %2#0 : !fir.ref<i32>
  %true = arith.constant true
  mif.error_stop code %3 quiet %true : (i32, i1) -> ()
  fir.unreachable
}
func.func @_QPerror_stop_code_quiet3() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "int_code", uniq_name = "_QFerror_stop_code_quiet3Eint_code"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFerror_stop_code_quiet3Eint_code"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.load %2#0 : !fir.ref<i32>
  %4 = hlfir.no_reassoc %3 : i32
  %false = arith.constant false
  mif.error_stop code %4 quiet %false : (i32, i1) -> ()
  fir.unreachable
}
func.func private @_FortranAStopStatement(i32, i1, i1) attributes {fir.runtime}
func.func private @_FortranAStopStatementText(!fir.ref<i8>, i64, i1, i1) attributes {fir.runtime}
fir.global linkonce @_QQclX63 constant : !fir.char<1> {
  %0 = fir.string_lit "c"(1) : !fir.char<1>
  fir.has_value %0 : !fir.char<1>
}
fir.global linkonce @_QQclX70726F6772616D206661696C6564 constant : !fir.char<1,14> {
  %0 = fir.string_lit "program failed"(14) : !fir.char<1,14>
  fir.has_value %0 : !fir.char<1,14>
}


// CHECK-label : func.func @_QPerror_stop_test
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: fir.store %[[FALSE]] to %[[QUIET:.*]] : !fir.ref<i1>
// CHECK2: %[[CODE_CHAR:.*]] = fir.absent !fir.boxchar<1>
// CHECK2: %[[CODE_INT:.*]] = fir.absent !fir.ref<i32>
// CHECK2: fir.call @_QMprifPprif_error_stop(%[[QUIET]], %[[CODE_INT]], %[[CODE_CHAR]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()

// CHECK-label : func.func @_QPerror_stop_code1
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: fir.store %[[FALSE]] to %[[QUIET:.*]] : !fir.ref<i1>
// CHECK: %[[CODE_CHAR:.*]] = fir.absent !fir.boxchar<1>
// CHECK: fir.call @_QMprifPprif_error_stop(%[[QUIET]], %[[CODE_INT:.*]], %[[CODE_CHAR]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()

// CHECK-label : func.func @_QPerror_stop_code2
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: fir.store %[[FALSE]] to %[[QUIET:.*]] : !fir.ref<i1>
// CHECK: %[[CODE_CHAR:.*]] = fir.absent !fir.boxchar<1>
// CHECK: fir.call @_QMprifPprif_error_stop(%[[QUIET]], %[[CODE_INT:.*]], %[[CODE_CHAR]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()

// CHECK-label : func.func @_QPerror_stop_code_char1
// CHECK: %[[CODE_CHAR:.*]] = fir.emboxchar %[[VAL_X:.*]]#0, %[[C128:.*]] : (!fir.ref<!fir.char<1,128>>, index) -> !fir.boxchar<1>
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: fir.store %[[FALSE]] to %[[QUIET:.*]] : !fir.ref<i1>
// CHECK: %[[CODE_INT:.*]] = fir.absent !fir.ref<i32>
// CHECK: fir.call @_QMprifPprif_error_stop(%[[QUIET]], %[[CODE_INT]], %[[CODE_CHAR]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()

// CHECK-label : func.func @_QPerror_stop_code_char2
// CHECK: %[[CODE_CHAR:.*]] = fir.emboxchar %[[VAL_X:.*]]#0, %[[C1:.*]] : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: fir.store %[[FALSE]] to %[[QUIET:.*]] : !fir.ref<i1>
// CHECK: %[[CODE_INT:.*]] = fir.absent !fir.ref<i32>
// CHECK: fir.call @_QMprifPprif_error_stop(%[[QUIET]], %[[CODE_INT]], %[[CODE_CHAR]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()

// CHECK-label : func.func @_QPerror_stop_code_char3
// CHECK: %[[CODE_CHAR:.*]] = fir.emboxchar %[[VAL_X:.*]]#0, %[[C14:.*]] : (!fir.ref<!fir.char<1,14>>, index) -> !fir.boxchar<1>
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: fir.store %[[FALSE]] to %[[QUIET:.*]] : !fir.ref<i1>
// CHECK: %[[CODE_INT:.*]] = fir.absent !fir.ref<i32>
// CHECK: fir.call @_QMprifPprif_error_stop(%[[QUIET]], %[[CODE_INT]], %[[CODE_CHAR]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()

// CHECK-label : func.func @_QPerror_stop_code_quiet1
// CHECK: %[[VAL_1:.*]] = fir.load %[[VAL_Q:.*]]#0 : !fir.ref<!fir.logical<4>>
// CHECK: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.logical<4>) -> i1
// CHECK: fir.store %[[VAL_2]] to %[[QUIET:.*]] : !fir.ref<i1>
// CHECK: %[[CODE_CHAR:.*]] = fir.absent !fir.boxchar<1>
// CHECK: fir.call @_QMprifPprif_error_stop(%[[QUIET]], %[[CODE_INT:.*]], %[[CODE_CHAR]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()

// CHECK-label : func.func @_QPerror_stop_code_quiet2
// CHECK: %[[TRUE:.*]] = arith.constant true 
// CHECK: fir.store %[[TRUE]] to %[[QUIET:.*]] : !fir.ref<i1>
// CHECK: %[[CODE_CHAR:.*]] = fir.absent !fir.boxchar<1>
// CHECK: fir.call @_QMprifPprif_error_stop(%[[QUIET]], %[[CODE_INT:.*]], %[[CODE_CHAR]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()

// CHECK-label : func.func @_QPerror_stop_code_quiet3
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: fir.store %[[FALSE]] to %[[QUIET:.*]] : !fir.ref<i1>
// CHECK: %[[CODE_CHAR:.*]] = fir.absent !fir.boxchar<1>
// CHECK: fir.call @_QMprifPprif_error_stop(%[[QUIET]], %[[CODE_INT:.*]], %[[CODE_CHAR]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()

