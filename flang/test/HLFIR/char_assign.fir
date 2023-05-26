// RUN: fir-opt --split-input-file %s -convert-hlfir-to-fir | FileCheck %s

// Verify that the special case of length-one character assignment
// is using valid fir.converts.

// -----

// CHECK-LABEL: func.func @_QPtest1
// CHECK: [[C_DECL:%[0-9]*]] = fir.declare{{.*}}{uniq_name = "_QFtest1Ec"} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
// CHECK: [[LOCAL_DECL:%[0-9]*]] = fir.declare{{.*}}{uniq_name = "_QFtest1Elocal"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
// CHECK: [[CVT:%[0-9]*]] = fir.convert [[C_DECL]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
// CHECK: [[LD:%[0-9]*]] = fir.load [[CVT]] : !fir.ref<!fir.char<1>>
// CHECK: fir.store [[LD]] to [[LOCAL_DECL]] : !fir.ref<!fir.char<1>>
module {
  func.func @_QPtest1(%arg0: !fir.boxchar<1> {fir.bindc_name = "c"}) {
    %c1 = arith.constant 1 : index
    %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %1:2 = hlfir.declare %0#0 typeparams %c1 {uniq_name = "_QFtest1Ec"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %2 = fir.alloca !fir.char<1> {bindc_name = "local", uniq_name = "_QFtest1Elocal"}
    %3:2 = hlfir.declare %2 typeparams %c1 {uniq_name = "_QFtest1Elocal"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    hlfir.assign %1#0 to %3#0 : !fir.boxchar<1>, !fir.ref<!fir.char<1>>
    return
  }
}

// -----

// CHECK-LABEL: func.func @_QPtest2
// CHECK: [[C_DECL:%[0-9]*]] = fir.declare{{.*}}{uniq_name = "_QFtest2Ec"} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
// CHECK: [[LOCAL_DECL:%[0-9]*]] = fir.declare{{.*}}{uniq_name = "_QFtest2Elocal"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
// CHECK: [[LD:%[0-9]*]] = fir.load [[LOCAL_DECL]] : !fir.ref<!fir.char<1>>
// CHECK: [[CVT:%[0-9]*]] = fir.convert [[C_DECL]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
// CHECK: fir.store [[LD]] to [[CVT]] : !fir.ref<!fir.char<1>>
module {
  func.func @_QPtest2(%arg0: !fir.boxchar<1> {fir.bindc_name = "c"}) {
    %c1 = arith.constant 1 : index
    %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %1:2 = hlfir.declare %0#0 typeparams %c1 {uniq_name = "_QFtest2Ec"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %2 = fir.alloca !fir.char<1> {bindc_name = "local", uniq_name = "_QFtest2Elocal"}
    %3:2 = hlfir.declare %2 typeparams %c1 {uniq_name = "_QFtest2Elocal"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    hlfir.assign %3#0 to %1#0 : !fir.ref<!fir.char<1>>, !fir.boxchar<1>
    return
  }
}
