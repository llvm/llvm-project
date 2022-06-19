! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test external procedure as actual argument with the implicit character type.

! CHECK-LABEL: func @_QQmain
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPext_func) : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_2:.*]] = fir.undefined i64
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_1]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_2]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QFPsub(%[[VAL_5]]) : (tuple<!fir.boxproc<() -> ()>, i64>) -> ()
! CHECK:  return

! CHECK-LABEL: func @_QPext_func(
! CEHCK: %[[ARG_0:.*]]: !fir.ref<!fir.char<1,?>>, %[[ARG_1:.*]]: index) -> !fir.boxchar<1> {
program m
  external :: ext_func
  call sub(ext_func)

contains
  subroutine sub(arg)
    character(20), external :: arg
    print *, arg()
  end
end

function ext_func() result(res)
  character(*) res
  res = "hello world"
end
