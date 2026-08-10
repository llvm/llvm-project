! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test external procedure as actual argument with the implicit character type.

! CHECK-LABEL: func.func @_QQmain()
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QPext_func) : (!fir.ref<!fir.char<1,20>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_1]] : ((!fir.ref<!fir.char<1,20>>, index) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_5:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %{{.*}}, [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QFPsub(%[[VAL_7]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

! CHECK-LABEL: func.func private @_QFPsub(
! CHECK-SAME: %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}
! CHECK: %[[VAL_5:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK: %[[VAL_7:.*]] = fir.emboxproc %[[VAL_6]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_11:.*]] = fir.extract_value %{{.*}}, [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %[[VAL_16:.*]] = fir.box_addr %[[VAL_11]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<!fir.char<1,20>>, index) -> !fir.boxchar<1>)
! CHECK: %[[VAL_18:.*]] = fir.call %[[VAL_16]](%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.char<1,20>>, index) -> !fir.boxchar<1>

! CHECK-LABEL: func.func @_QPext_func(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.char<1,20>>{{.*}}, %[[VAL_1:.*]]: index{{.*}}) -> !fir.boxchar<1>

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
  character(20) res
  res = "hello world"
end
