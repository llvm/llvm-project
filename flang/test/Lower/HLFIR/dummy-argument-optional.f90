! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! Test OPTIONAL lowering on caller/callee
module opt
  implicit none
contains

! Test optional character function
! CHECK-LABEL: func @_QMoptPchar_proc(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.char<1,3>>,
character(len=3) function char_proc(i)
  integer :: i
  char_proc = "XYZ"
end function
! CHECK-LABEL: func @_QMoptPuse_char_proc(
! CHECK-SAME: %[[arg0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc},
subroutine use_char_proc(f, c)
  optional :: f
  interface
    character(len=3) function f(i)
      integer :: i
    end function
  end interface
  character(len=3) :: c
! CHECK: %[[boxProc:.*]] = fir.extract_value %[[arg0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %[[boxAddr:.*]] = fir.box_addr %[[boxProc]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK: %[[boxProc2:.*]] = fir.emboxproc %[[boxAddr]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %{{.*}}, [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[boxProc3:.*]] = fir.extract_value %[[tuple3]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK: %{{.*}} = fir.is_present %[[boxProc3]] : (!fir.boxproc<() -> ()>) -> i1
  if (present(f)) then
    c = f(0)
  else
    c = "ABC"
  end if
end subroutine
! CHECK-LABEL: func @_QMoptPcall_use_char_proc(
subroutine call_use_char_proc()
  character(len=3) :: c
! CHECK: %[[boxProc:.*]] = fir.absent !fir.boxproc<() -> ()>
! CHECK: %[[undef:.*]] = fir.undefined index
! CHECK: %[[charLen:.*]] = fir.convert %[[undef]] : (index) -> i64
! CHECK: %[[tuple:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple2:.*]] = fir.insert_value %[[tuple]], %[[boxProc]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple3:.*]] = fir.insert_value %[[tuple2]], %[[charLen]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QMoptPuse_char_proc(%[[tuple3]], %{{.*}}){{.*}} : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxchar<1>) -> ()
  call use_char_proc(c=c)
! CHECK: %[[funcAddr:.*]] = fir.address_of(@_QMoptPchar_proc) : (!fir.ref<!fir.char<1,3>>, index, {{.*}}) -> !fir.boxchar<1>
! CHECK: %[[c3:.*]] = arith.constant 3 : i64
! CHECK: %[[boxProc2:.*]] = fir.emboxproc %[[funcAddr]] : ((!fir.ref<!fir.char<1,3>>, index, {{.*}}) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK: %[[tuple4:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple5:.*]] = fir.insert_value %[[tuple4]], %[[boxProc2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: %[[tuple6:.*]] = fir.insert_value %[[tuple5]], %[[c3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK: fir.call @_QMoptPuse_char_proc(%[[tuple6]], {{.*}}){{.*}} : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxchar<1>) -> ()
  call use_char_proc(char_proc, c)
end subroutine

end module
