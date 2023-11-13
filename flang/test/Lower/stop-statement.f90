! RUN: bbc %s -emit-fir -hlfir=false --canonicalize -o - | FileCheck %s

! CHECK-LABEL: stop_test
subroutine stop_test()
 ! CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
 ! CHECK-DAG: %[[false:.*]] = arith.constant false
 ! CHECK: fir.call @_Fortran{{.*}}StopStatement(%[[c0]], %[[false]], %[[false]])
 ! CHECK-NEXT: fir.unreachable
 stop
end subroutine

! CHECK-LABEL: stop_code
subroutine stop_code()
  stop 42
 ! CHECK-DAG: %[[c42:.*]] = arith.constant 42 : i32
 ! CHECK-DAG: %[[false:.*]] = arith.constant false
 ! CHECK: fir.call @_Fortran{{.*}}StopStatement(%[[c42]], %[[false]], %[[false]])
 ! CHECK-NEXT: fir.unreachable
end subroutine

! CHECK-LABEL: stop_error
subroutine stop_error()
  error stop
 ! CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
 ! CHECK-DAG: %[[true:.*]] = arith.constant true
 ! CHECK-DAG: %[[false:.*]] = arith.constant false
 ! CHECK: fir.call @_Fortran{{.*}}StopStatement(%[[c0]], %[[true]], %[[false]])
 ! CHECK-NEXT: fir.unreachable
end subroutine

! CHECK-LABEL: stop_quiet
subroutine stop_quiet()
  logical :: b
  stop, quiet = b
 ! CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
 ! CHECK-DAG: %[[false:.*]] = arith.constant false
 ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.logical<4> {bindc_name = "b", uniq_name = "_QFstop_quietEb"}
 ! CHECK: %[[b:.*]] = fir.load %[[ALLOCA]]
 ! CHECK: %[[bi1:.*]] = fir.convert %[[b]] : (!fir.logical<4>) -> i1
 ! CHECK: fir.call @_Fortran{{.*}}StopStatement(%[[c0]], %[[false]], %[[bi1]])
 ! CHECK-NEXT: fir.unreachable
end subroutine

! CHECK-LABEL: stop_quiet_constant
subroutine stop_quiet_constant()
  stop, quiet = .true.
 ! CHECK-DAG: %[[true:.*]] = arith.constant true
 ! CHECK-DAG: %[[false:.*]] = arith.constant false
 ! CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
 ! CHECK: fir.call @_Fortran{{.*}}StopStatement(%[[c0]], %[[false]], %[[true]])
 ! CHECK-NEXT: fir.unreachable
end subroutine

! CHECK-LABEL: stop_error_code_quiet
subroutine stop_error_code_quiet(b)
  logical :: b
  error stop 66, quiet = b
 ! CHECK-DAG: %[[c66:.*]] = arith.constant 66 : i32
 ! CHECK-DAG: %[[true:.*]] = arith.constant true
 ! CHECK-DAG: %[[b:.*]] = fir.load %arg0
 ! CHECK-DAG: %[[bi1:.*]] = fir.convert %[[b]] : (!fir.logical<4>) -> i1
 ! CHECK: fir.call @_Fortran{{.*}}StopStatement(%[[c66]], %[[true]], %[[bi1]])
 ! CHECK-NEXT: fir.unreachable
end subroutine

! CHECK-LABEL: stop_char_lit
subroutine stop_char_lit
  ! CHECK-DAG: %[[false:.*]] = arith.constant false
  ! CHECK-DAG: %[[five:.*]] = arith.constant 5 : index
  ! CHECK-DAG: %[[lit:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,5>>
  ! CHECK-DAG: %[[buff:.*]] = fir.convert %[[lit]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK-DAG: %[[len:.*]] = fir.convert %[[five]] : (index) -> i64
  ! CHECK: fir.call @{{.*}}StopStatementText(%[[buff]], %[[len]], %[[false]], %[[false]]) {{.*}}:
  ! CHECK-NEXT: fir.unreachable
  stop 'crash'
end subroutine stop_char_lit

! CHECK-DAG: func private @_Fortran{{.*}}StopStatement(i32, i1, i1) -> none
! CHECK-DAG: func private @_Fortran{{.*}}StopStatementText(!fir.ref<i8>, i64, i1, i1) -> none
