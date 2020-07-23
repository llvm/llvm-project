! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL stop_test
subroutine stop_test(b)
 ! CHECK-DAG: %[[c0:.*]] = constant 0 : i32
 ! CHECK-DAG: %[[false:.*]] = constant false
 ! CHECK: call @_Fortran{{.*}}StopStatement(%[[c0]], %[[false]], %[[false]])
 stop
end subroutine

! CHECK-LABEL stop_code
subroutine stop_code()
  stop 42
 ! CHECK-DAG: %[[c42:.*]] = constant 42 : i32
 ! CHECK-DAG: %[[false:.*]] = constant false
 ! CHECK: call @_Fortran{{.*}}StopStatement(%[[c42]], %[[false]], %[[false]])
end subroutine

! CHECK-LABEL stop_error
subroutine stop_error()
  error stop
 ! CHECK-DAG: %[[c0:.*]] = constant 0 : i32
 ! CHECK-DAG: %[[true:.*]] = constant true
 ! CHECK-DAG: %[[false:.*]] = constant false
 ! CHECK: call @_Fortran{{.*}}StopStatement(%[[c0]], %[[true]], %[[false]])
end subroutine

! CHECK-LABEL stop_quiet
subroutine stop_quiet(b)
  logical :: b
  stop, quiet = b
 ! CHECK-DAG: %[[c0:.*]] = constant 0 : i32
 ! CHECK-DAG: %[[false:.*]] = constant false
 ! CHECK-DAG: %[[b:.*]] = fir.load %arg0
 ! CHECK-DAG: %[[bi1:.*]] = fir.convert %[[b]] : (!fir.logical<4>) -> i1
 ! CHECK: call @_Fortran{{.*}}StopStatement(%[[c0]], %[[false]], %[[bi1]])
end subroutine

! CHECK-LABEL stop_error_code_quiet
subroutine stop_error_code_quiet(b)
  logical :: b
  error stop 66, quiet = b
 ! CHECK-DAG: %[[c66:.*]] = constant 66 : i32
 ! CHECK-DAG: %[[true:.*]] = constant true
 ! CHECK-DAG: %[[b:.*]] = fir.load %arg0
 ! CHECK-DAG: %[[bi1:.*]] = fir.convert %[[b]] : (!fir.logical<4>) -> i1
 ! CHECK: call @_Fortran{{.*}}StopStatement(%[[c66]], %[[true]], %[[bi1]])
end subroutine

! CHECK: func @_Fortran{{.*}}StopStatement(i32, i1, i1) -> none
