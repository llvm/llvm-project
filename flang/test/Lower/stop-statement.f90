! RUN: bbc %s -emit-fir --canonicalize -o - | FileCheck %s

! CHECK-LABEL stop_test
subroutine stop_test()
 ! CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
 ! CHECK-DAG: %[[false:.*]] = arith.constant false
 ! CHECK: fir.call @_Fortran{{.*}}StopStatement(%[[c0]], %[[false]], %[[false]])
 ! CHECK-NEXT: fir.unreachable
 stop
end subroutine 


! CHECK-LABEL stop_error
subroutine stop_error()
 error stop
 ! CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
 ! CHECK-DAG: %[[true:.*]] = arith.constant true
 ! CHECK-DAG: %[[false:.*]] = arith.constant false
 ! CHECK: fir.call @_Fortran{{.*}}StopStatement(%[[c0]], %[[true]], %[[false]])
 ! CHECK-NEXT: fir.unreachable
end subroutine
