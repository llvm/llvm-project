! REQUIRES: asserts
! RUN: bbc -emit-fir %s --math-runtime=precise -o - | FileCheck -check-prefix=CHECK %s
! RUN: bbc -emit-fir %s --math-runtime=precise -debug-only=flang-lower-intrinsic,flang-lower-expr 2>&1 | FileCheck -check-prefix=CHECK-WARN %s

! CHECK-LABEL: func.func @_QPtest
! CHECK: fir.call @atanh({{[^,]*}}){{.*}}: (i32) -> i32
! CHECK-LABEL: func.func @_QPtest2
! CHECK: %[[ADDR:.*]] = fir.address_of(@atanh) : (i32) -> i32
! CHECK: %[[CAST:.*]] = fir.convert %[[ADDR]] : ((i32) -> i32) -> ((f64) -> f64)
! CHECK: fir.call %[[CAST]]({{[^,]*}}){{.*}}: (f64) -> f64

subroutine test(x)
  interface
     integer function atanh(x) bind(c)
       integer,Value :: x
     end function atanh
  end interface
  integer :: x
  print *,atanh(x)
end subroutine test
subroutine test2(x)
  real(8) :: x
  print *,atanh(x)
end subroutine test2

! CHECK-LABEL: func.func @_QPtest3
! CHECK: fir.call @asinh({{[^,]*}}){{.*}}: (f64) -> f64
! CHECK-LABEL: func.func @_QPtest4
! CHECK: %[[ADDR:.*]] = fir.address_of(@asinh) : (f64) -> f64
! CHECK: %[[CAST:.*]] = fir.convert %[[ADDR]] : ((f64) -> f64) -> ((i32) -> i32)
! CHECK: fir.call %[[CAST]]({{[^,]*}}){{.*}}: (i32) -> i32
subroutine test3(x)
  real(8) :: x
  print *,asinh(x)
end subroutine test3
subroutine test4(x)
  interface
     integer function asinh(x) bind(c)
       integer,Value :: x
     end function asinh
  end interface
  integer :: x
  print *,asinh(x)
end subroutine test4

! CHECK-WARN:      warning: loc({{.*}}math-name-conflict.f90{{.*}}): function
! CHECK-WARN-SAME: signature mismatch for 'atanh' may lead to undefined behavior.
! CHECK-WARN:      warning: loc({{.*}}math-name-conflict.f90{{.*}}): function
! CHECK-WARN-SAME: name 'asinh' conflicts with a runtime function
! CHECK-WARN-SAME: name used by Flang - this may lead to undefined behavior
