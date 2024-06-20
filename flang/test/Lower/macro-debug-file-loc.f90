! Test that the expanded macros have the location information
! RUN: %flang_fc1 -mmlir --mlir-print-debuginfo -emit-fir -o - %s | FileCheck %s

#define CMD(fname) fname()

subroutine foo()
end subroutine

subroutine test()
  ! CHECK: fir.call @_QPfoo() fastmath<contract> : () -> () loc(#[[CALL_LOC:.*]])
  call CMD(foo)
end subroutine

#define IVAR i

integer function ifoo()
  ifoo = 0
end function

subroutine test2()
  integer :: i
  ! CHECK: fir.call @_QPifoo(){{.*}} loc(#[[IFOO_CALL_LOC:.*]])
  IVAR = ifoo()
end subroutine

! CHECK: #[[CALL_LOC]] = loc("{{.*}}macro-debug-file-loc.f90":11:3)
! CHECK: #[[IFOO_CALL_LOC]] = loc("{{.*}}macro-debug-file-loc.f90":23:3)
