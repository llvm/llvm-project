! Test that the expanded macros have the location information
! RUN: %flang_fc1 -mmlir --mlir-print-debuginfo -emit-fir -o - %s | FileCheck %s

#define CMD(fname) fname()

subroutine foo()
end subroutine

subroutine test()
  ! CHECK: fir.call @_QPfoo() fastmath<contract> : () -> () loc(#[[CALL_LOC:.*]])
  call CMD(foo)
end subroutine
! CHECK: #[[CALL_LOC]] = loc("{{.*}}macro-debug-file-loc.f90":11:3)
