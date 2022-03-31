! RUN: bbc %s -emit-fir --canonicalize -o - | FileCheck %s

! CHECK-LABEL pause_test
subroutine pause_test()
  ! CHECK: fir.call @_Fortran{{.*}}PauseStatement()
  ! CHECK-NEXT: return
  pause
end subroutine
