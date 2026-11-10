! RUN: split-file %s %t
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/automatic.f90 -o - 2>&1 | FileCheck %s --check-prefix=AUTOMATIC
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/assumed.f90 -o - 2>&1 | FileCheck %s --check-prefix=ASSUMED
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/host-copy.f90 -o - 2>&1 | FileCheck %s --check-prefix=HOST-COPY
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/host-copyout.f90 -o - 2>&1 | FileCheck %s --check-prefix=HOST-COPYOUT

! These CHARACTER scalars are represented as fir.boxchar. Check that
! OpenACC data-clause lowering emits a source-located NYI diagnostic.

!--- automatic.f90
subroutine automatic(n)
  integer, intent(in) :: n
  character(n) :: s
  !$acc serial copyin(s)
  !$acc end serial
end subroutine
! AUTOMATIC: error: loc("{{.*}}automatic.f90":4:{{[0-9]+}}): {{.*}}not yet implemented: OpenACC mapping of scalar CHARACTER variable with runtime-determined length or host association

!--- assumed.f90
subroutine assumed(s)
  character(*) :: s
  !$acc serial copyin(s)
  !$acc end serial
end subroutine
! ASSUMED: error: loc("{{.*}}assumed.f90":3:{{[0-9]+}}): {{.*}}not yet implemented: OpenACC mapping of scalar CHARACTER variable with runtime-determined length or host association

!--- host-copy.f90
program host_copy
  character(4) :: s
  call inner
contains
  subroutine inner
    !$acc serial copy(s)
    s = 'abcd'
    !$acc end serial
  end subroutine
end program
! HOST-COPY: error: loc("{{.*}}host-copy.f90":6:{{[0-9]+}}): {{.*}}not yet implemented: OpenACC mapping of scalar CHARACTER variable with runtime-determined length or host association

!--- host-copyout.f90
program host_copyout
  character*13 :: s
  call inner
contains
  subroutine inner
    !$acc serial copyout(s)
    s = 'abcd'
    !$acc end serial
  end subroutine
end program
! HOST-COPYOUT: error: loc("{{.*}}host-copyout.f90":6:{{[0-9]+}}): {{.*}}not yet implemented: OpenACC mapping of scalar CHARACTER variable with runtime-determined length or host association
