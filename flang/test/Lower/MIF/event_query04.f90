! RUN: not %flang_fc1 -emit-hlfir -fcoarray %s -o - 2>&1 | FileCheck %s

!CHECK: not yet implemented: coarray: EVENT_QUERY with coindexed COUNT.

subroutine query_with_coindex(k)
  use iso_fortran_env, only: event_type
  type(event_type), save :: ev[*]
  integer, save :: count[*]
  integer :: i
  
  call event_query(ev, count[i])
end subroutine
