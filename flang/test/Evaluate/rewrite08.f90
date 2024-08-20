! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
subroutine s(oi,ol)
  integer(1), optional, intent(in) :: oi
  logical(1), optional, intent(in) :: ol
  integer(1), allocatable :: ai
  logical(1), allocatable :: al
  integer(1), pointer :: pi
  logical(1), pointer :: pl
!CHECK: PRINT *, ishftc(-1_4,1_4,oi)
!CHECK: PRINT *, ishftc(-1_4,1_4,ai)
!CHECK: PRINT *, ishftc(-1_4,1_4,pi)
!CHECK: PRINT *, findloc([INTEGER(4)::1_4,2_4,1_4],1_4,back=ol)
!CHECK: PRINT *, findloc([INTEGER(4)::1_4,2_4,1_4],1_4,back=al)
!CHECK: PRINT *, findloc([INTEGER(4)::1_4,2_4,1_4],1_4,back=pl)
  print *, ishftc(-1,1,oi)
  print *, ishftc(-1,1,ai)
  print *, ishftc(-1,1,pi)
  print *, findloc([1,2,1],1,back=ol)
  print *, findloc([1,2,1],1,back=al)
  print *, findloc([1,2,1],1,back=pl)
end
