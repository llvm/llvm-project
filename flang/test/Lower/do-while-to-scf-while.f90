! RUN: bbc -emit-fir -hlfir=false -lower-do-while-to-scf-while %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPsimple_do_while()
! CHECK: scf.while
! CHECK: scf.condition
! CHECK: scf.yield
subroutine simple_do_while()
  implicit none
  integer :: i

  i = 1
  do while (i <= 10)
    print *, "i =", i
    i = i + 1
  end do
end subroutine simple_do_while

! CHECK-LABEL: func.func @_QPdo_while_with_exit()
! CHECK-NOT: scf.while
! CHECK: cf.cond_br
subroutine do_while_with_exit()
  implicit none
  integer :: i

  i = 1
  do while (i <= 10)
    if (i == 5) exit
    i = i + 1
  end do
end subroutine do_while_with_exit

! CHECK-LABEL: func.func @_QPnested_do_while()
! CHECK: scf.while
! CHECK: scf.while
subroutine nested_do_while()
  implicit none
  integer :: i, j

  i = 1
  do while (i <= 3)
    j = 1
    do while (j <= 2)
      j = j + 1
    end do
    i = i + 1
  end do
end subroutine nested_do_while

