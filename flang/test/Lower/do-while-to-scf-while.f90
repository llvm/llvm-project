! RUN: bbc -emit-hlfir -lower-do-while-to-scf-while %s -o - | FileCheck %s

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

! CHECK-LABEL: func.func @_QPdo_while_goto_internal_forward()
! CHECK: scf.while
subroutine do_while_goto_internal_forward()
  implicit none
  integer :: i, sum

  i = 0
  sum = 0
  do while (i < 10)
    i = i + 1

    if (mod(i, 2) == 0) goto 100
    sum = sum + i

100 continue
  end do
  print *, "sum=", sum
end subroutine do_while_goto_internal_forward

! CHECK-LABEL: func.func @_QPdo_while_goto_internal_backedge()
! CHECK-NOT: scf.while
! CHECK: cf.cond_br
! CHECK: cf.br
subroutine do_while_goto_internal_backedge()
  implicit none
  integer :: i, sum

  i = 0
  sum = 0
  do while (i < 5)
    i = i + 1

10  continue
    sum = sum + 1
    if (sum < 3) goto 10
  end do
  print *, "sum=", sum
end subroutine do_while_goto_internal_backedge

! CHECK-LABEL:   func.func @_QPtest_after_unstructured(
! CHECK:  scf.while
! CHECK-NOT: cf.br
! CHECK: return
subroutine test_after_unstructured(cdt, switch)
  logical :: cdt, eval
  integer :: switch, i = 1
  if (cdt) then
    select case (switch)
      case (0)
        call print1()
    end select
  end if
  do while(eval(i))
    call incr(i)
  end do
end subroutine
