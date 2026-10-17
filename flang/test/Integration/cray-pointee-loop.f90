! Check that re-associating a Cray pointee with its Cray pointer before each
! reference does not leave a runtime call (or descriptor reloads) in the loop.
! RUN: %flang_fc1 -emit-llvm -O2 -o - %s | FileCheck %s

integer function find(p, n, cur)
  integer(8) :: p
  integer :: n, cur, i
  integer :: pg(*)
  pointer (pp, pg)
  pp = p
  find = 0
  do i = 1, n
    if (pg(i) == cur) then
      find = i
      return
    end if
  end do
end function
! CHECK-LABEL: define {{.*}}i32 @find_(
! CHECK-NOT:     call
! CHECK-NOT:     alloca
! CHECK:         ret i32

subroutine copy_out(p, n, v)
  integer(8) :: p, n, i
  integer(4) :: v(*)
  integer(4) :: buf(*)
  pointer (pb, buf)
  pb = p
  do i = 1, n
    v(i) = buf(i)
  end do
end subroutine
! CHECK-LABEL: define void @copy_out_(
! CHECK-NOT:     PointerAssociateScalar
! CHECK-NOT:     alloca
! CHECK:         ret void
