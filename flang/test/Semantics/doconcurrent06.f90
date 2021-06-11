! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! C1167 -- An exit-stmt shall not appear within a DO CONCURRENT construct if 
! it belongs to that construct or an outer construct.

subroutine do_concurrent_test1(n)
  implicit none
  integer :: i1,i2,i3,i4,i5,i6,n
  mytest1: if (n>0) then
  nc1:       do concurrent(i1=1:n)
  nc2:         do i2=1,n
  nc3:           do concurrent(i3=1:n)
  nc4:             do i4=1,n
  nc5:               do concurrent(i5=1:n)
  nc6:                 do i6=1,n
!ERROR: EXIT must not leave a DO CONCURRENT statement
!ERROR: EXIT must not leave a DO CONCURRENT statement
!ERROR: EXIT must not leave a DO CONCURRENT statement
                         if (i6==10) exit mytest1
                       end do nc6
                     end do nc5
                   end do nc4
                 end do nc3
               end do nc2
             end do nc1
           end if mytest1
end subroutine do_concurrent_test1

subroutine do_concurrent_test2(n)
  implicit none
  integer :: i1,i2,i3,i4,i5,i6,n
  mytest2: if (n>0) then
  nc1:       do concurrent(i1=1:n)
  nc2:         do i2=1,n
  nc3:           do concurrent(i3=1:n)
  nc4:             do i4=1,n
  nc5:               do concurrent(i5=1:n)
  nc6:                 do i6=1,n
!ERROR: EXIT must not leave a DO CONCURRENT statement
!ERROR: EXIT must not leave a DO CONCURRENT statement
                         if (i6==10) exit nc3
                       end do nc6
                     end do nc5
                   end do nc4
                 end do nc3
               end do nc2
             end do nc1
           end if mytest2
end subroutine do_concurrent_test2

subroutine do_concurrent_test3(n)
  implicit none
  integer :: i1,i2,i3,i4,i5,i6,n
  mytest3: if (n>0) then
  nc1:       do concurrent(i1=1:n)
  nc2:         do i2=1,n
  nc3:           do concurrent(i3=1:n)
!ERROR: EXIT must not leave a DO CONCURRENT statement
                   if (i3==4) exit nc2
  nc4:             do i4=1,n
  nc5:               do concurrent(i5=1:n)
  nc6:                 do i6=1,n
                         if (i6==10) print *, "hello"
                       end do nc6
                     end do nc5
                   end do nc4
                 end do nc3
               end do nc2
             end do nc1
           end if mytest3
end subroutine do_concurrent_test3
