! RUN: %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK-NOT: exit from DO CONCURRENT construct

subroutine do_concurrent_test1(n)
  implicit none
  integer :: j,k,l,n
  mytest: if (n>0) then
  mydoc:    do concurrent(j=1:n)
              mydo: do k=1,n
                      if (k==5) exit
                      if (k==6) exit mydo
                    end do mydo
              do concurrent(l=1:n)
                if (l==5) print *, "test"
              end do
            end do mydoc
            do k=1,n
              if (k==5) exit mytest
            end do
          end if mytest
end subroutine do_concurrent_test1

subroutine do_concurrent_test2(n)
  implicit none
  integer :: i1,i2,i3,i4,i5,i6,n
  mytest2: if (n>0) then
  nc1:       do concurrent(i1=1:n)
  nc2:         do i2=1,n
  nc3:           do concurrent(i3=1:n)
  nc4:             do i4=1,n
                     if (i3==4) exit nc4
  nc5:               do concurrent(i5=1:n)
  nc6:                 do i6=1,n
                         if (i6==10) print *, "hello"
                       end do nc6
                     end do nc5
                   end do nc4
                 end do nc3
               end do nc2
             end do nc1
           end if mytest2
end subroutine do_concurrent_test2
