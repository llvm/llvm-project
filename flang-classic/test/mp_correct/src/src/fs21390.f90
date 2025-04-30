! Contributed by the University of Southampton

subroutine foo
implicit none
real, allocatable :: x(:)

allocate(x(1000))
x(:) = 0D0
write(*,*) '@1: ', allocated(x), loc(x)
call internal_of_foo
write(*,*) '@3: ', allocated(x), loc(x)

deallocate(x)

return

contains

subroutine internal_of_foo

integer :: i
real :: dummy

write(*,*) '@2: ', allocated(x), loc(x) 
!$OMP PARALLEL DO SHARED(x)
do i=1, 1000
dummy = dummy + x(i)
end do
!$OMP END PARALLEL DO
write(*,*) dummy

end subroutine internal_of_foo

end subroutine foo


program test
call foo
end program test

