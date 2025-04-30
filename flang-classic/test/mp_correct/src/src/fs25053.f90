! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program omp_bug 
integer, parameter :: cnt = 100 
integer, pointer :: bnd(:,:) 
integer, pointer :: chamber(:) 
integer :: i, c 

allocate(bnd(1:2,cnt)) 
do i = 1, cnt 
bnd(1:2, i) = (/ i, i+1 /) 
end do 

!print *, "bnd:", bnd



allocate(chamber(1:cnt+1), source=0) 

!$omp parallel do private(i) 
do i = 1,cnt 
chamber(bnd(1:2,i)) = 1 
end do 
!$omp end parallel do 

do i = 1, cnt+1 
c = chamber(i) 
if (c <= 0) then 
print *, "FAIL"
call abort() 
end if 
end do 

print *, "PASS"

end 
