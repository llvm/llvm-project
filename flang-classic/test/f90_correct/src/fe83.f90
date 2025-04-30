!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!


program testnewline
parameter(N=20)
integer :: results(N)
integer :: expected(N)

character*5 c1, c2, c3
logical l1(3), l2(3)
integer*8 abc(20)
integer iosint(3)
integer myintread(20)
integer myintwrite(20)
       data results / -1, -1, 0, 0, -1, 0, 0, 0, &
                      -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0/
       data expected / -1, -1, 0, 0, -1, 0, 0, 0, &
                      -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0/



! new_line
open(28, file='t.txt', status='replace', access='stream', form='formatted')
write(28,'(a)') 'hello'//new_line('x')//'world'
close(28)
!

! end of file
open(28, file='t.txt', form='formatted')
read(28,100,iostat=ios) c1
l1(1) = is_iostat_end(ios)
l2(1) = is_iostat_eor(ios)
read(28,100,iostat=ios) c2
l1(2) = is_iostat_end(ios)
l2(2) = is_iostat_eor(ios)
read(28,100,iostat=ios) c3
l1(3) = is_iostat_end(ios)
l2(3) = is_iostat_eor(ios)
100 format(a5)
close(28)

results(1)=(c1 == 'hello') 
results(2)=(c2 == 'world') 
results(3)=l1(1)
results(4)=l1(2)
results(5)=l1(3)
results(6)=l2(1)
results(7)=l2(2)
results(8)=l2(3)



!end of record
open(28, file='t2.txt',status='replace',form='formatted')
write(28,'(i13)') myintwrite
close(28)

open(28, file='t2.txt', form='formatted')
read(28,200,advance='no',iostat=ios) myintread
200 format(i20)
close(28)

results(9)=ios
results(10)=is_iostat_end(ios)
results(11)=is_iostat_eor(ios)

open(28, file='t2.txt', form='formatted')
read(28,300,advance='no',iostat=ios) myintread
300 format(i13)
close(28)

results(12)=ios
results(13)=is_iostat_end(ios)
results(14)=is_iostat_eor(ios)

!testing array
iosint=0
iosint=is_iostat_end(iosint)
results(15:17)=iosint

iosint=is_iostat_eor(iosint)
results(18:20)=iosint


call check(results,expected,N)

end
