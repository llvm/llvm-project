!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! just do recursive io calls to check that it does not hangs.

         function foo2(i)
             integer foo2,i
             foo2 = i
             print *, i
         end

         function foo(i)
           integer foo,foo2,i
           integer omp_get_thread_num, omp_get_num_threads
           character*20 writetome
           i = omp_get_thread_num()
           print *, i, foo2(i)
           foo=i+3
         end

         program mm
         integer a,b,foo,numme
         character*20 writetome
         integer omp_get_thread_num, omp_get_num_threads
         integer i
         a=1
         b=4
         i=2
!$omp parallel
         print *, "thread num:",omp_get_thread_num(),"num_thread:",omp_get_num_threads()
         write(writetome, '(i2)') foo(i)
!$omp end parallel


! if it gets here, then it is OK.  No hang.
         print *, "PASS" 

         end
