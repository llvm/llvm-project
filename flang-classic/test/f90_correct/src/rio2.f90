!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!


         function foo(i, writetome, numme)
           integer foo,i,numme
           integer omp_get_thread_num, omp_get_num_threads
           character*20 writetome
! it should overwrite the first position or value of 'a' that a caller has already wrote
           write(writetome, '(i1)') omp_get_thread_num()
           numme = omp_get_thread_num()
           foo=i+3
         end

         program mm
         use omp_lib
         integer a,b,foo,numme
         character*20 writetome
         a=1
         b=4
         call omp_set_num_threads(9)
!$omp parallel
         write(writetome, '(i1, i1, i1)') a,foo(2,writetome,numme),b
!$omp end parallel
!         print *, writetome,numme
! 'numme' should have the same value as the first value in writetome

         read(writetome, '(i1)') b      

         if (numme .eq. b) then
             print *, "PASS" 
         else
             print *, "FAIL"
         end if

         end
