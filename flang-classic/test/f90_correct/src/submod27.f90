!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! C1411 submodule specification-part must not contain format-stmt

module prettyprint
    double precision A, B, C
  interface 
    module subroutine niceprint(e,f,g)
    double precision, intent(in) :: e,f,g
    end subroutine niceprint
  end interface
  contains
    
end module prettyprint

submodule (prettyprint) niceprint
   400 FORMAT('|','|',3(F8.3,'|'),'|') !{error "PGF90-S-0310-Illegal statement in the specification part of a MODULE"}
contains
  module procedure niceprint
  ! 400 FORMAT('|','|',3(F8.3,'|'),'|')
   500 FORMAT(6H PASS )
   write(*,400)e,f,g
   write(*,500)
  end procedure  
end submodule niceprint

program foo
use prettyprint
200 FORMAT(' ',3F7.2)
300 FORMAT('|',3(F8.3,'|'))
    A = 3.141592
    B = -11.2
    C = 12.34567E-02
write(*,200)A,B,C
write(*,300)A,B,C
write(*,300)B,C,A
write(*,300)C,A,B
call niceprint(A,B,C)

end program foo

