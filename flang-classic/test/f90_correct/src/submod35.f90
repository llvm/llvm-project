!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
module first_submod35
character (len = 10) :: word
integer :: arr(2:7, 3:18)
integer :: arr2(4:17)
interface
    module subroutine printWord2
    end subroutine
    module subroutine printWord3
    end subroutine
    module subroutine printWord4
    end subroutine
    module subroutine printWord5
    end subroutine
end interface
contains
    subroutine printWord1
        print *, word;
    end subroutine
end module first_submod35

submodule (first_submod35) second_submod35
character (len = 10) word2
contains
   module procedure printWord2
        call set
        print *, word, word2
        print *, "shape of arr is ", shape(arr)
    end procedure
    subroutine set
        word2 = "second"
    end subroutine
end submodule second_submod35

submodule (first_submod35:second_submod35) third_submod35
character (len = 10) word3
contains
   module procedure printWord3
        call set
        print *, word, word2, word3
        print *, "lbounds of arr are: ", lbound(arr)
    end procedure
    subroutine set
        word3 = "third"
    end subroutine

end submodule third_submod35

submodule (first_submod35:third_submod35) fourth_submod35
character (len = 10) word4
contains
   module procedure printWord4
        call set
        print *, word, word2, word3,  word4
        print *, "ubounds of arr are: ", ubound(arr)
        ! check the upper bounds of arr in the fourth submodule
        if ( .NOT. ((ubound(arr, dim=1) .EQ. 7) .AND. (ubound(arr, dim=2) .EQ. 18) )) then
            print *, "TEST FAILED upper bounds incorrect"
        end if
    end procedure
    subroutine set
        word4 = "fourth"
    end subroutine

end submodule fourth_submod35

submodule (first_submod35:fourth_submod35) fifth_submod35
character (len = 10) word5
contains
   module procedure printWord5
        call set
        print *, word, word2, word3, word4, word5
        print *, "shape of result of shape of arr is: ", shape(shape(arr))
        if ( (word .EQ. "first") .AND. (word2 .EQ. "second") .AND. (word3 .EQ. "third") .AND. (word4 .EQ. "fourth") .AND. (word5 .EQ. "fifth") ) then
            print *, " PASS"
        else
            print *, "TEST FAILED submodule host association not working"
        end if
        ! check the upper bounds of arr in the fifth submodule
        if ( .NOT. ((ubound(arr, dim=1) .EQ. 7) .AND. (ubound(arr, dim=2) .EQ. 18) )) then
            print *, "TEST FAILED upper bounds incorrect"
        end if

    end procedure
    subroutine set
        word5 = "fifth"
    end subroutine
end submodule fifth_submod35

program dothis
use first_submod35
implicit none
word = 'first'

call printWord1
call printWord2
call printWord3
call printWord4
call printWord5
        ! check the upper bounds of arr in the main program
        if ( .NOT. ((ubound(arr, dim=1) .EQ. 7) .AND. (ubound(arr, dim=2) .EQ. 18) )) then
            print *, "TEST FAILED upper bounds incorrect"
        end if

end program dothis
