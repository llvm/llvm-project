!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! if a host allocatable is allocated in an internal
! procedure and that procedure is inlined, various compiler-created
! scalar temps associated with the allocatable may not be correctly
! defined
!
    LOGICAL, ALLOCATABLE :: TMPMASK (:,:)
    integer addr1(2), addr2(2), addr
    integer result(24)
    integer expect(24)
data expect/1,2,1,1,80,71,80,70,5600,5600,1,2,1,1,80,71,80,70,5600,5600,0,0,0,0/
    call testit()
    rewind(97)
    rewind(98)
    read(97,*) addr1
    read(97,*) result(1:4)
    read(97,*) result(5:8)
    read(97,*) result(9:10)
    read(98,*) addr2
    read(98,*) result(11:14)
    read(98,*) result(15:18)
    read(98,*) result(19:20)

    write(0,17) 'lb/ub ', result(1:8)
    write(0,17) 'lb/ub ', expect(1:8)
    write(0,17) ' size ', result(9:10)
    write(0,17) ' size ', expect(9:10)
    write(0,17) 'lb/ub ', result(11:18)
    write(0,17) 'lb/ub ', expect(11:18)
    write(0,17) ' size ', result(19:20)
    write(0,17) ' size ', expect(19:20)

17  format(a, 8i7)
    addr = loc(TMPMASK)
    expect(21:24) = addr
    result(21) = addr1(1)
    result(22) = addr1(2)
    result(23) = addr2(1)
    result(24) = addr2(2)
    write(0,18) ' addr ', result(21:24)
    write(0,18) ' addr ', expect(21:24)
18  format(a, 4i16)
    call check(result, expect, 24)
contains
subroutine a0_allocate()
    allocate (tmpmask(1:80,2:71))
    write (97,*), LOC(TMPMASK), LOC(TMPMASK(:,:))
    write (97,*), LBOUND(TMPMASK), LBOUND(TMPMASK(:,:))
    write (97,*), UBOUND(TMPMASK), UBOUND(TMPMASK(:,:))
    write (97,*), SIZE(TMPMASK), SIZE(TMPMASK(:,:))
end subroutine a0_allocate

subroutine testit()
    integer :: i, n = 1000
    call a0_allocate()
    write (98,*), LOC(TMPMASK), LOC(TMPMASK(:,:))
    write (98,*), LBOUND(TMPMASK), LBOUND(TMPMASK(:,:))
    write (98,*), UBOUND(TMPMASK), UBOUND(TMPMASK(:,:))
    write (98,*), SIZE(TMPMASK), SIZE(TMPMASK(:,:))
end subroutine
end
