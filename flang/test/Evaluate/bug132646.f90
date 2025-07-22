! RUN: %flang %s -o %t && %t | FileCheck %s

program main
    print *,kind(max0(1_1,2_1))
    print *,kind(max0(1_2,2_2))
    print *,kind(max0(1_4,2_4))
    print *,kind(max0(1_8,2_8))

    print *,kind(min0(1_1,2_1))
    print *,kind(min0(1_2,2_2))
    print *,kind(min0(1_4,2_4))
    print *,kind(min0(1_8,2_8))
end program main

! CHECK: 1
! CHECK-NEXT: 2
! CHECK-NEXT: 4
! CHECK-NEXT: 8

! CHECK-NEXT: 1
! CHECK-NEXT: 2
! CHECK-NEXT: 4
! CHECK-NEXT: 8

