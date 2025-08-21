! Check that the pie/pic/PIE/PIC macros are defined properly through the frontend driver

! RUN: %flang_fc1 -dM -E -o - %s \
! RUN:   | FileCheck %s
! CHECK-NOT: #define __PIC__
! CHECK-NOT: #define __PIE__
! CHECK-NOT: #define __pic__
! CHECK-NOT: #define __pie__
!
! RUN: %flang_fc1 -pic-level 1 -dM -E -o - %s \
! RUN:   | FileCheck --check-prefix=CHECK-PIC1 %s
! CHECK-PIC1: #define __PIC__ 1
! CHECK-PIC1-NOT: #define __PIE__
! CHECK-PIC1: #define __pic__ 1
! CHECK-PIC1-NOT: #define __pie__
!
! RUN: %flang_fc1 -pic-level 2 -dM -E -o - %s \
! RUN:   | FileCheck --check-prefix=CHECK-PIC2 %s
! CHECK-PIC2: #define __PIC__ 2
! CHECK-PIC2-NOT: #define __PIE__
! CHECK-PIC2: #define __pic__ 2
! CHECK-PIC2-NOT: #define __pie__
!
! RUN: %flang_fc1 -pic-level 1 -pic-is-pie -dM -E -o - %s \
! RUN:   | FileCheck --check-prefix=CHECK-PIE1 %s
! CHECK-PIE1: #define __PIC__ 1
! CHECK-PIE1: #define __PIE__ 1
! CHECK-PIE1: #define __pic__ 1
! CHECK-PIE1: #define __pie__ 1
!
! RUN: %flang_fc1 -pic-level 2 -pic-is-pie -dM -E -o - %s \
! RUN:   | FileCheck --check-prefix=CHECK-PIE2 %s
! CHECK-PIE2: #define __PIC__ 2
! CHECK-PIE2: #define __PIE__ 2
! CHECK-PIE2: #define __pic__ 2
! CHECK-PIE2: #define __pie__ 2

integer, parameter :: pic_level = __pic__
