! Check that the pie/pic/PIE/PIC macros are defined properly through the compiler driver

! RUN: %flang -fpic -dM -E -o - %s \
! RUN:   | FileCheck --check-prefix=CHECK-PIC1 %s
! CHECK-PIC1: #define __PIC__ 1
! CHECK-PIC1-NOT: #define __PIE__
! CHECK-PIC1: #define __pic__ 1
! CHECK-PIC1-NOT: #define __pie__
!
! RUN: %flang -fPIC -dM -E -o - %s \
! RUN:   | FileCheck --check-prefix=CHECK-PIC2 %s
! CHECK-PIC2: #define __PIC__ 2
! CHECK-PIC2-NOT: #define __PIE__
! CHECK-PIC2: #define __pic__ 2
! CHECK-PIC2-NOT: #define __pie__
!
! RUN: %flang -fpie -dM -E -o - %s \
! RUN:   | FileCheck --check-prefix=CHECK-PIE1 %s
! CHECK-PIE1: #define __PIC__ 1
! CHECK-PIE1: #define __PIE__ 1
! CHECK-PIE1: #define __pic__ 1
! CHECK-PIE1: #define __pie__ 1
!
! RUN: %flang -fPIE -dM -E -o - %s \
! RUN:   | FileCheck --check-prefix=CHECK-PIE2 %s
! CHECK-PIE2: #define __PIC__ 2
! CHECK-PIE2: #define __PIE__ 2
! CHECK-PIE2: #define __pic__ 2
! CHECK-PIE2: #define __pie__ 2

integer, parameter :: pic_level = __pic__
