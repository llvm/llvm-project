! RUN: %flang -dM -E -o - %s | FileCheck %s

! Check the default macros. Omit certain ones such as __LINE__
! or __FILE__, or target-specific ones, like __x86_64__.

! Macros are printed in the alphabetical order.

! CHECK: #define __DATE__
! CHECK: #define __TIME__
! CHECK: #define __flang__
! CHECK: #define __flang_major__
! CHECK: #define __flang_minor__
! CHECK: #define __flang_patchlevel__

