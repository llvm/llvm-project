! RUN: %flang -E -dM - < /dev/null | FileCheck %s

! Check that dumping macros from stdin still emits default definitions.

! CHECK: #define __DATE__
! CHECK: #define __TIME__
! CHECK: #define __flang__
! CHECK: #define __flang_major__
! CHECK: #define __flang_minor__
! CHECK: #define __flang_patchlevel__
