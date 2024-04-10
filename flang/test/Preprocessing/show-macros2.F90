! RUN: %flang -DFOO -DBAR=FOO -dM -E -o - %s | FileCheck %s

! Check command line definitions

! CHECK: #define BAR FOO
! CHECK: #define FOO 1
