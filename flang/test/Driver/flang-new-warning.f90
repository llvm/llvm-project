! RUN: amdflang-new -c %s 2>&1 | FileCheck %s
! CHECK: warning: the 'amdflang-new' and 'flang-new' commmands have been deprecated; please use 'amdflang' instead
! XFAIL: *
