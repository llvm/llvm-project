! RUN: not %flang-new  %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang-new  -emit-obj %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang-new  -fc1 -emit-obj %s 2>&1 | FileCheck %s --check-prefix=ERROR-FC1

! REQUIRES: new-flang-driver

! By default (e.g. when no options like `-E` are passed) flang-new
! creates a job that corresponds to `-emit-obj`. This option/action is
! not yet supported. Verify that this is correctly reported as error.

! ERROR: error: unknown argument: '-triple'
! ERROR: error: unknown argument: '-emit-obj'

! ERROR-FC1: error: unknown argument: '-emit-obj'
