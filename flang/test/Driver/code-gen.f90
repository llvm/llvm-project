! Although code-generation is not yet available, we do have frontend actions
! that correspond to `-c` and `-emit-obj`. For now these actions are just a
! placeholder and running them leads to a driver error. This test makes sure
! that these actions are indeed run (rather than `-c` or `-emit-obj` being
! rejected earlier).
! TODO: Replace this file with a proper test once code-generation is available.

!-----------
! RUN LINES
!-----------
! RUN: not %flang %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang -c %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang -emit-obj %s 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: not %flang -fc1 -emit-obj %s 2>&1 | FileCheck %s --check-prefix=ERROR

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! ERROR: code-generation is not available yet
