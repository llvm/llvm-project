! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: error: end of file
! CHECK: ^
! CHECK: in the context: END PROGRAM statement
! CHECK: in the context: main program

  integer :: i

  ! Add empty lines for emphasis

  i = 5
