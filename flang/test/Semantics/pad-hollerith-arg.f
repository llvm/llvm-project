! RUN: %flang_fc1 -fdebug-unparse %s | FileCheck %s
! Ensure that Hollerith actual arguments are blank padded.
! CHECK: CALL foo("abc     ")
      call foo(3habc)
      end
