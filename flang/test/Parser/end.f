! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: end.f:3:7: error: Program unit END statement may not be continued in fixed form source
      e
     + nd
! CHECK: end.f:6:7: error: Program unit END statement may not be continued in fixed form source
      end prog
     +        ram
! CHECK: end.f:9:7: error: Program unit END statement may not be continued in fixed form source
      end
     +       program
! CHECK: end.f:12:7: error: Program unit END statement may not be continued in fixed form source
      end
     +       program
     1                main
! CHECK: end.f:16:7: error: Program unit END statement may not be continued in fixed form source
      end program
     1            main
! CHECK: end.f:19:7: error: Initial line of continued statement must not appear to be a program unit END in fixed form source
      end
     +    = end + 1
! CHECK: end.f:22:7: error: Initial line of continued statement must not appear to be a program unit END in fixed form source
      end module
     +    = end module + 1
! CHECK-NOT: end.f:25:7: error: Initial line of continued statement must not appear to be a program unit END in fixed form source
      end =
     +      end + 1
! CHECK-NOT: end.f:28:7: error: Initial line of continued statement must not appear to be a program unit END in fixed form source
      end block data (
     +      1) = 666
