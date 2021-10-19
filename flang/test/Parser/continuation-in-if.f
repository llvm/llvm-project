! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! CHECK: CALL foo("N","N")
#ifdef transpose
      call foo('T',
#else
      call foo('N',
#endif
     $     'N')
      end
