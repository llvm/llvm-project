!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang -c -fhermetic-module-files -DWHICH=1 -J%t %s && %flang -c -fhermetic-module-files -DWHICH=2 -J%t %s && %flang_fc1 -fdebug-unparse -J%t %s | FileCheck %s

#if WHICH == 1
module modfile75a
  use iso_c_binding
end
#elif WHICH == 2
module modfile75b
  use modfile75a
end
#else
program test
  use modfile75b
!CHECK: INTEGER(KIND=4_4) n
  integer(c_int) n
end
#endif
