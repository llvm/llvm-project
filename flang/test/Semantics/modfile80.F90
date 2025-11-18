!RUN: %flang_fc1 -DPART1 %s
!RUN: %flang_fc1 -DPART2 -fhermetic-module-files %s
!RUN: %flang_fc1 -DPART3 | FileCheck --allow-empty %s
!CHECK-NOT: error:

#if defined PART1
module modfile80a
  interface generic
    module procedure specific
  end interface
 contains
  subroutine specific
  end
end
#elif defined PART2
module modfile80b
  use modfile80a
end
#else
program test
  use modfile80a
  use modfile80b
  call generic
end
#endif
