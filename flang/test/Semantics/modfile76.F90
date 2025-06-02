!RUN: %flang -c -fhermetic-module-files -DWHICH=1 %s && %flang -c -fhermetic-module-files -DWHICH=2 %s && %flang -c -fhermetic-module-files %s && cat modfile76c.mod | FileCheck %s

#if WHICH == 1
module modfile76a
  integer :: global_variable = 0
end
#elif WHICH == 2
module modfile76b
  use modfile76a
 contains
  subroutine test
  end
end
#else
module modfile76c
  use modfile76a
  use modfile76b
end
#endif

!CHECK: module modfile76c
!CHECK: module modfile76a
!CHECK: !dir$ begin_nested_hermetic_module
!CHECK: module modfile76b
!CHECK: module modfile76a
!CHECK: !dir$ end_nested_hermetic_module
