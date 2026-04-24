! Test handling of OPTIONAL in data directives
! RUN: %flang_fc1 -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine test(x)
 real, optional :: x(100)
 !$acc parallel firstprivate(x)
 call foo(x)
 !$acc end parallel
end subroutine

! CHECK: acc.firstprivate.recipe @firstprivatization_optional_ref_100xf32

! TODO: generate conditional copy
