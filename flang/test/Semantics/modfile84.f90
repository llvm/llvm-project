!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang_fc1 -fsyntax-only -J%t %S/Inputs/modfile84.f90
!RUN: not %flang_fc1 -fsyntax-only -J%t %s 2>&1 | FileCheck %s

!CHECK: error: Reference to 'foo' is ambiguous
!CHECK: 'foo' was use-associated from module 'modfile84a'
!CHECK: 'foo' was use-associated from module 'modfile84b'
!CHECK: error: 'foo' is not a callable procedure
!CHECK: 'foo' is USE-associated with 'foo' in module 'modfile84ab'
!CHECK: error: Reference to 'bar' is ambiguous
!CHECK: 'bar' was use-associated from module 'modfile84a'
!CHECK: 'bar' was use-associated from module 'modfile84b'
!CHECK: error: 'bar' is not a callable procedure
!CHECK: 'bar' is USE-associated with 'bar' in module 'modfile84ab'

use modfile84AB
call foo()
call bar()
end
