! RUN: %flang_fc1 -emit-fir -o /dev/null %s 2>&1 | FileCheck %s --allow-empty
! CHECK-NOT: error
! CHECK-NOT: ac-do-variable has no binding
!
! Regression test for https://github.com/llvm/llvm-project/issues/79850
! Ensure flang does not crash when compiling code that uses pointer components
! in structure constructors.

program p
  type child
    integer, pointer :: id
  end type

  integer, target :: t1(10)
  type(child) :: t2(10)

  t1 = 0
  t2 = (/ ( child(t1(i)), i=1,10 ) /)
end program
