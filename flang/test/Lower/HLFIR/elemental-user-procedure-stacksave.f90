! Check that stack save and restore needed for elemental function result
! allocation inside loops are not emitted directly in lowering, but inserted if
! needed in the stack-reclaim pass.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefix=CHECK-HLFIR
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-LLVM
subroutine foo(c1, c2)
  character(*), dimension(100) :: c1, c2
  interface
    elemental pure function func(c)
      character(*), intent(in) :: c
      character(len(c)) :: func
    end function
  end interface
  c1 = func(c2)
end subroutine

! CHECK-HLFIR-NOT: stacksave
! CHECK: return

! CHECK-LLVM: stacksave
! CHECK-LLVM: stackrestore
