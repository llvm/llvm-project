! Regression test for https://github.com/llvm/llvm-project/issues/178821

! Verify that the stack-arrays pass does not move a stack allocation
! for a temporary outside its stacksave/stackrestore scope when the
! size operand is shared across scopes (e.g. due to ConvertHLFIRtoFIR
! reusing a single size value, or CSE merging duplicates).

! RUN: %flang_fc1 -emit-fir -fstack-arrays %s -o - \
! RUN:   | fir-opt --stack-arrays \
! RUN:   | FileCheck %s

subroutine ss1(a)
  character*(*) a(1)
  character*1 t_s
  call u(t_s() // '?' // a)
  call u(t_s() // '?' // a)
end subroutine

! CHECK-LABEL: func.func @_QPss1
!
! First scope:
! CHECK:         llvm.intr.stacksave
! CHECK:         fir.alloca !fir.array<1x!fir.char<1,?>>
! CHECK-SAME:      {bindc_name = ".tmp.array"}
! CHECK:         fir.call @_QPu(
! CHECK:         llvm.intr.stackrestore
!
! Second scope (the second alloca must be AFTER the first stackrestore):
! CHECK:         llvm.intr.stacksave
! CHECK:         fir.alloca !fir.array<1x!fir.char<1,?>>
! CHECK-SAME:      {bindc_name = ".tmp.array"}
! CHECK:         fir.call @_QPu(
! CHECK:         llvm.intr.stackrestore
