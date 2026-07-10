! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefix=HLFIR
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
! Test from Fortran source through to LLVM IR.
! UNSUPPORTED: system-windows
! Disabled on 32-bit targets due to the additional `trunc` opcodes required
! UNSUPPORTED: target-x86
! UNSUPPORTED: target=sparc-{{.*}}
! UNSUPPORTED: target=sparcel-{{.*}}

! Assumed size array of assumed length character.
program test
  character :: x(3) = (/ '1','2','3' /)
  call sub(x)
contains
  subroutine sub(x)
    character(*) x(:)
    forall (i=1:2)
       x(i:i)(1:1) = x(i+1:i+1)(1:1)
    end forall
    print *,x
  end subroutine sub
end program test

! HLFIR-LABEL: func.func private @_QFPsub(
! HLFIR: hlfir.forall lb {
! HLFIR: hlfir.yield %{{.*}} : i32
! HLFIR: } ub {
! HLFIR: hlfir.yield %{{.*}} : i32
! HLFIR: }  (%[[I_ARG:.*]]: i32) {
! HLFIR:   %[[I_REF:.*]] = hlfir.forall_index "i" %[[I_ARG]]
! HLFIR:   hlfir.region_assign {
! HLFIR:     hlfir.designate {{.*}} substr {{.*}}
! HLFIR:     hlfir.yield %{{.*}}
! HLFIR:   } to {
! HLFIR:     hlfir.designate {{.*}} substr {{.*}}
! HLFIR:     hlfir.yield %{{.*}}
! HLFIR:   }

! LLVM-LABEL: define internal void @_QFPsub(
! LLVM: call ptr @_FortranACreateValueStack
! LLVM: call void @_FortranAPushValue
! LLVM: call void @_FortranAValueAt
! LLVM: call void @_FortranAAssign
! LLVM: call void @_FortranADestroyValueStack
