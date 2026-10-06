! Verify that the default pipeline translates the lowering marker only at
! optimization levels that optimize for speed, and that llvm.readonly reaches
! the final LLVM IR.
! force-no-alias currently defaults to true, so llvm.noalias is expected at
! both levels; llvm.nocapture and llvm.readonly are the attributes gated here.
!
! RUN: %flang_fc1 -emit-llvm -O0 \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=function-attr \
! RUN:   -mmlir --mlir-print-ir-module-scope %s -o /dev/null 2>&1 \
! RUN:   | FileCheck %s --check-prefix=O0
! RUN: %flang_fc1 -emit-llvm -O1 \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=function-attr \
! RUN:   -mmlir --mlir-print-ir-module-scope %s -o /dev/null 2>&1 \
! RUN:   | FileCheck %s --check-prefix=O1
! RUN: %flang_fc1 -emit-llvm -O1 %s -o - | FileCheck %s --check-prefix=LLVM

module readonly_pipeline_mod
contains
  subroutine readonly_pipeline(x, y)
    integer, intent(in) :: x
    integer, intent(inout) :: y
    y = x
  end subroutine

  subroutine pointer_descriptor_readonly(p)
    integer, intent(in), pointer :: p
    p = 42
  end subroutine
end module

! O0-LABEL: func.func @_QMreadonly_pipeline_modPreadonly_pipeline(
! O0-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.read_only, llvm.noalias},
! O0-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "y", llvm.noalias})

! O1-LABEL: func.func @_QMreadonly_pipeline_modPreadonly_pipeline(
! O1-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "x", fir.read_only, llvm.noalias, llvm.nocapture, llvm.readonly},
! O1-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "y", llvm.noalias, llvm.nocapture})

! LLVM-LABEL: define void @_QMreadonly_pipeline_modPreadonly_pipeline(
! LLVM-SAME:    ptr {{.*}}readonly{{.*}} %0,

! O1-LABEL: func.func @_QMreadonly_pipeline_modPpointer_descriptor_readonly(%{{.*}}: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "p", fir.read_only, llvm.nocapture, llvm.readonly})

! LLVM-LABEL: define void @_QMreadonly_pipeline_modPpointer_descriptor_readonly(
! LLVM-SAME:    ptr {{.*}}readonly{{.*}} %0
! LLVM:         store i32 42
