! Test driver handling of -fexperimental-array-section-reduction and
! -fno-experimental-array-section-reduction.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fexperimental-array-section-reduction \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-experimental-array-section-reduction \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fno-experimental-array-section-reduction -fexperimental-array-section-reduction \
! RUN:   | FileCheck %s --check-prefix=ENABLED

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -fexperimental-array-section-reduction -fno-experimental-array-section-reduction \
! RUN:   | FileCheck %s --check-prefix=DISABLED

! DISABLED: "-fc1"
! DISABLED-NOT: "-fexperimental-array-section-reduction"

! ENABLED: "-fc1"
! ENABLED-SAME: "-fexperimental-array-section-reduction"

! Prove the flag actually inserts the pass into the pipeline. This is an MLIR
! HLFIR pass, so it is invisible to LLVM's -mllvm -print-pipeline-passes (that
! prints the LLVM pass pipeline, which is unchanged by an MLIR pass); dump the
! scheduled MLIR pass pipeline instead -- the same mechanism mlir-pass-pipeline.f90
! uses -- and check that ArraySectionReduction is listed only with the flag. The
! pass is scheduled at O1+, so -O2 is used. Requires asserts for --mlir-pass-statistics.

! RUN: %if asserts %{ %flang_fc1 -S -O2 -fexperimental-array-section-reduction \
! RUN:     -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline \
! RUN:     %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=INSERTED %}
! RUN: %if asserts %{ %flang_fc1 -S -O2 \
! RUN:     -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline \
! RUN:     %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ABSENT %}

! INSERTED: ArraySectionReduction
! ABSENT-NOT: ArraySectionReduction

end program
