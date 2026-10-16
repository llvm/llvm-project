! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=52 %s -o - \
! RUN:   | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - \
! RUN:   | FileCheck %s

module scoring
contains
  subroutine cpu()
  end subroutine
  subroutine vendor()
  end subroutine

  subroutine base_sections()
    !$omp declare variant(cpu) match(device={kind(cpu)})
    !$omp declare variant(vendor) &
    !$omp& match(implementation={vendor(score(5): llvm)})
  end subroutine

  ! SECTION is a separator: the context depth is two, CPU scores 5, and the
  ! vendor scores 6 whether the optional first SECTION is present or absent.
  ! CHECK-LABEL: func.func @_QMscoringPtest_explicit_section()
  ! CHECK: omp.parallel
  ! CHECK: omp.sections
  ! CHECK: omp.section
  ! CHECK-NOT: fir.call @_QMscoringPcpu
  ! CHECK: fir.call @_QMscoringPvendor()
  ! CHECK-NOT: fir.call @_QMscoringPcpu
  ! CHECK: return
  subroutine test_explicit_section()
    !$omp parallel sections
      !$omp section
        call base_sections()
    !$omp end parallel sections
  end subroutine

  ! CHECK-LABEL: func.func @_QMscoringPtest_implicit_section()
  ! CHECK: omp.parallel
  ! CHECK: omp.sections
  ! CHECK: omp.section
  ! CHECK-NOT: fir.call @_QMscoringPcpu
  ! CHECK: fir.call @_QMscoringPvendor()
  ! CHECK-NOT: fir.call @_QMscoringPcpu
  ! CHECK: return
  subroutine test_implicit_section()
    !$omp parallel sections
      call base_sections()
    !$omp end parallel sections
  end subroutine
end module
