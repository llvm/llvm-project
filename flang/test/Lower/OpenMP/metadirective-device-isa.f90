! Test metadirective with device={isa(...)} trait selectors.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -target-feature +neon %s -o - | FileCheck --check-prefix=NEON %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -target-feature +neon -target-feature +sve %s -o - | FileCheck --check-prefix=SVE %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -target-feature +sse %s -o - | FileCheck --check-prefix=SSE %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -target-feature +avx %s -o - | FileCheck --check-prefix=AVX %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o - | FileCheck --check-prefix=NONE %s

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 -target-feature +neon %s -o - | FileCheck --check-prefix=NEON %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 -target-feature +neon -target-feature +sve %s -o - | FileCheck --check-prefix=SVE %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 -target-feature +sse %s -o - | FileCheck --check-prefix=SSE %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 -target-feature +avx %s -o - | FileCheck --check-prefix=AVX %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 %s -o - | FileCheck --check-prefix=NONE %s

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 -target-feature +neon %s -o - | FileCheck --check-prefix=NEON %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 -target-feature +neon -target-feature +sve %s -o - | FileCheck --check-prefix=SVE %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 -target-feature +sse %s -o - | FileCheck --check-prefix=SSE %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 -target-feature +avx %s -o - | FileCheck --check-prefix=AVX %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 %s -o - | FileCheck --check-prefix=NONE %s

! NEON-LABEL: func.func @_QPtest_isa_neon()
! NEON:         omp.barrier
! SVE-LABEL: func.func @_QPtest_isa_neon()
! SVE:         omp.barrier
! SSE-LABEL: func.func @_QPtest_isa_neon()
! SSE-NOT:     omp.barrier
! SSE:         return
! AVX-LABEL: func.func @_QPtest_isa_neon()
! AVX-NOT:     omp.barrier
! AVX:         return
! NONE-LABEL: func.func @_QPtest_isa_neon()
! NONE-NOT:     omp.barrier
! NONE:         return
subroutine test_isa_neon()
  !$omp metadirective &
  !$omp & when(device={isa("neon")}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! NEON-LABEL: func.func @_QPtest_isa_sve()
! NEON-NOT:     omp.barrier
! NEON:         return
! SVE-LABEL: func.func @_QPtest_isa_sve()
! SVE:         omp.barrier
! SSE-LABEL: func.func @_QPtest_isa_sve()
! SSE-NOT:     omp.barrier
! SSE:         return
! AVX-LABEL: func.func @_QPtest_isa_sve()
! AVX-NOT:     omp.barrier
! AVX:         return
! NONE-LABEL: func.func @_QPtest_isa_sve()
! NONE-NOT:     omp.barrier
! NONE:         return
subroutine test_isa_sve()
  !$omp metadirective &
  !$omp & when(device={isa("sve")}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! NEON-LABEL: func.func @_QPtest_isa_sse()
! NEON-NOT:     omp.barrier
! NEON:         return
! SVE-LABEL: func.func @_QPtest_isa_sse()
! SVE-NOT:     omp.barrier
! SVE:         return
! SSE-LABEL: func.func @_QPtest_isa_sse()
! SSE:         omp.barrier
! AVX-LABEL: func.func @_QPtest_isa_sse()
! AVX-NOT:     omp.barrier
! AVX:         return
! NONE-LABEL: func.func @_QPtest_isa_sse()
! NONE-NOT:     omp.barrier
! NONE:         return
subroutine test_isa_sse()
  !$omp metadirective &
  !$omp & when(device={isa("sse")}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! NEON-LABEL: func.func @_QPtest_isa_avx()
! NEON-NOT:     omp.barrier
! NEON:         return
! SVE-LABEL: func.func @_QPtest_isa_avx()
! SVE-NOT:     omp.barrier
! SVE:         return
! SSE-LABEL: func.func @_QPtest_isa_avx()
! SSE-NOT:     omp.barrier
! SSE:         return
! AVX-LABEL: func.func @_QPtest_isa_avx()
! AVX:         omp.barrier
! NONE-LABEL: func.func @_QPtest_isa_avx()
! NONE-NOT:     omp.barrier
! NONE:         return
subroutine test_isa_avx()
  !$omp metadirective &
  !$omp & when(device={isa("avx")}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! NEON-LABEL: func.func @_QPtest_isa_multi_when()
! NEON:         omp.barrier
! NEON-NOT:     omp.taskwait
! SVE-LABEL: func.func @_QPtest_isa_multi_when()
! SVE:         omp.barrier
! SVE-NOT:     omp.taskwait
! SSE-LABEL: func.func @_QPtest_isa_multi_when()
! SSE-NOT:     omp.barrier
! SSE:         omp.taskwait
! AVX-LABEL: func.func @_QPtest_isa_multi_when()
! AVX-NOT:     omp.barrier
! AVX-NOT:     omp.taskwait
! AVX:         return
! NONE-LABEL: func.func @_QPtest_isa_multi_when()
! NONE-NOT:     omp.barrier
! NONE-NOT:     omp.taskwait
! NONE:         return
subroutine test_isa_multi_when()
  !$omp metadirective &
  !$omp & when(device={isa("neon")}: barrier) &
  !$omp & when(device={isa("sse")}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! NEON-LABEL: func.func @_QPtest_isa_no_match_default()
! NEON:         omp.barrier
! SVE-LABEL: func.func @_QPtest_isa_no_match_default()
! SVE:         omp.barrier
! SSE-LABEL: func.func @_QPtest_isa_no_match_default()
! SSE:         omp.barrier
! AVX-LABEL: func.func @_QPtest_isa_no_match_default()
! AVX:         omp.barrier
! NONE-LABEL: func.func @_QPtest_isa_no_match_default()
! NONE:         omp.barrier
subroutine test_isa_no_match_default()
  !$omp metadirective &
  !$omp & when(device={isa("sve2")}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(barrier)
#else
  !$omp & default(barrier)
#endif
end subroutine

! Test device={arch()} selector. Arch properties without a known TraitProperty
! mapping are matched against target features, like ISA traits.

! NEON-LABEL: func.func @_QPtest_arch_neon()
! NEON:         omp.barrier
! SVE-LABEL: func.func @_QPtest_arch_neon()
! SVE:         omp.barrier
! SSE-LABEL: func.func @_QPtest_arch_neon()
! SSE-NOT:     omp.barrier
! SSE:         return
! AVX-LABEL: func.func @_QPtest_arch_neon()
! AVX-NOT:     omp.barrier
! AVX:         return
! NONE-LABEL: func.func @_QPtest_arch_neon()
! NONE-NOT:     omp.barrier
! NONE:         return
subroutine test_arch_neon()
  !$omp metadirective &
  !$omp & when(device={arch("neon")}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! NEON-LABEL: func.func @_QPtest_arch_no_match()
! NEON-NOT:     omp.barrier
! NEON:         omp.taskwait
! SVE-LABEL: func.func @_QPtest_arch_no_match()
! SVE-NOT:     omp.barrier
! SVE:         omp.taskwait
! SSE-LABEL: func.func @_QPtest_arch_no_match()
! SSE-NOT:     omp.barrier
! SSE:         omp.taskwait
! AVX-LABEL: func.func @_QPtest_arch_no_match()
! AVX-NOT:     omp.barrier
! AVX:         omp.taskwait
! NONE-LABEL: func.func @_QPtest_arch_no_match()
! NONE-NOT:     omp.barrier
! NONE:         omp.taskwait
subroutine test_arch_no_match()
  !$omp metadirective &
  !$omp & when(device={arch("unknown_arch")}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(taskwait)
#else
  !$omp & default(taskwait)
#endif
end subroutine
