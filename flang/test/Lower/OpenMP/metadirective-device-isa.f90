! Test metadirective with device={isa(...)} trait selectors.

! RUN: %if aarch64-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -triple aarch64-unknown-linux-gnu -target-feature +neon %s -o - | FileCheck --check-prefix=NEON %s %}
! RUN: %if aarch64-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -triple aarch64-unknown-linux-gnu -target-feature +neon -target-feature +sve %s -o - | FileCheck --check-prefix=SVE %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -triple x86_64-unknown-linux-gnu -target-feature +sse %s -o - | FileCheck --check-prefix=SSE %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -triple x86_64-unknown-linux-gnu -target-feature +avx %s -o - | FileCheck --check-prefix=AVX %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 -triple x86_64-unknown-linux-gnu %s -o - | FileCheck --check-prefix=NONE %s %}

! RUN: %if aarch64-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 -triple aarch64-unknown-linux-gnu -target-feature +neon %s -o - | FileCheck --check-prefix=NEON %s %}
! RUN: %if aarch64-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 -triple aarch64-unknown-linux-gnu -target-feature +neon -target-feature +sve %s -o - | FileCheck --check-prefix=SVE %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 -triple x86_64-unknown-linux-gnu -target-feature +sse %s -o - | FileCheck --check-prefix=SSE %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 -triple x86_64-unknown-linux-gnu -target-feature +avx %s -o - | FileCheck --check-prefix=AVX %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 -triple x86_64-unknown-linux-gnu %s -o - | FileCheck --check-prefix=NONE %s %}

! RUN: %if aarch64-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 -triple aarch64-unknown-linux-gnu -target-feature +neon %s -o - | FileCheck --check-prefix=NEON %s %}
! RUN: %if aarch64-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 -triple aarch64-unknown-linux-gnu -target-feature +neon -target-feature +sve %s -o - | FileCheck --check-prefix=SVE %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 -triple x86_64-unknown-linux-gnu -target-feature +sse %s -o - | FileCheck --check-prefix=SSE %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 -triple x86_64-unknown-linux-gnu -target-feature +avx %s -o - | FileCheck --check-prefix=AVX %s %}
! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 -triple x86_64-unknown-linux-gnu %s -o - | FileCheck --check-prefix=NONE %s %}

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

! NEON-LABEL: func.func @_QPtest_begin_isa_neon()
! NEON:         omp.parallel
! SVE-LABEL: func.func @_QPtest_begin_isa_neon()
! SVE:         omp.parallel
! SSE-LABEL: func.func @_QPtest_begin_isa_neon()
! SSE-NOT:     omp.parallel
! SSE:         return
! AVX-LABEL: func.func @_QPtest_begin_isa_neon()
! AVX-NOT:     omp.parallel
! AVX:         return
! NONE-LABEL: func.func @_QPtest_begin_isa_neon()
! NONE-NOT:     omp.parallel
! NONE:         return
subroutine test_begin_isa_neon()
  integer :: x
  x = 0
#ifdef OMP_52
  !$omp begin metadirective &
  !$omp & when(device={isa("neon")}: parallel) &
  !$omp & otherwise(nothing)
#else
  !$omp begin metadirective &
  !$omp & when(device={isa("neon")}: parallel)
#endif
  x = 1
  !$omp end metadirective
end subroutine

! NEON-LABEL: func.func @_QPtest_begin_isa_sse()
! NEON-NOT:     omp.parallel
! NEON:         return
! SVE-LABEL: func.func @_QPtest_begin_isa_sse()
! SVE-NOT:     omp.parallel
! SVE:         return
! SSE-LABEL: func.func @_QPtest_begin_isa_sse()
! SSE:         omp.parallel
! AVX-LABEL: func.func @_QPtest_begin_isa_sse()
! AVX-NOT:     omp.parallel
! AVX:         return
! NONE-LABEL: func.func @_QPtest_begin_isa_sse()
! NONE-NOT:     omp.parallel
! NONE:         return
subroutine test_begin_isa_sse()
  integer :: x
  x = 0
#ifdef OMP_52
  !$omp begin metadirective &
  !$omp & when(device={isa("sse")}: parallel) &
  !$omp & otherwise(nothing)
#else
  !$omp begin metadirective &
  !$omp & when(device={isa("sse")}: parallel)
#endif
  x = 1
  !$omp end metadirective
end subroutine
