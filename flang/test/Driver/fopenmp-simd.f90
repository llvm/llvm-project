! RUN: %flang -target x86_64-linux-gnu -fopenmp-simd %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-OPENMP-SIMD-FLAG --check-prefix=CHECK-NO-LD-ANY
! RUN: %flang -target x86_64-darwin -fopenmp-simd %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-OPENMP-SIMD-FLAG --check-prefix=CHECK-NO-LD-ANY
! RUN: %flang -target x86_64-freebsd -fopenmp-simd %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-OPENMP-SIMD-FLAG --check-prefix=CHECK-NO-LD-ANY
! RUN: %flang -target x86_64-windows-gnu -fopenmp-simd %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-OPENMP-SIMD-FLAG --check-prefix=CHECK-NO-LD-ANY

! CHECK-OPENMP-SIMD-FLAG: "-fopenmp-simd"
! CHECK-NO-LD-ANY-NOT: "-l{{(omp|gomp|iomp5)}}"

! -fopenmp-simd enables openmp support only for simd constructs
! RUN: %flang_fc1 -fopenmp-simd %s -emit-fir -o - | FileCheck --check-prefix=CHECK-OMP-SIMD %s
! RUN: %flang_fc1 -fno-openmp-simd %s -emit-fir -o - | FileCheck --check-prefix=CHECK-NO-OMP-SIMD %s
! RUN: %flang_fc1 -fopenmp-simd -fno-openmp-simd %s -emit-fir -o - | FileCheck --check-prefix=CHECK-NO-OMP-SIMD %s
! RUN: %flang_fc1 -fno-openmp-simd -fopenmp-simd %s -emit-fir -o - | FileCheck --check-prefix=CHECK-OMP-SIMD %s
! -fopenmp-simd should have no effect if -fopenmp is already set
! RUN: %flang_fc1 -fopenmp %s -emit-fir -o - | FileCheck --check-prefix=CHECK-OMP %s
! RUN: %flang_fc1 -fopenmp -fopenmp-simd %s -emit-fir -o - | FileCheck --check-prefix=CHECK-OMP %s
! RUN: %flang_fc1 -fopenmp -fno-openmp-simd %s -emit-fir -o - | FileCheck --check-prefix=CHECK-OMP %s

subroutine main
  ! CHECK-OMP-SIMD-NOT: omp.parallel
  ! CHECK-OMP-SIMD-NOT: omp.wsloop
  ! CHECK-OMP-SIMD-NOT: omp.loop_nest
  ! CHECK-OMP-SIMD: fir.do_loop
  ! CHECK-NO-OMP-SIMD-NOT: omp.parallel
  ! CHECK-NO-OMP-SIMD-NOT: omp.wsloop
  ! CHECK-NO-OMP-SIMD-NOT: omp.loop_nest
  ! CHECK-NO-OMP-SIMD: fir.do_loop
  ! CHECK-OMP: omp.parallel
  ! CHECK-OMP: omp.wsloop
  ! CHECK-OMP: omp.loop_nest
  ! CHECK-OMP-NOT: fir.do_loop
  !$omp parallel do
  do i = 1, 10
    print *, "test"
  end do
  ! CHECK-NO-OMP-SIMD-NOT: omp.yield
  ! CHECK-NO-OMP-SIMD-NOT: omp.terminator
  ! CHECK-OMP-SIMD-NOT: omp.yield
  ! CHECK-OMP-SIMD-NOT: omp.terminator
  ! CHECK-OMP: omp.yield
  ! CHECK-OMP: omp.terminator
  !$omp end parallel do

  ! CHECK-OMP-SIMD: omp.simd
  ! CHECK-NO-OMP-SIMD-NOT: omp.simd
  ! CHECK-OMP: omp.simd
  !$omp simd
  ! CHECK-OMP-SIMD: omp.loop_nest
  ! CHECK-NO-OMP-SIMD-NOT: omp.loop_nest
  ! CHECK-NO-OMP-SIMD: fir.do_loop
  ! CHECK-OMP: omp.loop_nest
  ! CHECK-OMP-NOT: fir.do_loop
  do i = 1, 10
    print *, "test"
  ! CHECK-OMP-SIMD: omp.yield
  ! CHECK-NO-OMP-SIMD-NOT: omp.yield
  ! CHECK-OMP: omp.yield
  end do
end subroutine
