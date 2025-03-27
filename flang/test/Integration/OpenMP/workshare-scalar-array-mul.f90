!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix HLFIR-O3
!RUN: %flang_fc1 -emit-fir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix FIR-O3

!RUN: %flang_fc1 -emit-hlfir -fopenmp -O0 %s -o - | FileCheck %s --check-prefix HLFIR-O0
!RUN: %flang_fc1 -emit-fir -fopenmp -O0 %s -o - | FileCheck %s --check-prefix FIR-O0

program test
  real :: arr_01(10)
  !$omp parallel workshare
    arr_01 = arr_01*2
  !$omp end parallel workshare
end program

! HLFIR-O3:    omp.parallel {
! HLFIR-O3:      omp.workshare {
! HLFIR-O3:        hlfir.elemental
! HLFIR-O3:        hlfir.assign
! HLFIR-O3:        hlfir.destroy
! HLFIR-O3:        omp.terminator
! HLFIR-O3:      omp.terminator

! FIR-O3:    omp.parallel {
! FIR-O3:      omp.wsloop nowait {
! FIR-O3:        omp.loop_nest
! FIR-O3:      omp.barrier
! FIR-O3:      omp.terminator

! HLFIR-O0:    omp.parallel {
! HLFIR-O0:      omp.workshare {
! HLFIR-O0:        hlfir.elemental
! HLFIR-O0:        hlfir.assign
! HLFIR-O0:        hlfir.destroy
! HLFIR-O0:        omp.terminator
! HLFIR-O0:      omp.terminator

! Check the copyprivate copy function
! FIR-O0:  func.func private @_workshare_copy_heap_{{.*}}(%[[DST:.*]]: {{.*}}, %[[SRC:.*]]: {{.*}})
! FIR-O0:    fir.load %[[SRC]]
! FIR-O0:    fir.store {{.*}} to %[[DST]]

! Check that we properly handle the temporary array
! FIR-O0:    omp.parallel {
! FIR-O0:      %[[CP:.*]] = fir.alloca !fir.heap<!fir.array<10xf32>>
! FIR-O0:      omp.single copyprivate(%[[CP]] -> @_workshare_copy_heap_
! FIR-O0:        fir.allocmem
! FIR-O0:        fir.store
! FIR-O0:        omp.terminator
! FIR-O0:      fir.load %[[CP]]
! FIR-O0:      omp.wsloop {
! FIR-O0:        omp.loop_nest
! FIR-O0:          omp.yield
! FIR-O0:      omp.single nowait {
! FIR-O0:        fir.call @_FortranAAssign
! FIR-O0:        fir.freemem
! FIR-O0:        omp.terminator
! FIR-O0:      omp.barrier
! FIR-O0:      omp.terminator
