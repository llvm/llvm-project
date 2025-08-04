!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix HLFIR
!RUN: %flang_fc1 -emit-fir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix FIR

subroutine sb1(a, x, y, z)
  integer :: a
  integer :: x(:)
  integer :: y(:)
  integer, allocatable :: z(:)
  !$omp parallel workshare
  z = a * x + y
  !$omp end parallel workshare
end subroutine

! HLFIR:  func.func @_QPsb1
! HLFIR:    omp.parallel {
! HLFIR:      omp.workshare {
! HLFIR:        hlfir.elemental {{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! HLFIR:        hlfir.elemental {{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! HLFIR:        hlfir.assign
! HLFIR:        hlfir.destroy
! HLFIR:        hlfir.destroy
! HLFIR-NOT:    omp.barrier
! HLFIR:        omp.terminator
! HLFIR:      }
! HLFIR-NOT:  omp.barrier
! HLFIR:      omp.terminator
! HLFIR:    }
! HLFIR:    return
! HLFIR:  }
! HLFIR:}


! FIR:  func.func private @_workshare_copy_heap_Uxi32(%{{[a-z0-9]+}}: !fir.ref<!fir.heap<!fir.array<?xi32>>>, %{{[a-z0-9]+}}: !fir.ref<!fir.heap<!fir.array<?xi32>>>
! FIR:  func.func private @_workshare_copy_i32(%{{[a-z0-9]+}}: !fir.ref<i32>, %{{[a-z0-9]+}}: !fir.ref<i32>

! FIR:  func.func @_QPsb1
! FIR:    omp.parallel {
! FIR:      omp.single copyprivate(%{{[a-z0-9]+}} -> @_workshare_copy_i32 : !fir.ref<i32>, %{{[a-z0-9]+}} -> @_workshare_copy_heap_Uxi32 : !fir.ref<!fir.heap<!fir.array<?xi32>>>) {
! FIR:        fir.allocmem
! FIR:      omp.wsloop {
! FIR:        omp.loop_nest
! FIR:      omp.single nowait {
! FIR:        fir.call @_FortranAAssign
! FIR:        fir.freemem
! FIR:        omp.terminator
! FIR:      }
! FIR:      omp.barrier
! FIR:      omp.terminator
! FIR:    }
