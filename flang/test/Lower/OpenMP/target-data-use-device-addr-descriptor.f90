! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-fir %s -o - | fir-opt | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | fir-opt -o /dev/null

! Descriptor map results and target data block arguments must have matching
! types, including when the descriptor is passed by value to the subroutine.
! Reparse textual FIR and HLFIR to catch mismatches between the types printed
! in the use_device_addr clause and those used inside the region.

subroutine device_addr_default(x)
  integer, target, intent(in) :: x(:)
  !$omp target data use_device_addr(x)
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPdevice_addr_default(
! CHECK: omp.target_data {{.*}}use_device_addr(%{{.*}} -> %[[ARG:.*]], %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
! CHECK-NEXT: %[[BOX:.*]] = fir.load %[[ARG]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK-NEXT: %{{.*}} = fir.declare %[[BOX]] {{.*}} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>

subroutine device_addr_rank_two(x)
  integer, target, intent(in) :: x(:, :)
  interface
    subroutine consume_rank_two(x)
      integer, intent(in) :: x(:, :)
    end subroutine
  end interface
  !$omp target data use_device_addr(x)
    call consume_rank_two(x)
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPdevice_addr_rank_two(
! CHECK: omp.target_data {{.*}}use_device_addr(%{{.*}} -> %[[ARG:.*]], %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.array<?x?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?x?xi32>>>) {
! CHECK-NEXT: %[[BOX:.*]] = fir.load %[[ARG]] : !fir.ref<!fir.box<!fir.array<?x?xi32>>>
! CHECK-NEXT: %{{.*}} = fir.declare %[[BOX]] {{.*}} : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
! CHECK: fir.call @_QPconsume_rank_two(%{{.*}}) {{.*}} : (!fir.box<!fir.array<?x?xi32>>) -> ()

! OpenMP 5.0 treats a non-C_PTR use_device_ptr item as use_device_addr.
subroutine device_ptr_promoted(x)
  integer, target, intent(in) :: x(:)
  !$omp target data use_device_ptr(x)
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPdevice_ptr_promoted(
! CHECK: omp.target_data {{.*}}use_device_addr(%{{.*}} -> %[[ARG:.*]], %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
! CHECK-NEXT: %[[BOX:.*]] = fir.load %[[ARG]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK-NEXT: %{{.*}} = fir.declare %[[BOX]] {{.*}} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>

subroutine device_addr_optional(x)
  integer, target, optional, intent(in) :: x(:)
  !$omp target data use_device_addr(x)
  !$omp end target data
end subroutine

! CHECK-LABEL: func.func @_QPdevice_addr_optional(
! CHECK: omp.target_data {{.*}}use_device_addr(%{{.*}} -> %[[ARG:.*]], %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
! CHECK-NEXT: %[[BOX:.*]] = fir.load %[[ARG]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK-NEXT: %{{.*}} = fir.declare %[[BOX]] {{.*}} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
