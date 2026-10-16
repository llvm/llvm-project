!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-ALL,LLVM-HOST %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | tco -test-gen | FileCheck --check-prefixes=MLIR-ALL,MLIR-HOST %s
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-ALL,LLVM-DEVICE %s %}
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | tco -test-gen | FileCheck --check-prefixes=MLIR-ALL,MLIR-DEVICE %s %}
! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | tco -test-gen | FileCheck --check-prefixes=MLIR-ALL,MLIR-HOST %s
! RUN: %if amdgpu-registered-target %{ bbc -target amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | tco -test-gen | FileCheck --check-prefixes=MLIR-ALL,MLIR-DEVICE %s %}

! Check that the correct LLVM IR operations are kept for the host and device
! after running the whole set of translation and transformation passes from
! Fortran for a type-bound procedure.

module mymodule
  implicit none

  type :: myclass
  contains
    procedure :: myfunc => myfunc
  end type myclass

contains
  ! MLIR-HOST: llvm.func @_QMmymodulePmyfunc
  ! MLIR-DEVICE-NOT: llvm.func @_QMmymodulePmyfunc

  ! LLVM-HOST: define void @_QMmymodulePmyfunc
  ! LLVM-DEVICE-NOT: define void @_QMmymodulePmyfunc
  subroutine myfunc(self)
    class(myclass) :: self
    call foo()
  end subroutine
end module

! MLIR-ALL: llvm.func @_QPmain

! LLVM-HOST: define void @main_
! LLVM-DEVICE-NOT: define void @main_
subroutine main(x)
  use mymodule
  implicit none

  integer, intent(inout) :: x
  class(myclass), allocatable :: myobj
  allocate(myobj)

  ! Indirect function call only present on the host.
  ! MLIR-HOST: llvm.call %{{.*}}(%{{.*}})
  ! MLIR-DEVICE-NOT: llvm.call %{{.*}}(%{{.*}})

  ! LLVM-HOST: %[[MYFUNC_PTR:.*]] = inttoptr i64 %{{.*}} to ptr
  ! LLVM-HOST: call void %[[MYFUNC_PTR]](ptr %{{.*}})
  ! LLVM-DEVICE-NOT: call void %{{.*}}(ptr %{{.*}})
  call myobj%myfunc()

  !$omp target map(tofrom: x)
  x = x + 1
  !$omp end target

  deallocate(myobj)

  ! LLVM-HOST: ret void
end subroutine main

! LLVM-ALL: define {{.*}}void @__omp_offloading{{.*}}main_{{.*}}
