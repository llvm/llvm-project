! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-HOST,LLVM-ALL %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-HOST,MLIR-ALL %s
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM-DEVICE,LLVM-ALL %s %}
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s %}
! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-HOST,MLIR-ALL %s
! RUN: %if amdgpu-registered-target %{ bbc -target amdgcn-amd-amdhsa -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -emit-hlfir %s -o - | FileCheck --check-prefixes=MLIR-DEVICE,MLIR-ALL %s %}

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
  ! MLIR-HOST: func.func @_QMmymodulePmyfunc
  ! MLIR-DEVICE-NOT: func.func @_QMmymodulePmyfunc

  ! LLVM-HOST: define void @_QMmymodulePmyfunc
  ! LLVM-DEVICE-NOT: define void @_QMmymodulePmyfunc
  subroutine myfunc(self)
    class(myclass) :: self
    call foo()
  end subroutine
end module

! MLIR-ALL: func.func @_QPmain

! LLVM-HOST: define void @main_
! LLVM-DEVICE-NOT: define void @main_
subroutine main(x)
  use mymodule
  implicit none

  integer, intent(inout) :: x
  class(myclass), allocatable :: myobj
  allocate(myobj)

  ! MLIR-ALL: fir.dispatch "myfunc"

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

! MLIR-ALL: fir.type_info @_QMmymoduleTmyclass {{.*}} dispatch_table
! MLIR-ALL-NEXT: fir.dt_entry "myfunc", @_QMmymodulePmyfunc
