! Module PARAMETERs used from another TU must be initialized linkonce_odr.
! PARAMETERs with !$acc declare or a CUDA data attribute keep strong linkage:
! the defining TU has the initializer; the using TU emits a declaration.
!
! RUN: split-file %s %t
! RUN: bbc -emit-hlfir %t/mod_params.f90 -o %t/mod_params.mlir --module=%t
! RUN: bbc -emit-hlfir %t/kernel.f90 -o - -I %t | FileCheck %s --check-prefix=LINKONCE
! RUN: bbc -fopenacc -emit-hlfir %t/mod_decl.f90 -o - --module=%t | FileCheck %s --check-prefix=DECL-DEF
! RUN: bbc -fopenacc -emit-hlfir %t/use_decl.f90 -o - -I %t | FileCheck %s --check-prefix=DECL-USE
! RUN: bbc -emit-hlfir %t/use_decl.f90 -o - -I %t | FileCheck %s --check-prefix=DECL-USE-NOACC
! RUN: bbc -fcuda -emit-hlfir %t/mod_cuda.f90 -o - --module=%t | FileCheck %s --check-prefix=CUDA-CONST
! RUN: bbc -fcuda -emit-hlfir %t/use_cuda.f90 -o - -I %t | FileCheck %s --check-prefix=CUDA-USE
! RUN: bbc -emit-hlfir %t/use_cuda.f90 -o - -I %t | FileCheck %s --check-prefix=CUDA-USE-NOFCUDA

//--- mod_params.f90
module mod_params
  implicit none
  real(8), parameter :: arr_val(4) = [1.0d0, 2.0d0, 3.0d0, 4.0d0]
end module

//--- kernel.f90
module kernel_mod
  use mod_params, only: arr_val
  implicit none
contains
  subroutine do_kernel(x)
    real(8), intent(inout) :: x
    x = x * arr_val(1)
  end subroutine
end module

! LINKONCE: fir.global linkonce_odr @_QMmod_paramsECarr_val({{.*}}) {{.*}}constant : !fir.array<4xf64>
! LINKONCE-NOT: fir.global @_QMmod_paramsECarr_val {{.*}}constant

//--- mod_decl.f90
module mod_decl
  implicit none
  real, parameter :: p = 1.5
  !$acc declare create(p)
end module

//--- use_decl.f90
subroutine use_decl()
  use mod_decl
  implicit none
  real :: x
  x = p
end subroutine

! Defining TU has an initialized definition; using TU is a declaration.
! DECL-DEF: fir.global @_QMmod_declECp {acc.declare = #acc.declare<dataClause = acc_create>} constant : f32 {
! DECL-DEF: fir.has_value
! DECL-DEF-NOT: fir.global linkonce_odr @_QMmod_declECp

! DECL-USE: fir.global @_QMmod_declECp {acc.declare = #acc.declare<dataClause = acc_create>
! DECL-USE-NOT: fir.global @_QMmod_declECp(
! DECL-USE-NOT: fir.has_value
! DECL-USE-NOT: fir.global linkonce_odr @_QMmod_declECp

! Without -fopenacc the .mod does not restore AccDeclare, so this is a plain
! PARAMETER and gets linkonce_odr.
! DECL-USE-NOACC: fir.global linkonce_odr @_QMmod_declECp

//--- mod_cuda.f90
module mod_cuda
  integer, parameter :: host_vals(2) = [11, 12]
  integer, constant, parameter :: const_vals(2) = [-4, -8]
end module

//--- use_cuda.f90
subroutine use_cuda()
  use mod_cuda
  implicit none
  integer :: x
  x = host_vals(1) + const_vals(1)
end subroutine

! Plain PARAMETER still gets linkonce_odr; CUDA constant PARAMETER stays strong
! with an initializer in the defining TU and a declaration in the consumer.
! CUDA-CONST-DAG: fir.global linkonce_odr @_QMmod_cudaEChost_vals
! CUDA-CONST-DAG: fir.global @_QMmod_cudaECconst_vals({{.*}}) {{.*}}data_attr = #cuf.cuda<constant>

! CUDA data attributes are stored in the .mod, so the consumer keeps external
! linkage for const_vals with or without -fcuda.  '{' after the name is a
! declaration (no dense initializer); DAG because emission order is not a contract.
! CUDA-USE-DAG: fir.global linkonce_odr @_QMmod_cudaEChost_vals({{.*}})
! CUDA-USE-DAG: fir.global @_QMmod_cudaECconst_vals {{{.*}}data_attr = #cuf.cuda<constant>
! CUDA-USE-NOFCUDA-DAG: fir.global linkonce_odr @_QMmod_cudaEChost_vals({{.*}})
! CUDA-USE-NOFCUDA-DAG: fir.global @_QMmod_cudaECconst_vals {{{.*}}data_attr = #cuf.cuda<constant>
! CUDA-USE-NOFCUDA-NOT: fir.global linkonce_odr @_QMmod_cudaECconst_vals
