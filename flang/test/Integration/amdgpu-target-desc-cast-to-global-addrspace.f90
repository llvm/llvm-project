!REQUIRES: amdgpu-registered-target

!RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -target-cpu gfx908 %s -o - | FileCheck %s

subroutine maintest
  implicit none

  type r1_t
  end type r1_t

  type(r1_t), pointer :: A
end subroutine

! CHECK: @[[TYPE_DESC:.*XdtXr1_t]] = linkonce_odr addrspace(1) constant %_QM__fortran_type_infoTderivedtype

! CHECK: define void @maintest_() {{.*}} {
! CHECK:   store { {{.*}} } { {{.*}}, ptr addrspacecast (ptr addrspace(1) @[[TYPE_DESC]] to ptr), {{.*}} }, {{.*}}
! CHECK: }
