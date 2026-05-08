! Lowering of !$acc declare rejects a component reference in a module
! declaration section

! RUN: split-file %s %t
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_module_create_component_only.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-MOD
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_module_copyin_component_only.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-MOD
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_module_device_resident_component_only.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-MOD
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_module_link_component_only.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-MOD

! ERR-MOD: not yet implemented: OpenACC declare does not support a component reference in a module; `acc declare` the whole variable instead

//--- declare_module_create_component_only.f90

module test_decl_mod_create
  type :: record_with_array
    real :: vals(8)
  end type
  type(record_with_array) :: obj
  !$acc declare create(obj%vals)
end module

//--- declare_module_copyin_component_only.f90

module test_decl_mod_copyin
  type :: record_with_array
    real :: vals(8)
  end type
  type(record_with_array) :: obj
  !$acc declare copyin(obj%vals)
end module

//--- declare_module_device_resident_component_only.f90

module test_decl_mod_device_resident
  type :: record_with_array
    real :: vals(8)
  end type
  type(record_with_array) :: obj
  !$acc declare device_resident(obj%vals)
end module

//--- declare_module_link_component_only.f90

module test_decl_mod_link
  type :: record_with_array
    real :: vals(8)
  end type
  type(record_with_array) :: obj
  !$acc declare link(obj%vals)
end module
