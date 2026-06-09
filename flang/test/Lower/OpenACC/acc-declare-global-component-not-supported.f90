! Reject unsupported !$acc declare module structure component cases:
! - `obj%vals` without the whole variable.
! - Non-simple designators (array element base, chained `%`).

! RUN: split-file %s %t
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_module_create_component_only.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-WHOLE
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_module_copyin_component_only.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-WHOLE
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_module_device_resident_component_only.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-WHOLE
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_module_link_component_only.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-WHOLE
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_array_element_component.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-TODO
! RUN: not bbc -fopenacc -emit-hlfir %t/declare_nested_structure_components.f90 -o - 2>&1 | FileCheck %s --check-prefixes=ERR-TODO

! ERR-WHOLE: not yet implemented: OpenACC declare: whole variable must appear in ACC DECLARE before a component reference; list the whole variable in this or an earlier clause
! ERR-TODO: not yet implemented: OpenACC declare does not support this component reference in a module; ACC DECLARE the whole variable instead

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

//--- declare_array_element_component.f90

module mod_arr_elem
  type :: record_with_array
    real :: vals(8)
  end type
  type(record_with_array), dimension(4) :: objs
  !$acc declare create(objs(1)%vals)
end module

//--- declare_nested_structure_components.f90

module mod_nested
  type :: inner_t
    real :: z
  end type
  type :: outer_t
    type(inner_t) :: inner
  end type
  type(outer_t) :: o
  !$acc declare create(o%inner%z)
end module
