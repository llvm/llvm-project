! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! OpenMP Version 5.2
! Inherited from 2.11.3 allocate Directive
! If list items within the ALLOCATE directive have the SAVE attribute, are a common block name, or are declared in the scope of a
! module, then only predefined memory allocator parameters can be used in the allocator clause
! SAVE and common block names can't be declared as allocatable, only module scope variables are tested

module AllocateModule
    integer, allocatable :: a, b
end module

subroutine allocate()
    use omp_lib
    use AllocateModule

    integer(kind=omp_allocator_handle_kind) :: custom_allocator
    type(omp_alloctrait) :: trait(1)

    trait(1)%key = fallback
    trait(1)%value = default_mem_fb
    custom_allocator = omp_init_allocator(omp_default_mem_space, 1, trait)

    !$omp allocators allocate(omp_default_mem_alloc: a)
        allocate(a)

    !ERROR: If list items within the ALLOCATORS directive have the SAVE attribute, are a common block name, or are declared in the scope of a module, then only predefined memory allocator parameters can be used in the allocator clause
    !$omp allocators allocate(custom_allocator: b)
        allocate(b)
end subroutine
