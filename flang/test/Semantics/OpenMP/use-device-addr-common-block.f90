! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50 -Wno-openmp-usage

! OpenMP 5.0 allows map list items to share original storage when they are the
! same variable or array section. A named COMMON block has the same meaning as
! its explicit members, so repeating its map clauses is valid.

subroutine valid_map_use_device_addr
  integer :: x, y
  common /valid/ x, y

  !$omp target data map(tofrom: /valid/) use_device_addr(/valid/)
  !$omp end target data

  !$omp target data use_device_addr(/valid/) map(from: /valid/)
  !$omp end target data

  !$omp target data map(alloc: /valid/) use_device_addr(/valid/)
  !$omp end target data
end subroutine

subroutine repeated_maps
  integer :: x, y
  common /repeated/ x, y

  !$omp target data map(from: /repeated/) map(to: /repeated/)
  !$omp end target data

  !$omp target data map(to: /repeated/) map(to: /repeated/) map(to: /repeated/)
  !$omp end target data
end subroutine

subroutine repeated_map_controls
  integer :: x, y, scalar
  common /controls/ x, y

  !$omp target data map(from: x, y) map(to: x, y) map(to: x, y)
  !$omp end target data

  !$omp target data map(from: scalar) map(to: scalar) map(to: scalar)
  !$omp end target data
end subroutine

subroutine duplicate_and_exclusive_clauses
  integer :: x, y
  common /duplicates/ x, y

  !ERROR: List item 'duplicates' present at multiple USE_DEVICE_ADDR clauses
  !ERROR: 'duplicates' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data use_device_addr(/duplicates/) use_device_addr(/duplicates/)
  !$omp end target data

  !ERROR: 'duplicates' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data use_device_ptr(/duplicates/) use_device_ptr(/duplicates/)
  !$omp end target data

  !ERROR: 'duplicates' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data use_device_ptr(/duplicates/) use_device_addr(/duplicates/)
  !$omp end target data

  !ERROR: 'duplicates' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data use_device_addr(/duplicates/) use_device_ptr(/duplicates/)
  !$omp end target data
end subroutine

subroutine repeated_maps_with_use_device_addr
  integer :: x, y
  common /third/ x, y

  !$omp target data map(from: /third/) map(to: /third/) use_device_addr(/third/)
  !$omp end target data

  !$omp target data map(from: /third/) use_device_addr(/third/) map(to: /third/)
  !$omp end target data

  !$omp target data use_device_addr(/third/) map(from: /third/) map(to: /third/)
  !$omp end target data

  !ERROR: List item 'third' present at multiple USE_DEVICE_ADDR clauses
  !ERROR: 'third' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data map(from: /third/) map(to: /third/) use_device_addr(/third/) use_device_addr(/third/)
  !$omp end target data

  !ERROR: List item 'third' present at multiple USE_DEVICE_ADDR clauses
  !ERROR: 'third' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data use_device_addr(/third/) map(from: /third/) map(to: /third/) use_device_addr(/third/)
  !$omp end target data
end subroutine

subroutine nested_repeated_maps
  integer :: x, y
  common /nested_repeated/ x, y

  !$omp target data map(from: /nested_repeated/) use_device_addr(/nested_repeated/) map(to: /nested_repeated/)
    !$omp target data use_device_addr(/nested_repeated/) map(from: /nested_repeated/) map(to: /nested_repeated/)
    !$omp end target data
  !$omp end target data
end subroutine

subroutine unrelated_data_sharing_conflict
  integer :: x, y
  common /conflict/ x, y

  !ERROR: 'conflict' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target map(to: /conflict/) map(from: /conflict/) private(/conflict/)
  !$omp end target
end subroutine

subroutine map_use_device_ptr_non_cptr
  integer :: x, y
  common /non_cptr/ x, y

  !ERROR: 'non_cptr' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data map(tofrom: /non_cptr/) use_device_ptr(/non_cptr/)
  !$omp end target data

  !ERROR: 'non_cptr' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data use_device_ptr(/non_cptr/) map(tofrom: /non_cptr/)
  !$omp end target data
end subroutine

subroutine map_use_device_ptr_cptr
  use iso_c_binding, only : c_ptr
  type(c_ptr) :: p, q
  common /all_cptr/ p, q

  !ERROR: 'all_cptr' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data map(tofrom: /all_cptr/) use_device_ptr(/all_cptr/)
  !$omp end target data

  !ERROR: 'all_cptr' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data use_device_ptr(/all_cptr/) map(tofrom: /all_cptr/)
  !$omp end target data
end subroutine

subroutine map_use_device_ptr_mixed
  use iso_c_binding, only : c_ptr
  type(c_ptr) :: p
  integer :: x
  common /mixed/ p, x

  !ERROR: 'mixed' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data map(tofrom: /mixed/) use_device_ptr(/mixed/)
  !$omp end target data

  !ERROR: 'mixed' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp target data use_device_ptr(/mixed/) map(tofrom: /mixed/)
  !$omp end target data
end subroutine
