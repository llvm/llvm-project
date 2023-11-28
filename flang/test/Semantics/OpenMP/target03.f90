! RUN: %flang_fc1 -fopenmp -fdebug-dump-parse-tree -pedantic %s 2>&1 | FileCheck %s

program main
  use omp_lib
  integer :: x, y
contains
  subroutine foo()
  !$omp target
    !CHECK: portability: The result of an OMP_GET_DEFAULT_DEVICE routine called within a TARGET region is unspecified.
    x = omp_get_default_device()
    !CHECK: portability: The result of an OMP_GET_NUM_DEVICES routine called within a TARGET region is unspecified.
    y = omp_get_num_devices()
    !CHECK: portability: The result of an OMP_SET_DEFAULT_DEVICE routine called within a TARGET region is unspecified.
    call omp_set_default_device(x)
  !$omp end target
  end subroutine
end program
