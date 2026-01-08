! RUN: %python %S/test_errors.py %s %flang_fc1
program test
  real, allocatable :: array(:,:)
  real, pointer :: array_ptr(:)
  interface
     subroutine sub1(array, array_ptr)
       real array(:,:)
       real, pointer :: array_ptr(:)
       namelist /my_namelist/ array, array_ptr
     end subroutine sub1
  end interface
end program test
