! RUN: %python %S/test_errors.py %s %flang_fc1

module submodules_03_one
   integer :: one_i
   interface
      subroutine inside_one()
      end subroutine
   end interface
 end module

 submodule (submodules_03_one) submodules_03_sub_one
 contains
   subroutine inside_one()
   one_i = 6
   end subroutine
 end submodule

 module submodules_03_two
   integer :: two_i
   interface
      subroutine inside_one()
      end subroutine
   end interface
 end module

 submodule (submodules_03_two) sub_one
   contains
   subroutine inside_one()
   two_i = 6
   end subroutine
 end submodule

 program p
 use submodules_03_one
 use submodules_03_two
 !ERROR: Reference to 'inside_one' is ambiguous
 call inside_one()
 end program
 
