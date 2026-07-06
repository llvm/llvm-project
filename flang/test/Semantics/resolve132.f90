! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror

module submodules_03_one
   integer :: one_i
   interface
      subroutine inside_one()
      end subroutine
   end interface
 end module

 submodule (submodules_03_one) submodules_03_sub_one
 contains
   !PORTABILITY: Subprogram 'inside_one' in this submodule hides an external interface from its parent module; did you mean 'MODULE SUBROUTINE'? [-Wportability]
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
   !PORTABILITY: Subprogram 'inside_one' in this submodule hides an external interface from its parent module; did you mean 'MODULE SUBROUTINE'? [-Wportability]
   subroutine inside_one()
   two_i = 6
   end subroutine
 end submodule

 program p
 use submodules_03_one
 use submodules_03_two
 call inside_one()
 end program
 
