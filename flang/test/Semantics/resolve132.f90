! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror

module parent_mod_9
   interface
      module subroutine inside_one()
      end subroutine
   end interface
 end module

 submodule (parent_mod_9) sub_9
 contains
   !PORTABILITY: Subprogram 'inside_one' in this submodule is missing the MODULE prefix to implement the module procedure interface from its parent; did you mean 'MODULE SUBROUTINE'? [-Wportability]
   subroutine inside_one()
   end subroutine
 end submodule

! Same check for a function.
module parent_mod_9f
   interface
      module integer function inside_func()
      end function
   end interface
 end module

 submodule (parent_mod_9f) sub_9f
 contains
   !PORTABILITY: Subprogram 'inside_func' in this submodule is missing the MODULE prefix to implement the module procedure interface from its parent; did you mean 'MODULE FUNCTION'? [-Wportability]
   integer function inside_func()
     inside_func = 0
   end function
 end submodule

module m2
  interface
    module subroutine sub()
    end subroutine
  end interface
end module

submodule (m2) s2
contains
   !PORTABILITY: Subprogram 'sub' in this submodule is missing the MODULE prefix to implement the module procedure interface from its parent; did you mean 'MODULE SUBROUTINE'? [-Wportability]
  integer function sub()
    sub = 2
  end function
end submodule
Program abc
end
