! RUN: %flang_fc1 -Wunused-variable %s 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -Wunused-variable -Werror %s
! RUN: %flang_fc1 -Wno-unused-variable %s 2>&1 | FileCheck %s --check-prefix=NOWARN

! CHECK: warning: unused variable unused_var_in_submod_subroutine [-Wunused-variable]
! CHECK: warning: unused variable my_type_var1 [-Wunused-variable]
! CHECK: warning: unused variable not_dummy_arg [-Wunused-variable]
! CHECK: warning: unused variable in_subroutine [-Wunused-variable]
! CHECK: warning: unused variable c1 [-Wunused-variable]

! NOWARN-NOT: warning: unused variable unused_var_in_submod_subroutine [-Wunused-variable]
! NOWARN-NOT: warning: unused variable my_type_var1 [-Wunused-variable]
! NOWARN-NOT: warning: unused variable not_dummy_arg [-Wunused-variable]
! NOWARN-NOT: warning: unused variable in_subroutine [-Wunused-variable]
! NOWARN-NOT: warning: unused variable c1 [-Wunused-variable]
module test
        integer :: var_in_module
        contains
        subroutine module_subroutine(a)
                integer :: unused_var_in_submod_subroutine
                integer :: a
        end subroutine
end module test

program main
        type :: my_type
                integer :: val
                integer :: unused_val
        end type
        interface
                subroutine subroutine_in_interface()
                        integer :: w
                end subroutine
                function function_in_interface() result(j)
                        integer :: x
                        integer :: j
                end function
        end interface
        type(my_type) :: my_type_var1
        type(my_type) :: my_type_var2
        integer :: not_dummy_arg

        integer :: variable_common
        common variable_common

        my_type_var2%val = 12



        print *, function_used_all()

        contains
        subroutine subroutine_all_used(a3)
                integer, intent(in) :: a3
                integer :: in_subroutine_used
                in_subroutine_used = a3
        end subroutine

        subroutine subroutine_unused(a4)
                integer, intent(in) :: a4
                integer :: in_subroutine
        end subroutine

        function function_used_all() result(c1)
                integer :: in_function
                integer :: c1
                c1 = in_function
        end function 

        function function_unused_all() result(c1)
        end function 
end program
