! RUN: %flang_fc1 -Wunused-dummy-argument %s 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -Wunused-dummy-argument -Werror %s
! RUN: %flang_fc1 -Wno-unused-dummy-argument %s 2>&1 | FileCheck %s --check-prefix=NOWARN

! CHECK: warning: unused dummy argument a4 [-Wunused-dummy-argument]
! CHECK: warning: unused dummy argument b4 [-Wunused-dummy-argument]
! CHECK: warning: unused dummy argument a6 [-Wunused-dummy-argument]
! CHECK: warning: unused dummy argument b6 [-Wunused-dummy-argument]

! NOWARN-NOT: warning: unused dummy argument a4 [-Wunused-dummy-argument]
! NOWARN-NOT: warning: unused dummy argument b4 [-Wunused-dummy-argument]
! NOWARN-NOT: warning: unused dummy argument a6 [-Wunused-dummy-argument]
! NOWARN-NOT: warning: unused dummy argument b6 [-Wunused-dummy-argument]

program main
        type :: my_type
                integer :: val
        end type
        integer :: not_dummy_arg
        interface
                subroutine subroutine_interface(a)
                        integer, intent(in) :: a
                end subroutine

                function function_interface(a2)
                        integer, intent(in) :: a2
                end function
        end interface
contains
        subroutine subroutine_all_used(a3, b3)
                integer, intent(inout) :: a3, b3
                a3 = a3 + b3
        end subroutine

        subroutine subroutine_unused_both(a4, b4)
                integer, intent(inout) :: a4(10)
                type(my_type) :: b4
        end subroutine


        function function_used_all(a5, b5) result(c1)
                integer, intent(inout) :: a5(10)
                type(my_type), intent(in) :: b5
                integer :: c1
                a5(1) = b5%val
                c1 = a5(2)
        end function 

        function function_unused_both(a6, b6) result(c2)
                integer, intent(inout) :: a6(10)
                type(my_type) :: b6
                integer :: c2
        end function
end program
