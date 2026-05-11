! RUN: %flang -c %s
! RUN: %flang_skip_driver -c %s

module example_module
    implicit none

    abstract interface

        subroutine sub_i
          implicit none
        end subroutine

    end interface

contains

    subroutine call_internal(string)
        implicit none
        character(len=*), intent(in) :: string

        call call_it(print_it)

    contains

        subroutine print_it
            implicit none

            print *, string
        end subroutine

    end subroutine

    subroutine call_it(sub)
        implicit none
        procedure(sub_i) :: sub

        call sub
    end subroutine

end module

program module_torture
    use example_module
    implicit none

    call call_internal("Hello, World!")
end program
