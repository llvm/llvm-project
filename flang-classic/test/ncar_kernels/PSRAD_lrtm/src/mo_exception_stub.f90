
! KGEN-generated Fortran source file
!
! Filename    : mo_exception_stub.f90
! Generated at: 2015-02-19 15:30:31
! KGEN version: 0.4.4



    MODULE mo_exception
        IMPLICIT NONE
        PRIVATE
        PUBLIC finish
        ! normal message
        ! informational message
        ! warning message: number of warnings counted
        ! error message: number of errors counted
        ! report parameter value
        ! debugging message
        !++mgs
        CONTAINS

        ! read subroutines

        SUBROUTINE finish(name, text, exit_no)
            CHARACTER(LEN=*), intent(in) :: name
            CHARACTER(LEN=*), intent(in), optional :: text
            INTEGER, intent(in), optional :: exit_no
            INTEGER :: ifile
            IF (present(exit_no)) THEN
                ifile = exit_no
                ELSE
                ifile = 6
            END IF 
            WRITE (ifile, '(/,80("*"),/)')
            IF (present(text)) THEN
                WRITE (ifile, '(1x,a,a,a)') trim(name), ': ', trim(text)
                ELSE
                WRITE (ifile, '(1x,a,a)') trim(name), ': '
            END IF 
            WRITE (ifile, '(/,80("-"),/,/)')
            STOP
        END SUBROUTINE finish

    END MODULE mo_exception
