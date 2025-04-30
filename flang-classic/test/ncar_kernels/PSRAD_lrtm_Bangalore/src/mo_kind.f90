
! KGEN-generated Fortran source file
!
! Filename    : mo_kind.f90
! Generated at: 2015-02-19 15:30:37
! KGEN version: 0.4.4



    MODULE mo_kind
        ! L. Kornblueh, MPI, August 2001, added working precision and comments
        IMPLICIT NONE
        ! Number model from which the SELECTED_*_KIND are requested:
        !
        !                   4 byte REAL      8 byte REAL
        !          CRAY:        -            precision =   13
        !                                    exponent  = 2465
        !          IEEE:    precision =  6   precision =   15
        !                   exponent  = 37   exponent  =  307
        !
        ! Most likely this are the only possible models.
        ! Floating point section:
        INTEGER, parameter :: pd = 12
        INTEGER, parameter :: rd = 307
        INTEGER, parameter :: pi8 = 14
        INTEGER, parameter :: dp = selected_real_kind(pd,rd)
        ! Floating point working precision
        INTEGER, parameter :: wp = dp
        ! Integer section
        INTEGER, parameter :: i8 = selected_int_kind(pi8)
        ! Working precision for index variables
        !
        ! predefined preprocessor macros:
        !
        ! xlf         __64BIT__   checked with P6 and AIX
        ! gfortran    __LP64__    checked with Darwin and Linux
        ! Intel, PGI  __x86_64__  checked with Linux
        ! Sun         __x86_64    checked with Linux
        CONTAINS

        ! read subroutines

    END MODULE mo_kind
