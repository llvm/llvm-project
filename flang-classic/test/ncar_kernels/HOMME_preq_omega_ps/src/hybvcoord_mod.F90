
! KGEN-generated Fortran source file
!
! Filename    : hybvcoord_mod.F90
! Generated at: 2015-03-16 09:25:31
! KGEN version: 0.4.5



    MODULE hybvcoord_mod
        USE kinds, ONLY: r8 => real_kind
        USE dimensions_mod, ONLY: plevp => nlevp
        USE dimensions_mod, ONLY: plev => nlev
        IMPLICIT NONE
        PRIVATE
        !-----------------------------------------------------------------------
        ! hvcoord_t: Hybrid level definitions: p = a*p0 + b*ps
        !            interfaces   p(k) = hyai(k)*ps0 + hybi(k)*ps
        !            midpoints    p(k) = hyam(k)*ps0 + hybm(k)*ps
        !-----------------------------------------------------------------------
        TYPE, public :: hvcoord_t
            REAL(KIND=r8) :: ps0 ! base state surface-pressure for level definitions
            REAL(KIND=r8) :: hyai(plevp) ! ps0 component of hybrid coordinate - interfaces
            REAL(KIND=r8) :: hyam(plev) ! ps0 component of hybrid coordinate - midpoints
            REAL(KIND=r8) :: hybi(plevp) ! ps  component of hybrid coordinate - interfaces
            REAL(KIND=r8) :: hybm(plev) ! ps  component of hybrid coordinate - midpoints
            REAL(KIND=r8) :: hybd(plev) ! difference in b (hybi) across layers
            REAL(KIND=r8) :: prsfac ! log pressure extrapolation factor (time, space independent)
            REAL(KIND=r8) :: etam(plev) ! eta-levels at midpoints
            REAL(KIND=r8) :: etai(plevp) ! eta-levels at interfaces
            INTEGER :: nprlev ! number of pure pressure levels at top
            INTEGER :: pad
        END TYPE hvcoord_t

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_hvcoord_t
        END INTERFACE kgen_read

        CONTAINS

        ! write subroutines
        ! No module extern variables
        SUBROUTINE kgen_read_hvcoord_t(var, kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            TYPE(hvcoord_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%ps0
            READ(UNIT=kgen_unit) var%hyai
            READ(UNIT=kgen_unit) var%hyam
            READ(UNIT=kgen_unit) var%hybi
            READ(UNIT=kgen_unit) var%hybm
            READ(UNIT=kgen_unit) var%hybd
            READ(UNIT=kgen_unit) var%prsfac
            READ(UNIT=kgen_unit) var%etam
            READ(UNIT=kgen_unit) var%etai
            READ(UNIT=kgen_unit) var%nprlev
            READ(UNIT=kgen_unit) var%pad
        END SUBROUTINE
        !_____________________________________________________________________

        !_______________________________________________________________________

    END MODULE hybvcoord_mod
