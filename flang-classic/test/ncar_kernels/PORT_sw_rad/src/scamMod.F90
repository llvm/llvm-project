
! KGEN-generated Fortran source file
!
! Filename    : scamMod.F90
! Generated at: 2015-07-07 00:48:24
! KGEN version: 0.4.13



    MODULE scammod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !-----------------------------------------------------------------------
        !BOP
        !
        ! !MODULE: scamMod
        !
        ! !DESCRIPTION:
        ! scam specific routines and data
        !
        ! !USES:
        !
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !
        IMPLICIT NONE
        PRIVATE ! By default all data is public to this module
        !
        ! !PUBLIC INTERFACES:
        !
        ! SCAM default run-time options for CLM
        ! SCAM default run-time options
        ! SCAM run-time options
        !
        ! !PUBLIC MODULE DATA:
        !
        ! input namelist latitude for scam
        ! input namelist longitude for scam
        LOGICAL, public :: single_column ! Using IOP file or not
        ! Using IOP file or not
        ! perturb initial values
        ! perturb forcing
        ! If using diurnal averaging or not
        LOGICAL, public :: scm_crm_mode ! column radiation mode
        ! If this is a restart step or not
        ! Logical flag settings from GUI
        ! If true, update u/v after TPHYS
        ! If true, T, U & V will be passed to SLT
        ! use flux divergence terms for T and q?
        ! use flux divergence terms for constituents?
        ! do we want available diagnostics?
        ! Error code from netCDF reads
        ! 3D q advection
        ! 3D T advection
        ! vertical q advection
        ! vertical T advection
        ! surface pressure tendency
        ! model minus observed humidity
        ! actual W.V. Mixing ratio
        ! actual W.V. Mixing ratio
        ! actual W.V. Mixing ratio
        ! actual
        ! actual
        ! observed precipitation
        ! observed surface latent heat flux
        ! observed surface sensible heat flux
        ! observed apparent heat source
        ! observed apparent heat sink
        ! model minus observed temp
        ! ground temperature
        ! actual temperature
        ! air temperature at the surface
        ! model minus observed uwind
        ! actual u wind
        ! model minus observed vwind
        ! actual v wind
        ! observed cld
        ! observed clwp
        REAL(KIND=r8), public :: aldirobs(1) ! observed aldir
        REAL(KIND=r8), public :: aldifobs(1) ! observed aldif
        REAL(KIND=r8), public :: asdirobs(1) ! observed asdir
        REAL(KIND=r8), public :: asdifobs(1) ! observed asdif
        ! Vertical motion (slt)
        ! Vertical motion (slt)
        ! Divergence of moisture
        ! Divergence of temperature
        ! Horiz Divergence of E/W
        ! Horiz Divergence of N/S
        ! mo_drydep algorithm
        !
        ! index into iop dataset
        ! Length of time-step
        ! Date in (yyyymmdd) of start time
        ! Time of day of start time (sec)
        ! do we need to read next iop timepoint
        ! dataset contains divq
        ! dataset contains divt
        ! dataset contains divq3d
        ! dataset contains vertdivt
        ! dataset contains vertdivq
        ! dataset contains divt3d
        ! dataset contains divu
        ! dataset contains divv
        ! dataset contains omega
        ! dataset contains phis
        ! dataset contains ptend
        ! dataset contains ps
        ! dataset contains q
        ! dataset contains Q1
        ! dataset contains Q2
        ! dataset contains prec
        ! dataset contains lhflx
        ! dataset contains shflx
        ! dataset contains t
        ! dataset contains tg
        ! dataset contains tsair
        ! dataset contains u
        ! dataset contains v
        ! dataset contains cld
        ! dataset contains cldliq
        ! dataset contains cldice
        ! dataset contains numliq
        ! dataset contains numice
        ! dataset contains clwp
        LOGICAL*4, public :: have_aldir ! dataset contains aldir
        LOGICAL*4, public :: have_aldif ! dataset contains aldif
        LOGICAL*4, public :: have_asdir ! dataset contains asdir
        LOGICAL*4, public :: have_asdif ! dataset contains asdif
        ! use the specified surface properties
        ! use relaxation
        ! use cam generated forcing
        ! use 3d forcing
        ! IOP name for CLUBB
        !=======================================================================
            PUBLIC kgen_read_externs_scammod
        CONTAINS

        ! write subroutines

        ! module extern variables

        SUBROUTINE kgen_read_externs_scammod(kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            READ(UNIT=kgen_unit) single_column
            READ(UNIT=kgen_unit) scm_crm_mode
            READ(UNIT=kgen_unit) aldirobs
            READ(UNIT=kgen_unit) aldifobs
            READ(UNIT=kgen_unit) asdirobs
            READ(UNIT=kgen_unit) asdifobs
            READ(UNIT=kgen_unit) have_aldir
            READ(UNIT=kgen_unit) have_aldif
            READ(UNIT=kgen_unit) have_asdir
            READ(UNIT=kgen_unit) have_asdif
        END SUBROUTINE kgen_read_externs_scammod

        !=======================================================================
        !
        !-----------------------------------------------------------------------
        !


        !
        !-----------------------------------------------------------------------
        !

        !
        !-----------------------------------------------------------------------
        !
        !
        !-----------------------------------------------------------------------
        !
    END MODULE scammod
