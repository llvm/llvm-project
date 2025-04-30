
! KGEN-generated Fortran source file
!
! Filename    : camsrfexch.F90
! Generated at: 2015-07-07 00:48:25
! KGEN version: 0.4.13



    MODULE camsrfexch
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !-----------------------------------------------------------------------
        !
        ! Module to handle data that is exchanged between the CAM atmosphere
        ! model and the surface models (land, sea-ice, and ocean).
        !
        !-----------------------------------------------------------------------
        !
        ! USES:
        !
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        USE constituents, ONLY: pcnst
        USE ppgrid, ONLY: pcols
        IMPLICIT NONE
        !-----------------------------------------------------------------------
        ! PRIVATE: Make default data and interfaces private
        !-----------------------------------------------------------------------
        PRIVATE ! By default all data is private to this module
        !
        ! Public interfaces
        !
        ! Atmosphere to surface data allocation method
        ! Merged hub surface to atmosphere data allocation method
        ! Set options to allocate optional parts of data type
        !
        ! Public data types
        !
        PUBLIC cam_out_t ! Data from atmosphere
        PUBLIC cam_in_t ! Merged surface data
        !---------------------------------------------------------------------------
        ! This is the data that is sent from the atmosphere to the surface models
        !---------------------------------------------------------------------------
        TYPE cam_out_t
            INTEGER :: lchnk ! chunk index
            INTEGER :: ncol ! number of columns in chunk
            REAL(KIND=r8) :: tbot(pcols) ! bot level temperature
            REAL(KIND=r8) :: zbot(pcols) ! bot level height above surface
            REAL(KIND=r8) :: ubot(pcols) ! bot level u wind
            REAL(KIND=r8) :: vbot(pcols) ! bot level v wind
            REAL(KIND=r8) :: qbot(pcols,pcnst) ! bot level specific humidity
            REAL(KIND=r8) :: pbot(pcols) ! bot level pressure
            REAL(KIND=r8) :: rho(pcols) ! bot level density
            REAL(KIND=r8) :: netsw(pcols) !
            REAL(KIND=r8) :: flwds(pcols) !
            REAL(KIND=r8) :: precsc(pcols) !
            REAL(KIND=r8) :: precsl(pcols) !
            REAL(KIND=r8) :: precc(pcols) !
            REAL(KIND=r8) :: precl(pcols) !
            REAL(KIND=r8) :: soll(pcols) !
            REAL(KIND=r8) :: sols(pcols) !
            REAL(KIND=r8) :: solld(pcols) !
            REAL(KIND=r8) :: solsd(pcols) !
            REAL(KIND=r8) :: thbot(pcols) !
            REAL(KIND=r8) :: co2prog(pcols) ! prognostic co2
            REAL(KIND=r8) :: co2diag(pcols) ! diagnostic co2
            REAL(KIND=r8) :: psl(pcols)
            REAL(KIND=r8) :: bcphiwet(pcols) ! wet deposition of hydrophilic black carbon
            REAL(KIND=r8) :: bcphidry(pcols) ! dry deposition of hydrophilic black carbon
            REAL(KIND=r8) :: bcphodry(pcols) ! dry deposition of hydrophobic black carbon
            REAL(KIND=r8) :: ocphiwet(pcols) ! wet deposition of hydrophilic organic carbon
            REAL(KIND=r8) :: ocphidry(pcols) ! dry deposition of hydrophilic organic carbon
            REAL(KIND=r8) :: ocphodry(pcols) ! dry deposition of hydrophobic organic carbon
            REAL(KIND=r8) :: dstwet1(pcols) ! wet deposition of dust (bin1)
            REAL(KIND=r8) :: dstdry1(pcols) ! dry deposition of dust (bin1)
            REAL(KIND=r8) :: dstwet2(pcols) ! wet deposition of dust (bin2)
            REAL(KIND=r8) :: dstdry2(pcols) ! dry deposition of dust (bin2)
            REAL(KIND=r8) :: dstwet3(pcols) ! wet deposition of dust (bin3)
            REAL(KIND=r8) :: dstdry3(pcols) ! dry deposition of dust (bin3)
            REAL(KIND=r8) :: dstwet4(pcols) ! wet deposition of dust (bin4)
            REAL(KIND=r8) :: dstdry4(pcols) ! dry deposition of dust (bin4)
        END TYPE cam_out_t
        !---------------------------------------------------------------------------
        ! This is the merged state of sea-ice, land and ocean surface parameterizations
        !---------------------------------------------------------------------------
        TYPE cam_in_t
            INTEGER :: lchnk ! chunk index
            INTEGER :: ncol ! number of active columns
            REAL(KIND=r8) :: asdir(pcols) ! albedo: shortwave, direct
            REAL(KIND=r8) :: asdif(pcols) ! albedo: shortwave, diffuse
            REAL(KIND=r8) :: aldir(pcols) ! albedo: longwave, direct
            REAL(KIND=r8) :: aldif(pcols) ! albedo: longwave, diffuse
            REAL(KIND=r8) :: lwup(pcols) ! longwave up radiative flux
            REAL(KIND=r8) :: lhf(pcols) ! latent heat flux
            REAL(KIND=r8) :: shf(pcols) ! sensible heat flux
            REAL(KIND=r8) :: wsx(pcols) ! surface u-stress (N)
            REAL(KIND=r8) :: wsy(pcols) ! surface v-stress (N)
            REAL(KIND=r8) :: tref(pcols) ! ref height surface air temp
            REAL(KIND=r8) :: qref(pcols) ! ref height specific humidity
            REAL(KIND=r8) :: u10(pcols) ! 10m wind speed
            REAL(KIND=r8) :: ts(pcols) ! merged surface temp
            REAL(KIND=r8) :: sst(pcols) ! sea surface temp
            REAL(KIND=r8) :: snowhland(pcols) ! snow depth (liquid water equivalent) over land
            REAL(KIND=r8) :: snowhice(pcols) ! snow depth over ice
            REAL(KIND=r8) :: fco2_lnd(pcols) ! co2 flux from lnd
            REAL(KIND=r8) :: fco2_ocn(pcols) ! co2 flux from ocn
            REAL(KIND=r8) :: fdms(pcols) ! dms flux
            REAL(KIND=r8) :: landfrac(pcols) ! land area fraction
            REAL(KIND=r8) :: icefrac(pcols) ! sea-ice areal fraction
            REAL(KIND=r8) :: ocnfrac(pcols) ! ocean areal fraction
            REAL(KIND=r8), pointer, dimension(:) :: ram1 !aerodynamical resistance (s/m) (pcols)
            REAL(KIND=r8), pointer, dimension(:) :: fv !friction velocity (m/s) (pcols)
            REAL(KIND=r8), pointer, dimension(:) :: soilw !volumetric soil water (m3/m3)
            REAL(KIND=r8) :: cflx(pcols,pcnst) ! constituent flux (evap)
            REAL(KIND=r8) :: ustar(pcols) ! atm/ocn saved version of ustar
            REAL(KIND=r8) :: re(pcols) ! atm/ocn saved version of re
            REAL(KIND=r8) :: ssq(pcols) ! atm/ocn saved version of ssq
            REAL(KIND=r8), pointer, dimension(:,:) :: depvel ! deposition velocities
        END TYPE cam_in_t
        ! .true. => aerosol dust package is being used
        !===============================================================================

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_cam_out_t
            MODULE PROCEDURE kgen_read_cam_in_t
        END INTERFACE kgen_read

        PUBLIC kgen_verify
        INTERFACE kgen_verify
            MODULE PROCEDURE kgen_verify_cam_out_t
            MODULE PROCEDURE kgen_verify_cam_in_t
        END INTERFACE kgen_verify

        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim1_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), POINTER, DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim1_ptr

            SUBROUTINE kgen_read_real_r8_dim2_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), POINTER, DIMENSION(:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2
                INTEGER, DIMENSION(2,2) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim2_ptr

        ! No module extern variables
        SUBROUTINE kgen_read_cam_out_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(cam_out_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%lchnk
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%lchnk **", var%lchnk
            END IF
            READ(UNIT=kgen_unit) var%ncol
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ncol **", var%ncol
            END IF
            READ(UNIT=kgen_unit) var%tbot
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%tbot **", var%tbot
            END IF
            READ(UNIT=kgen_unit) var%zbot
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%zbot **", var%zbot
            END IF
            READ(UNIT=kgen_unit) var%ubot
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ubot **", var%ubot
            END IF
            READ(UNIT=kgen_unit) var%vbot
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%vbot **", var%vbot
            END IF
            READ(UNIT=kgen_unit) var%qbot
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%qbot **", var%qbot
            END IF
            READ(UNIT=kgen_unit) var%pbot
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%pbot **", var%pbot
            END IF
            READ(UNIT=kgen_unit) var%rho
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%rho **", var%rho
            END IF
            READ(UNIT=kgen_unit) var%netsw
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%netsw **", var%netsw
            END IF
            READ(UNIT=kgen_unit) var%flwds
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%flwds **", var%flwds
            END IF
            READ(UNIT=kgen_unit) var%precsc
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%precsc **", var%precsc
            END IF
            READ(UNIT=kgen_unit) var%precsl
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%precsl **", var%precsl
            END IF
            READ(UNIT=kgen_unit) var%precc
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%precc **", var%precc
            END IF
            READ(UNIT=kgen_unit) var%precl
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%precl **", var%precl
            END IF
            READ(UNIT=kgen_unit) var%soll
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%soll **", var%soll
            END IF
            READ(UNIT=kgen_unit) var%sols
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%sols **", var%sols
            END IF
            READ(UNIT=kgen_unit) var%solld
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%solld **", var%solld
            END IF
            READ(UNIT=kgen_unit) var%solsd
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%solsd **", var%solsd
            END IF
            READ(UNIT=kgen_unit) var%thbot
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%thbot **", var%thbot
            END IF
            READ(UNIT=kgen_unit) var%co2prog
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%co2prog **", var%co2prog
            END IF
            READ(UNIT=kgen_unit) var%co2diag
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%co2diag **", var%co2diag
            END IF
            READ(UNIT=kgen_unit) var%psl
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%psl **", var%psl
            END IF
            READ(UNIT=kgen_unit) var%bcphiwet
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%bcphiwet **", var%bcphiwet
            END IF
            READ(UNIT=kgen_unit) var%bcphidry
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%bcphidry **", var%bcphidry
            END IF
            READ(UNIT=kgen_unit) var%bcphodry
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%bcphodry **", var%bcphodry
            END IF
            READ(UNIT=kgen_unit) var%ocphiwet
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ocphiwet **", var%ocphiwet
            END IF
            READ(UNIT=kgen_unit) var%ocphidry
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ocphidry **", var%ocphidry
            END IF
            READ(UNIT=kgen_unit) var%ocphodry
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ocphodry **", var%ocphodry
            END IF
            READ(UNIT=kgen_unit) var%dstwet1
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dstwet1 **", var%dstwet1
            END IF
            READ(UNIT=kgen_unit) var%dstdry1
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dstdry1 **", var%dstdry1
            END IF
            READ(UNIT=kgen_unit) var%dstwet2
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dstwet2 **", var%dstwet2
            END IF
            READ(UNIT=kgen_unit) var%dstdry2
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dstdry2 **", var%dstdry2
            END IF
            READ(UNIT=kgen_unit) var%dstwet3
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dstwet3 **", var%dstwet3
            END IF
            READ(UNIT=kgen_unit) var%dstdry3
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dstdry3 **", var%dstdry3
            END IF
            READ(UNIT=kgen_unit) var%dstwet4
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dstwet4 **", var%dstwet4
            END IF
            READ(UNIT=kgen_unit) var%dstdry4
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dstdry4 **", var%dstdry4
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_read_cam_in_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(cam_in_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%lchnk
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%lchnk **", var%lchnk
            END IF
            READ(UNIT=kgen_unit) var%ncol
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ncol **", var%ncol
            END IF
            READ(UNIT=kgen_unit) var%asdir
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%asdir **", var%asdir
            END IF
            READ(UNIT=kgen_unit) var%asdif
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%asdif **", var%asdif
            END IF
            READ(UNIT=kgen_unit) var%aldir
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%aldir **", var%aldir
            END IF
            READ(UNIT=kgen_unit) var%aldif
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%aldif **", var%aldif
            END IF
            READ(UNIT=kgen_unit) var%lwup
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%lwup **", var%lwup
            END IF
            READ(UNIT=kgen_unit) var%lhf
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%lhf **", var%lhf
            END IF
            READ(UNIT=kgen_unit) var%shf
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%shf **", var%shf
            END IF
            READ(UNIT=kgen_unit) var%wsx
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%wsx **", var%wsx
            END IF
            READ(UNIT=kgen_unit) var%wsy
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%wsy **", var%wsy
            END IF
            READ(UNIT=kgen_unit) var%tref
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%tref **", var%tref
            END IF
            READ(UNIT=kgen_unit) var%qref
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%qref **", var%qref
            END IF
            READ(UNIT=kgen_unit) var%u10
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%u10 **", var%u10
            END IF
            READ(UNIT=kgen_unit) var%ts
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ts **", var%ts
            END IF
            READ(UNIT=kgen_unit) var%sst
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%sst **", var%sst
            END IF
            READ(UNIT=kgen_unit) var%snowhland
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%snowhland **", var%snowhland
            END IF
            READ(UNIT=kgen_unit) var%snowhice
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%snowhice **", var%snowhice
            END IF
            READ(UNIT=kgen_unit) var%fco2_lnd
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%fco2_lnd **", var%fco2_lnd
            END IF
            READ(UNIT=kgen_unit) var%fco2_ocn
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%fco2_ocn **", var%fco2_ocn
            END IF
            READ(UNIT=kgen_unit) var%fdms
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%fdms **", var%fdms
            END IF
            READ(UNIT=kgen_unit) var%landfrac
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%landfrac **", var%landfrac
            END IF
            READ(UNIT=kgen_unit) var%icefrac
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%icefrac **", var%icefrac
            END IF
            READ(UNIT=kgen_unit) var%ocnfrac
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ocnfrac **", var%ocnfrac
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_ptr(var%ram1, kgen_unit, printvar=printvar//"%ram1")
            ELSE
                CALL kgen_read_real_r8_dim1_ptr(var%ram1, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_ptr(var%fv, kgen_unit, printvar=printvar//"%fv")
            ELSE
                CALL kgen_read_real_r8_dim1_ptr(var%fv, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_ptr(var%soilw, kgen_unit, printvar=printvar//"%soilw")
            ELSE
                CALL kgen_read_real_r8_dim1_ptr(var%soilw, kgen_unit)
            END IF
            READ(UNIT=kgen_unit) var%cflx
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%cflx **", var%cflx
            END IF
            READ(UNIT=kgen_unit) var%ustar
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ustar **", var%ustar
            END IF
            READ(UNIT=kgen_unit) var%re
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%re **", var%re
            END IF
            READ(UNIT=kgen_unit) var%ssq
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ssq **", var%ssq
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_ptr(var%depvel, kgen_unit, printvar=printvar//"%depvel")
            ELSE
                CALL kgen_read_real_r8_dim2_ptr(var%depvel, kgen_unit)
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_cam_out_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(cam_out_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
!
! Tolerance has to be changed to 1.0e-12 if FMA instructions are generated.
! Without FMA, tolerance can be set to 1.0e-13.
! Only array solld falls outside the default tolerance of 1.0e-15.
!
            CALL kgen_init_check(dtype_check_status,tolerance=real(1.0e-12,kind=kgen_dp))
            CALL kgen_verify_integer("lchnk", dtype_check_status, var%lchnk, ref_var%lchnk)
            CALL kgen_verify_integer("ncol", dtype_check_status, var%ncol, ref_var%ncol)
            CALL kgen_verify_real_r8_dim1("tbot", dtype_check_status, var%tbot, ref_var%tbot)
            CALL kgen_verify_real_r8_dim1("zbot", dtype_check_status, var%zbot, ref_var%zbot)
            CALL kgen_verify_real_r8_dim1("ubot", dtype_check_status, var%ubot, ref_var%ubot)
            CALL kgen_verify_real_r8_dim1("vbot", dtype_check_status, var%vbot, ref_var%vbot)
            CALL kgen_verify_real_r8_dim2("qbot", dtype_check_status, var%qbot, ref_var%qbot)
            CALL kgen_verify_real_r8_dim1("pbot", dtype_check_status, var%pbot, ref_var%pbot)
            CALL kgen_verify_real_r8_dim1("rho", dtype_check_status, var%rho, ref_var%rho)
            CALL kgen_verify_real_r8_dim1("netsw", dtype_check_status, var%netsw, ref_var%netsw)
            CALL kgen_verify_real_r8_dim1("flwds", dtype_check_status, var%flwds, ref_var%flwds)
            CALL kgen_verify_real_r8_dim1("precsc", dtype_check_status, var%precsc, ref_var%precsc)
            CALL kgen_verify_real_r8_dim1("precsl", dtype_check_status, var%precsl, ref_var%precsl)
            CALL kgen_verify_real_r8_dim1("precc", dtype_check_status, var%precc, ref_var%precc)
            CALL kgen_verify_real_r8_dim1("precl", dtype_check_status, var%precl, ref_var%precl)
            CALL kgen_verify_real_r8_dim1("soll", dtype_check_status, var%soll, ref_var%soll)
            CALL kgen_verify_real_r8_dim1("sols", dtype_check_status, var%sols, ref_var%sols)
            CALL kgen_verify_real_r8_dim1("solld", dtype_check_status, var%solld, ref_var%solld)
            CALL kgen_verify_real_r8_dim1("solsd", dtype_check_status, var%solsd, ref_var%solsd)
            CALL kgen_verify_real_r8_dim1("thbot", dtype_check_status, var%thbot, ref_var%thbot)
            CALL kgen_verify_real_r8_dim1("co2prog", dtype_check_status, var%co2prog, ref_var%co2prog)
            CALL kgen_verify_real_r8_dim1("co2diag", dtype_check_status, var%co2diag, ref_var%co2diag)
            CALL kgen_verify_real_r8_dim1("psl", dtype_check_status, var%psl, ref_var%psl)
            CALL kgen_verify_real_r8_dim1("bcphiwet", dtype_check_status, var%bcphiwet, ref_var%bcphiwet)
            CALL kgen_verify_real_r8_dim1("bcphidry", dtype_check_status, var%bcphidry, ref_var%bcphidry)
            CALL kgen_verify_real_r8_dim1("bcphodry", dtype_check_status, var%bcphodry, ref_var%bcphodry)
            CALL kgen_verify_real_r8_dim1("ocphiwet", dtype_check_status, var%ocphiwet, ref_var%ocphiwet)
            CALL kgen_verify_real_r8_dim1("ocphidry", dtype_check_status, var%ocphidry, ref_var%ocphidry)
            CALL kgen_verify_real_r8_dim1("ocphodry", dtype_check_status, var%ocphodry, ref_var%ocphodry)
            CALL kgen_verify_real_r8_dim1("dstwet1", dtype_check_status, var%dstwet1, ref_var%dstwet1)
            CALL kgen_verify_real_r8_dim1("dstdry1", dtype_check_status, var%dstdry1, ref_var%dstdry1)
            CALL kgen_verify_real_r8_dim1("dstwet2", dtype_check_status, var%dstwet2, ref_var%dstwet2)
            CALL kgen_verify_real_r8_dim1("dstdry2", dtype_check_status, var%dstdry2, ref_var%dstdry2)
            CALL kgen_verify_real_r8_dim1("dstwet3", dtype_check_status, var%dstwet3, ref_var%dstwet3)
            CALL kgen_verify_real_r8_dim1("dstdry3", dtype_check_status, var%dstdry3, ref_var%dstdry3)
            CALL kgen_verify_real_r8_dim1("dstwet4", dtype_check_status, var%dstwet4, ref_var%dstwet4)
            CALL kgen_verify_real_r8_dim1("dstdry4", dtype_check_status, var%dstdry4, ref_var%dstdry4)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_cam_in_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(cam_in_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_integer("lchnk", dtype_check_status, var%lchnk, ref_var%lchnk)
            CALL kgen_verify_integer("ncol", dtype_check_status, var%ncol, ref_var%ncol)
            CALL kgen_verify_real_r8_dim1("asdir", dtype_check_status, var%asdir, ref_var%asdir)
            CALL kgen_verify_real_r8_dim1("asdif", dtype_check_status, var%asdif, ref_var%asdif)
            CALL kgen_verify_real_r8_dim1("aldir", dtype_check_status, var%aldir, ref_var%aldir)
            CALL kgen_verify_real_r8_dim1("aldif", dtype_check_status, var%aldif, ref_var%aldif)
            CALL kgen_verify_real_r8_dim1("lwup", dtype_check_status, var%lwup, ref_var%lwup)
            CALL kgen_verify_real_r8_dim1("lhf", dtype_check_status, var%lhf, ref_var%lhf)
            CALL kgen_verify_real_r8_dim1("shf", dtype_check_status, var%shf, ref_var%shf)
            CALL kgen_verify_real_r8_dim1("wsx", dtype_check_status, var%wsx, ref_var%wsx)
            CALL kgen_verify_real_r8_dim1("wsy", dtype_check_status, var%wsy, ref_var%wsy)
            CALL kgen_verify_real_r8_dim1("tref", dtype_check_status, var%tref, ref_var%tref)
            CALL kgen_verify_real_r8_dim1("qref", dtype_check_status, var%qref, ref_var%qref)
            CALL kgen_verify_real_r8_dim1("u10", dtype_check_status, var%u10, ref_var%u10)
            CALL kgen_verify_real_r8_dim1("ts", dtype_check_status, var%ts, ref_var%ts)
            CALL kgen_verify_real_r8_dim1("sst", dtype_check_status, var%sst, ref_var%sst)
            CALL kgen_verify_real_r8_dim1("snowhland", dtype_check_status, var%snowhland, ref_var%snowhland)
            CALL kgen_verify_real_r8_dim1("snowhice", dtype_check_status, var%snowhice, ref_var%snowhice)
            CALL kgen_verify_real_r8_dim1("fco2_lnd", dtype_check_status, var%fco2_lnd, ref_var%fco2_lnd)
            CALL kgen_verify_real_r8_dim1("fco2_ocn", dtype_check_status, var%fco2_ocn, ref_var%fco2_ocn)
            CALL kgen_verify_real_r8_dim1("fdms", dtype_check_status, var%fdms, ref_var%fdms)
            CALL kgen_verify_real_r8_dim1("landfrac", dtype_check_status, var%landfrac, ref_var%landfrac)
            CALL kgen_verify_real_r8_dim1("icefrac", dtype_check_status, var%icefrac, ref_var%icefrac)
            CALL kgen_verify_real_r8_dim1("ocnfrac", dtype_check_status, var%ocnfrac, ref_var%ocnfrac)
            CALL kgen_verify_real_r8_dim1_ptr("ram1", dtype_check_status, var%ram1, ref_var%ram1)
            CALL kgen_verify_real_r8_dim1_ptr("fv", dtype_check_status, var%fv, ref_var%fv)
            CALL kgen_verify_real_r8_dim1_ptr("soilw", dtype_check_status, var%soilw, ref_var%soilw)
            CALL kgen_verify_real_r8_dim2("cflx", dtype_check_status, var%cflx, ref_var%cflx)
            CALL kgen_verify_real_r8_dim1("ustar", dtype_check_status, var%ustar, ref_var%ustar)
            CALL kgen_verify_real_r8_dim1("re", dtype_check_status, var%re, ref_var%re)
            CALL kgen_verify_real_r8_dim1("ssq", dtype_check_status, var%ssq, ref_var%ssq)
            CALL kgen_verify_real_r8_dim2_ptr("depvel", dtype_check_status, var%depvel, ref_var%depvel)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
            SUBROUTINE kgen_verify_integer( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer, intent(in) :: var, ref_var
                check_status%numTotal = check_status%numTotal + 1
                IF ( var == ref_var ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                    endif
                ELSE
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        if(check_status%verboseLevel > 2) then
                            WRITE(*,*) "KERNEL: ", var
                            WRITE(*,*) "REF.  : ", ref_var
                        end if
                    end if
                    check_status%numFatal = check_status%numFatal + 1
                END IF
            END SUBROUTINE kgen_verify_integer

            SUBROUTINE kgen_verify_real_r8_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:) :: temp, temp2
                integer :: n
                check_status%numTotal = check_status%numTotal + 1
                IF ( ALL( var == ref_var ) ) THEN
                
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    allocate(temp(SIZE(var,dim=1)))
                    allocate(temp2(SIZE(var,dim=1)))
                
                    n = count(var/=ref_var)
                    where(abs(ref_var) > check_status%minvalue)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
                
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
                
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                
                    deallocate(temp,temp2)
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim1

            SUBROUTINE kgen_verify_real_r8_dim2( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:) :: temp, temp2
                integer :: n
                check_status%numTotal = check_status%numTotal + 1
                IF ( ALL( var == ref_var ) ) THEN
                
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2)))
                
                    n = count(var/=ref_var)
                    where(abs(ref_var) > check_status%minvalue)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
                
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
                
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                
                    deallocate(temp,temp2)
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim2

            SUBROUTINE kgen_verify_real_r8_dim1_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:), POINTER :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:) :: temp, temp2
                integer :: n
                IF ( ASSOCIATED(var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                IF ( ALL( var == ref_var ) ) THEN
                
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    allocate(temp(SIZE(var,dim=1)))
                    allocate(temp2(SIZE(var,dim=1)))
                
                    n = count(var/=ref_var)
                    where(abs(ref_var) > check_status%minvalue)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
                
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
                
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                
                    deallocate(temp,temp2)
                END IF
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim1_ptr

            SUBROUTINE kgen_verify_real_r8_dim2_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:), POINTER :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:) :: temp, temp2
                integer :: n
                IF ( ASSOCIATED(var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                IF ( ALL( var == ref_var ) ) THEN
                
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2)))
                
                    n = count(var/=ref_var)
                    where(abs(ref_var) > check_status%minvalue)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
                
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
                
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                
                    deallocate(temp,temp2)
                END IF
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim2_ptr

        !===============================================================================
        !-----------------------------------------------------------------------
        !
        ! BOP
        !
        ! !IROUTINE: hub2atm_alloc
        !
        ! !DESCRIPTION:
        !
        !   Allocate space for the surface to atmosphere data type. And initialize
        !   the values.
        !
        !-----------------------------------------------------------------------
        !
        ! !INTERFACE
        !

        !
        !===============================================================================
        !
        !-----------------------------------------------------------------------
        !
        ! BOP
        !
        ! !IROUTINE: atm2hub_alloc
        !
        ! !DESCRIPTION:
        !
        !   Allocate space for the atmosphere to surface data type. And initialize
        !   the values.
        !
        !-----------------------------------------------------------------------
        !
        ! !INTERFACE
        !



        !======================================================================
        !
        ! BOP
        !
        ! !IROUTINE: hub2atm_setopts
        !
        ! !DESCRIPTION:
        !
        !   Method for outside packages to influence what is allocated
        !   (For now, just aerosol dust controls if fv, ram1, and soilw
        !   arrays are allocated.)
        !
        !-----------------------------------------------------------------------
        !
        ! !INTERFACE
        !


    END MODULE camsrfexch
