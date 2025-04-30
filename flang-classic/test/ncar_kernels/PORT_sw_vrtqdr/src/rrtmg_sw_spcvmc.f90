
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_sw_spcvmc.f90
! Generated at: 2015-07-31 21:01:00
! KGEN version: 0.4.13



    MODULE rrtmg_sw_spcvmc
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !  --------------------------------------------------------------------------
        ! |                                                                          |
        ! |  Copyright 2002-2007, Atmospheric & Environmental Research, Inc. (AER).  |
        ! |  This software may be used, copied, or redistributed as long as it is    |
        ! |  not sold and this copyright notice is reproduced on each copy made.     |
        ! |  This model is provided as is without any express or implied warranties. |
        ! |                       (http://www.rtweb.aer.com/)                        |
        ! |                                                                          |
        !  --------------------------------------------------------------------------
        ! ------- Modules -------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        USE parrrsw, ONLY: ngptsw
        USE rrtmg_sw_vrtqdr, ONLY: vrtqdr_sw
        IMPLICIT NONE
        PUBLIC spcvmc_sw
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        ! ---------------------------------------------------------------------------

        SUBROUTINE spcvmc_sw(nlayers, ncol, kgen_unit)
                USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
            ! ---------------------------------------------------------------------------
            !
            ! Purpose: Contains spectral loop to compute the shortwave radiative fluxes,
            !          using the two-stream method of H. Barker and McICA, the Monte-Carlo
            !          Independent Column Approximation, for the representation of
            !          sub-grid cloud variability (i.e. cloud overlap).
            !
            ! Interface:  *spcvmc_sw* is called from *rrtmg_sw.F90* or rrtmg_sw.1col.F90*
            !
            ! Method:
            !    Adapted from two-stream model of H. Barker;
            !    Two-stream model options (selected with kmodts in rrtmg_sw_reftra.F90):
            !        1: Eddington, 2: PIFM, Zdunkowski et al., 3: discret ordinates
            !
            ! Modifications:
            !
            ! Original: H. Barker
            ! Revision: Merge with RRTMG_SW: J.-J.Morcrette, ECMWF, Feb 2003
            ! Revision: Add adjustment for Earth/Sun distance : MJIacono, AER, Oct 2003
            ! Revision: Bug fix for use of PALBP and PALBD: MJIacono, AER, Nov 2003
            ! Revision: Bug fix to apply delta scaling to clear sky: AER, Dec 2004
            ! Revision: Code modified so that delta scaling is not done in cloudy profiles
            !           if routine cldprop is used; delta scaling can be applied by swithcing
            !           code below if cldprop is not used to get cloud properties.
            !           AER, Jan 2005
            ! Revision: Modified to use McICA: MJIacono, AER, Nov 2005
            ! Revision: Uniform formatting for RRTMG: MJIacono, AER, Jul 2006
            ! Revision: Use exponential lookup table for transmittance: MJIacono, AER,
            !           Aug 2007
            !
            ! ------------------------------------------------------------------
            ! ------- Declarations ------
            ! ------- Input -------
            integer, intent(in) :: kgen_unit
            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            INTEGER, intent(in) :: nlayers
            ! delta-m scaling flag
            ! [0 = direct and diffuse fluxes are unscaled]
            ! [1 = direct and diffuse fluxes are scaled]
            INTEGER, intent(in) :: ncol ! column loop index
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            ! layer pressure (hPa, mb)
            !   Dimensions: (ncol,nlayers)
            ! layer temperature (K)
            !   Dimensions: (ncol,nlayers)
            ! level (interface) pressure (hPa, mb)
            !   Dimensions: (ncol,0:nlayers)
            ! level temperatures (hPa, mb)
            !   Dimensions: (ncol,0:nlayers)
            ! surface temperature (K)
            ! molecular amounts (mol/cm2)
            !   Dimensions: (ncol,mxmol,nlayers)
            ! dry air column density (mol/cm2)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            ! Earth/Sun distance adjustment
            !   Dimensions: (ncol,jpband)
            ! surface albedo (diffuse)
            !   Dimensions: (ncol,nbndsw)
            ! surface albedo (direct)
            !   Dimensions: (ncol, nbndsw)
            ! cosine of solar zenith angle
            ! cloud fraction [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            ! cloud optical depth [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            ! cloud asymmetry parameter [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            ! cloud single scattering albedo [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            ! cloud optical depth, non-delta scaled [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            ! aerosol optical depth
            !   Dimensions: (ncol,nlayers,nbndsw)
            ! aerosol asymmetry parameter
            !   Dimensions: (ncol,nlayers,nbndsw)
            ! aerosol single scattering albedo
            !   Dimensions: (ncol,nlayers,nbndsw)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            !   Dimensions: (ncol,nlayers)
            ! ------- Output -------
            !   All Dimensions: (nlayers+1)
            ! Added for net near-IR flux diagnostic
            ! Output - inactive                                              !   All Dimensions: (nlayers+1)
            !      real(kind=r8), intent(out) :: puvcu(:)
            !      real(kind=r8), intent(out) :: puvfu(:)
            !      real(kind=r8), intent(out) :: pvscd(:)
            !      real(kind=r8), intent(out) :: pvscu(:)
            !      real(kind=r8), intent(out) :: pvsfd(:)
            !      real(kind=r8), intent(out) :: pvsfu(:)
            ! shortwave spectral flux up (nswbands,nlayers+1)
            ! shortwave spectral flux down (nswbands,nlayers+1)
            ! ------- Local -------
            INTEGER :: klev
            INTEGER :: iw(ncol)
            !      integer, parameter :: nuv = ??
            !      integer, parameter :: nvs = ??
            REAL(KIND=r8) :: zrdndc(ncol,nlayers+1)
            REAL(KIND=r8) :: ref_zrdndc(ncol,nlayers+1)
            REAL(KIND=r8) :: zrefc(ncol,nlayers+1)
            REAL(KIND=r8) :: zrefdc(ncol,nlayers+1)
            REAL(KIND=r8) :: zrupc(ncol,nlayers+1)
            REAL(KIND=r8) :: ref_zrupc(ncol,nlayers+1)
            REAL(KIND=r8) :: zrupdc(ncol,nlayers+1)
            REAL(KIND=r8) :: ref_zrupdc(ncol,nlayers+1)
            REAL(KIND=r8) :: ztrac(ncol,nlayers+1)
            REAL(KIND=r8) :: ztradc(ncol,nlayers+1)
            REAL(KIND=r8) :: ztdbtc(ncol,nlayers+1)
            REAL(KIND=r8) :: zdbtc(ncol,nlayers+1)
            !     real(kind=r8) :: zincflux                                   ! inactive
            ! Arrays from rrtmg_sw_taumoln routines
            !      real(kind=r8) :: ztaug(nlayers,16), ztaur(nlayers,16)
            !      real(kind=r8) :: zsflxzen(16)
            ! Arrays from rrtmg_sw_vrtqdr routine
            REAL(KIND=r8) :: zcd(ncol,nlayers+1,ngptsw)
            REAL(KIND=r8) :: ref_zcd(ncol,nlayers+1,ngptsw)
            REAL(KIND=r8) :: zcu(ncol,nlayers+1,ngptsw)
            REAL(KIND=r8) :: ref_zcu(ncol,nlayers+1,ngptsw)
            ! Inactive arrays
            !     real(kind=r8) :: zbbcd(nlayers+1), zbbcu(nlayers+1)
            !     real(kind=r8) :: zbbfd(nlayers+1), zbbfu(nlayers+1)
            !     real(kind=r8) :: zbbfddir(nlayers+1), zbbcddir(nlayers+1)
            ! ------------------------------------------------------------------
            ! Initializations
            !      zincflux = 0.0_r8
            ! ??? ! ???
                    tolerance = 1.E-14
                    CALL kgen_init_check(check_status, tolerance)
                    READ(UNIT=kgen_unit) klev
                    READ(UNIT=kgen_unit) iw
                    READ(UNIT=kgen_unit) zrdndc
                    READ(UNIT=kgen_unit) zrefc
                    READ(UNIT=kgen_unit) zrefdc
                    READ(UNIT=kgen_unit) zrupc
                    READ(UNIT=kgen_unit) zrupdc
                    READ(UNIT=kgen_unit) ztrac
                    READ(UNIT=kgen_unit) ztradc
                    READ(UNIT=kgen_unit) ztdbtc
                    READ(UNIT=kgen_unit) zdbtc
                    READ(UNIT=kgen_unit) zcd
                    READ(UNIT=kgen_unit) zcu

                    READ(UNIT=kgen_unit) ref_zrdndc
                    READ(UNIT=kgen_unit) ref_zrupc
                    READ(UNIT=kgen_unit) ref_zrupdc
                    READ(UNIT=kgen_unit) ref_zcd
                    READ(UNIT=kgen_unit) ref_zcu


                    ! call to kernel
            		call vrtqdr_sw(ncol,klev, iw, &
zrefc, zrefdc, ztrac, ztradc, &
zdbtc, zrdndc, zrupc, zrupdc, ztdbtc, &
zcd, zcu)
                    ! kernel verification for output variables
                    CALL kgen_verify_real_r8_dim2( "zrdndc", check_status, zrdndc, ref_zrdndc)
                    CALL kgen_verify_real_r8_dim2( "zrupc", check_status, zrupc, ref_zrupc)
                    CALL kgen_verify_real_r8_dim2( "zrupdc", check_status, zrupdc, ref_zrupdc)
                    CALL kgen_verify_real_r8_dim3( "zcd", check_status, zcd, ref_zcd)
                    CALL kgen_verify_real_r8_dim3( "zcu", check_status, zcu, ref_zcu)
                    CALL kgen_print_check("vrtqdr_sw", check_status)
                    CALL system_clock(start_clock, rate_clock)
                    DO kgen_intvar=1,10
                        CALL vrtqdr_sw(ncol, klev, iw, zrefc, zrefdc, ztrac, ztradc, zdbtc, zrdndc, zrupc, zrupdc, ztdbtc, zcd, zcu)
                    END DO
                    CALL system_clock(stop_clock, rate_clock)
                    WRITE(*,*)
                    PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim2(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: var
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
            END SUBROUTINE kgen_read_real_r8_dim2

            SUBROUTINE kgen_read_real_r8_dim3(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2,idx3
                INTEGER, DIMENSION(2,3) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    READ(UNIT = kgen_unit) kgen_bound(1, 3)
                    READ(UNIT = kgen_unit) kgen_bound(2, 3)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim3


        ! verify subroutines
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

            SUBROUTINE kgen_verify_real_r8_dim3( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:,:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:,:) :: temp, temp2
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
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                
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
            END SUBROUTINE kgen_verify_real_r8_dim3

        END SUBROUTINE spcvmc_sw
    END MODULE rrtmg_sw_spcvmc
