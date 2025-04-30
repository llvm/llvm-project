
! KGEN-generated Fortran source file
!
! Filename    : radlw.F90
! Generated at: 2015-07-06 23:28:43
! KGEN version: 0.4.13



    MODULE radlw
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE rrtmg_state, ONLY : kgen_read_mod31 => kgen_read
    USE rrtmg_state, ONLY : kgen_verify_mod31 => kgen_verify
        !-----------------------------------------------------------------------
        !
        ! Purpose: Longwave radiation calculations.
        !
        !-----------------------------------------------------------------------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        USE ppgrid, ONLY: pcols
        USE ppgrid, ONLY: pverp
        USE parrrtm, ONLY: ngptlw
        USE parrrtm, ONLY: nbndlw
        USE rrtmg_lw_rad, ONLY: rrtmg_lw
        IMPLICIT NONE
        PRIVATE
        PUBLIC rad_rrtmg_lw
        integer, parameter :: maxiter = 100
        character(len=80), parameter :: kname = "rrtmg_lw"
        ! Public methods
        ! initialize constants
        ! driver for longwave radiation code
        ! Private data
        ! top level to solve for longwave cooling
        !===============================================================================
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !===============================================================================

        SUBROUTINE rad_rrtmg_lw(lchnk, ncol, rrtmg_levs, r_state, kgen_unit)
                USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
            !-----------------------------------------------------------------------
            USE rrtmg_state, ONLY: rrtmg_state_t
            !------------------------------Arguments--------------------------------
            !
            ! Input arguments
            !
            integer, intent(in) :: kgen_unit
            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            INTEGER, intent(in) :: lchnk ! chunk identifier
            INTEGER, intent(in) :: ncol ! number of atmospheric columns
            INTEGER, intent(in) :: rrtmg_levs ! number of levels rad is applied
            !
            ! Input arguments which are only passed to other routines
            !
            TYPE(rrtmg_state_t), intent(in) :: r_state
            ! Level pressure (Pascals)
            ! aerosol absorption optics depth (LW)
            ! Cloud cover
            ! Cloud longwave optical depth by band
            !
            ! Output arguments
            !
            ! Longwave heating rate
            ! Clearsky longwave heating rate
            ! Surface cooling flux
            ! Net outgoing flux
            ! Upward flux at top of model
            ! Clear sky surface cooing
            ! Net clear sky outgoing flux
            ! Upward clear-sky flux at top of model
            ! Down longwave flux at surface
            ! Down longwave clear flux at surface
            ! clear sky net flux at interfaces
            ! net flux at interfaces
            ! longwave spectral flux up
            ! longwave spectral flux down
            !
            !---------------------------Local variables-----------------------------
            !
            ! indices
            ! Total upwards longwave flux
            ! Clear sky upwards longwave flux
            ! Total downwards longwave flux
            ! Clear sky downwards longwv flux
            INTEGER :: inflglw ! Flag for cloud parameterization method
            INTEGER :: iceflglw ! Flag for ice cloud param method
            INTEGER :: liqflglw ! Flag for liquid cloud param method
            INTEGER :: icld
            INTEGER :: ref_icld ! Flag for cloud overlap method
            ! 0=clear, 1=random, 2=maximum/random, 3=maximum
            REAL(KIND=r8) :: tsfc(pcols) ! surface temperature
            REAL(KIND=r8) :: emis(pcols,nbndlw) ! surface emissivity
            REAL(KIND=r8) :: taua_lw(pcols,rrtmg_levs-1,nbndlw) ! aerosol optical depth by band
            ! Inverse of seconds per day
            ! Cloud arrays for McICA
            INTEGER, parameter :: nsubclw = ngptlw ! rrtmg_lw g-point (quadrature point) dimension
            ! permute seed for sub-column generator
            ! in-cloud cloud ice water path
            ! in-cloud cloud liquid water path
            REAL(KIND=r8) :: rei(pcols,rrtmg_levs-1) ! ice particle effective radius (microns)
            REAL(KIND=r8) :: rel(pcols,rrtmg_levs-1) ! liquid particle radius (micron)
            REAL(KIND=r8) :: cld_stolw(nsubclw, pcols, rrtmg_levs-1) ! cloud fraction (mcica)
            REAL(KIND=r8) :: cicewp_stolw(nsubclw, pcols, rrtmg_levs-1) ! cloud ice water path (mcica)
            REAL(KIND=r8) :: cliqwp_stolw(nsubclw, pcols, rrtmg_levs-1) ! cloud liquid water path (mcica)
            ! ice particle size (mcica)
            ! liquid particle size (mcica)
            REAL(KIND=r8) :: tauc_stolw(nsubclw, pcols, rrtmg_levs-1) ! cloud optical depth (mcica - optional)
            ! Includes extra layer above model top
            REAL(KIND=r8) :: uflx(pcols,rrtmg_levs+1)
            REAL(KIND=r8) :: ref_uflx(pcols,rrtmg_levs+1) ! Total upwards longwave flux
            REAL(KIND=r8) :: uflxc(pcols,rrtmg_levs+1)
            REAL(KIND=r8) :: ref_uflxc(pcols,rrtmg_levs+1) ! Clear sky upwards longwave flux
            REAL(KIND=r8) :: dflx(pcols,rrtmg_levs+1)
            REAL(KIND=r8) :: ref_dflx(pcols,rrtmg_levs+1) ! Total downwards longwave flux
            REAL(KIND=r8) :: dflxc(pcols,rrtmg_levs+1)
            REAL(KIND=r8) :: ref_dflxc(pcols,rrtmg_levs+1) ! Clear sky downwards longwv flux
            REAL(KIND=r8) :: hr(pcols,rrtmg_levs)
            REAL(KIND=r8) :: ref_hr(pcols,rrtmg_levs) ! Longwave heating rate (K/d)
            REAL(KIND=r8) :: hrc(pcols,rrtmg_levs)
            REAL(KIND=r8) :: ref_hrc(pcols,rrtmg_levs) ! Clear sky longwave heating rate (K/d)
            REAL(KIND=r8) :: lwuflxs(nbndlw,pcols,pverp+1)
            REAL(KIND=r8) :: ref_lwuflxs(nbndlw,pcols,pverp+1) ! Longwave spectral flux up
            REAL(KIND=r8) :: lwdflxs(nbndlw,pcols,pverp+1)
            REAL(KIND=r8) :: ref_lwdflxs(nbndlw,pcols,pverp+1) ! Longwave spectral flux down
            !-----------------------------------------------------------------------
            ! mji/rrtmg
            ! Calculate cloud optical properties here if using CAM method, or if using one of the
            ! methods in RRTMG_LW, then pass in cloud physical properties and zero out cloud optical
            ! properties here
            ! Zero optional cloud optical depth input array tauc_lw,
            ! if inputting cloud physical properties into RRTMG_LW
            !          tauc_lw(:,:,:) = 0.
            ! Or, pass in CAM cloud longwave optical depth to RRTMG_LW
            ! do nbnd = 1, nbndlw
            !    tauc_lw(nbnd,:ncol,:pver) = cldtau(:ncol,:pver)
            ! end do
            ! Call mcica sub-column generator for RRTMG_LW
            ! Call sub-column generator for McICA in radiation
            ! Select cloud overlap approach (1=random, 2=maximum-random, 3=maximum)
            ! Set permute seed (must be offset between LW and SW by at least 140 to insure
            ! effective randomization)
            ! These fields are no longer supplied by CAM.
            !
            ! Call RRTMG_LW model
            !
            ! Set input flags for cloud parameterizations
            ! Use separate specification of ice and liquid cloud optical depth.
            ! Use either Ebert and Curry ice parameterization (iceflglw = 0 or 1),
            ! or use Key (Streamer) approach (iceflglw = 2), or use Fu method
            ! (iceflglw = 3), and Hu/Stamnes for liquid (liqflglw = 1).
            ! For use in Fu method (iceflglw = 3), rei is converted in RRTMG_LW
            ! from effective radius to generalized effective size using the
            ! conversion of D. Mitchell, JAS, 2002.  For ice particles outside
            ! the effective range of either the Key or Fu approaches, the
            ! Ebert and Curry method is applied.
            ! Input CAM cloud optical depth directly
            ! Use E&C approach for ice to mimic CAM3
            !   inflglw = 2
            !   iceflglw = 1
            !   liqflglw = 1
            ! Use merged Fu and E&C params for ice
            !   inflglw = 2
            !   iceflglw = 3
            !   liqflglw = 1
            ! Convert incoming water amounts from specific humidity to vmr as needed;
            ! Convert other incoming molecular amounts from mmr to vmr as needed;
            ! Convert pressures from Pa to hPa;
            ! Set surface emissivity to 1.0 here, this is treated in land surface model;
            ! Set surface temperature
            ! Set aerosol optical depth to zero for now
            tolerance = 5.E-13
            CALL kgen_init_check(check_status, tolerance)
            READ(UNIT=kgen_unit) inflglw
            READ(UNIT=kgen_unit) iceflglw
            READ(UNIT=kgen_unit) liqflglw
            READ(UNIT=kgen_unit) icld
            READ(UNIT=kgen_unit) tsfc
            READ(UNIT=kgen_unit) emis
            READ(UNIT=kgen_unit) taua_lw
            READ(UNIT=kgen_unit) rei
            READ(UNIT=kgen_unit) rel
            READ(UNIT=kgen_unit) cld_stolw
            READ(UNIT=kgen_unit) cicewp_stolw
            READ(UNIT=kgen_unit) cliqwp_stolw
            READ(UNIT=kgen_unit) tauc_stolw
            READ(UNIT=kgen_unit) uflx
            READ(UNIT=kgen_unit) uflxc
            READ(UNIT=kgen_unit) dflx
            READ(UNIT=kgen_unit) dflxc
            READ(UNIT=kgen_unit) hr
            READ(UNIT=kgen_unit) hrc
            READ(UNIT=kgen_unit) lwuflxs
            READ(UNIT=kgen_unit) lwdflxs

            READ(UNIT=kgen_unit) ref_icld
            READ(UNIT=kgen_unit) ref_uflx
            READ(UNIT=kgen_unit) ref_uflxc
            READ(UNIT=kgen_unit) ref_dflx
            READ(UNIT=kgen_unit) ref_dflxc
            READ(UNIT=kgen_unit) ref_hr
            READ(UNIT=kgen_unit) ref_hrc
            READ(UNIT=kgen_unit) ref_lwuflxs
            READ(UNIT=kgen_unit) ref_lwdflxs


            ! call to kernel
   print *,'lchnk: ',lchnk
   print *,'ncol: ',ncol
   print *,'nbndlw: ',nbndlw
   print *,'ngptw: ',ngptlw
   print *,'rrtmg_levs: ',rrtmg_levs
   call rrtmg_lw(lchnk  ,ncol ,rrtmg_levs    ,icld    ,                 &
        r_state%pmidmb  ,r_state%pintmb  ,r_state%tlay    ,r_state%tlev    ,tsfc    ,r_state%h2ovmr, &
        r_state%o3vmr   ,r_state%co2vmr  ,r_state%ch4vmr  ,r_state%o2vmr   ,r_state%n2ovmr  ,r_state%cfc11vmr,r_state%cfc12vmr, &
        r_state%cfc22vmr,r_state%ccl4vmr ,emis    ,inflglw ,iceflglw,liqflglw, &
        cld_stolw,tauc_stolw,cicewp_stolw,cliqwp_stolw ,rei, rel, &
        taua_lw, &
        uflx    ,dflx    ,hr      ,uflxc   ,dflxc   ,hrc, &
        lwuflxs, lwdflxs)
            ! kernel verification for output variables
            CALL kgen_verify_integer( "icld", check_status, icld, ref_icld)
            CALL kgen_verify_real_r8_dim2( "uflx", check_status, uflx, ref_uflx)
            CALL kgen_verify_real_r8_dim2( "uflxc", check_status, uflxc, ref_uflxc)
            CALL kgen_verify_real_r8_dim2( "dflx", check_status, dflx, ref_dflx)
            CALL kgen_verify_real_r8_dim2( "dflxc", check_status, dflxc, ref_dflxc)
            CALL kgen_verify_real_r8_dim2( "hr", check_status, hr, ref_hr)
            CALL kgen_verify_real_r8_dim2( "hrc", check_status, hrc, ref_hrc)
            CALL kgen_verify_real_r8_dim3( "lwuflxs", check_status, lwuflxs, ref_lwuflxs)
            CALL kgen_verify_real_r8_dim3( "lwdflxs", check_status, lwdflxs, ref_lwdflxs)
            CALL kgen_print_check("rrtmg_lw", check_status)
            CALL system_clock(start_clock, rate_clock)
            DO kgen_intvar=1,maxiter
                CALL rrtmg_lw(lchnk, ncol, rrtmg_levs, icld, r_state % pmidmb, r_state % pintmb, r_state % tlay, &
r_state % tlev, tsfc, r_state % h2ovmr, r_state % o3vmr, r_state % co2vmr, r_state % ch4vmr, r_state % o2vmr, &
r_state % n2ovmr, r_state % cfc11vmr, r_state % cfc12vmr, r_state % cfc22vmr, r_state % ccl4vmr, emis, inflglw, &
iceflglw, liqflglw, cld_stolw, tauc_stolw, cicewp_stolw, cliqwp_stolw, rei, rel, taua_lw, uflx, dflx, hr, uflxc, &
dflxc, hrc, lwuflxs, lwdflxs)
            END DO
            CALL system_clock(stop_clock, rate_clock)
            WRITE(*,*)
            PRINT *, TRIM(kname), ": Elapsed time (usec): ", 1.0e6*(stop_clock - start_clock)/REAL(rate_clock*maxiter)
            !
            !----------------------------------------------------------------------
            ! All longitudes: store history tape quantities
            ! Flux units are in W/m2 on output from rrtmg_lw and contain output for
            ! extra layer above model top with vertical indexing from bottom to top.
            ! Heating units are in K/d on output from RRTMG and contain output for
            ! extra layer above model top with vertical indexing from bottom to top.
            ! Heating units are converted to J/kg/s below for use in CAM.
            !
            ! Reverse vertical indexing here for CAM arrays to go from top to bottom.
            !
            ! mji/ cam excluded this?
            ! Pass longwave heating to CAM arrays and convert from K/d to J/kg/s
            ! Return 0 above solution domain
            ! Pass spectral fluxes, reverse layering
            ! order=(/3,1,2/) maps the first index of lwuflxs to the third index of lu.
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

        END SUBROUTINE rad_rrtmg_lw
        !-------------------------------------------------------------------------------

        !-------------------------------------------------------------------------------
    END MODULE radlw
