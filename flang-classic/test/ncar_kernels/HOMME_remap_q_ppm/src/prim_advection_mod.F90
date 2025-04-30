
! KGEN-generated Fortran source file
!
! Filename    : prim_advection_mod.F90
! Generated at: 2015-02-24 15:34:48
! KGEN version: 0.4.4



    MODULE vertremap_mod
        !**************************************************************************************
        !
        !  Purpose:
        !        Construct sub-grid-scale polynomials using piecewise spline method with
        !        monotone filters.
        !
        !  References: PCM - Zerroukat et al., Q.J.R. Meteorol. Soc., 2005. (ZWS2005QJR)
        !              PSM - Zerroukat et al., Int. J. Numer. Meth. Fluids, 2005. (ZWS2005IJMF)
        !
        !**************************************************************************************
        USE kinds, ONLY: real_kind
        USE dimensions_mod, ONLY: nlev
        USE perf_mod, ONLY: t_startf
        USE perf_mod, ONLY: t_stopf ! _EXTERNAL
        INTEGER, PARAMETER :: kgen_dp = selected_real_kind(15, 307)
        PUBLIC remap1
        type, public  ::  check_t
            logical :: Passed
            integer :: numFatal
            integer :: numTotal
            integer :: numIdentical
            integer :: numWarning
            integer :: VerboseLevel
            real(kind=kgen_dp) :: tolerance
        end type check_t
        ! remap any field, splines, monotone
        ! remap any field, splines, no filter
        ! todo: tweak interface to match remap1 above, rename remap1_ppm:
        PUBLIC remap_q_ppm ! remap state%Q, PPM, monotone
        CONTAINS
        subroutine kgen_init_check(check,tolerance)
          type(check_t), intent(inout) :: check
          real(kind=kgen_dp), intent(in), optional :: tolerance
           check%Passed   = .TRUE.
           check%numFatal = 0
           check%numWarning = 0
           check%numTotal = 0
           check%numIdentical = 0
           check%VerboseLevel = 1
           if(present(tolerance)) then
             check%tolerance = tolerance
           else
              check%tolerance = 1.E-14
           endif
        end subroutine kgen_init_check
        subroutine kgen_print_check(kname, check)
           character(len=*) :: kname
           type(check_t), intent(in) ::  check
           write (*,*)
           write (*,*) TRIM(kname),' KGENPrtCheck: Tolerance for normalized RMS: ',check%tolerance
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of variables checked: ',check%numTotal
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of Identical results: ',check%numIdentical
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of warnings detected: ',check%numWarning
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of fatal errors detected: ', check%numFatal
           if (check%numFatal> 0) then
                write(*,*) TRIM(kname),' KGENPrtCheck: verification FAILED'
           else
                write(*,*) TRIM(kname),' KGENPrtCheck: verification PASSED'
           endif
        end subroutine kgen_print_check
        !=======================================================================================================!
        !remap_calc_grids computes the vertical pressures and pressure differences for one vertical column for the reference grid
        !and for the deformed Lagrangian grid. This was pulled out of each routine since it was a repeated task.

        !=======================================================================================================!

        SUBROUTINE remap1(nx, qsize, qdp, dp1, dp2, kgen_unit)
            ! remap 1 field
            ! input:  Qdp   field to be remapped (NOTE: MASS, not MIXING RATIO)
            !         dp1   layer thickness (source)
            !         dp2   layer thickness (target)
            !
            ! output: remaped Qdp, conserving mass, monotone on Q=Qdp/dp
            !
            IMPLICIT NONE
            integer, intent(in) :: kgen_unit

            ! read interface
            interface kgen_read_var
                procedure read_var_real_real_kind_dim4
            end interface kgen_read_var



            ! verification interface
            interface kgen_verify_var
                procedure verify_var_logical
                procedure verify_var_integer
                procedure verify_var_real
                procedure verify_var_character
                procedure verify_var_real_real_kind_dim4
            end interface kgen_verify_var

            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            INTEGER, intent(in) :: nx
            INTEGER, intent(in) :: qsize
            REAL(KIND=real_kind), intent(inout) :: qdp(nx,nx,nlev,qsize)
            REAL(KIND=real_kind), allocatable :: ref_qdp(:,:,:,:)
            REAL(KIND=real_kind), intent(in) :: dp1(nx,nx,nlev)
            REAL(KIND=real_kind), intent(in) :: dp2(nx,nx,nlev)
            ! ========================
            ! Local Variables
            ! ========================
                tolerance = 1.E-14
                CALL kgen_init_check(check_status, tolerance)
                ! None
                call kgen_read_var(ref_qdp, kgen_unit)
                ! call to kernel
                CALL remap_q_ppm(qdp, nx, qsize, dp1, dp2)
                ! kernel verification for output variables
                call kgen_verify_var("qdp", check_status, qdp, ref_qdp)
                CALL kgen_print_check("remap_q_ppm", check_status)
                CALL system_clock(start_clock, rate_clock)
                DO kgen_intvar=1,10
                    CALL remap_q_ppm(qdp, nx, qsize, dp1, dp2)
                END DO
                CALL system_clock(stop_clock, rate_clock)
                WRITE(*,*)
                PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
            ! q loop
        CONTAINS

        ! read subroutines
        subroutine read_var_real_real_kind_dim4(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=real_kind), intent(out), dimension(:,:,:,:), allocatable :: var
            integer, dimension(2,4) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                READ(UNIT = kgen_unit) kgen_bound(1, 3)
                READ(UNIT = kgen_unit) kgen_bound(2, 3)
                READ(UNIT = kgen_unit) kgen_bound(1, 4)
                READ(UNIT = kgen_unit) kgen_bound(2, 4)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1, kgen_bound(2, 4) - kgen_bound(1, 4) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine

        subroutine verify_var_logical(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            logical, intent(in) :: var, ref_var
        
            check_status%numTotal = check_status%numTotal + 1
            IF ( var .eqv. ref_var ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                endif
            ELSE
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                    if(check_status%verboseLevel > 2) then
                        WRITE(*,*) "KERNEL: ", var
                        WRITE(*,*) "REF.  : ", ref_var
                    endif
                endif
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine
        
        subroutine verify_var_integer(varname, check_status, var, ref_var)
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
                    endif
                endif
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine
        
        subroutine verify_var_real(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            real, intent(in) :: var, ref_var
        
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
                    endif
                endif
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine
        
        subroutine verify_var_character(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            character(*), intent(in) :: var, ref_var
        
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
        end subroutine

        subroutine verify_var_real_real_kind_dim4(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            real(kind=real_kind), intent(in), dimension(:,:,:,:) :: var
            real(kind=real_kind), intent(in), allocatable, dimension(:,:,:,:) :: ref_var
            real(kind=real_kind) :: nrmsdiff, rmsdiff
            real(kind=real_kind), allocatable :: temp(:,:,:,:), temp2(:,:,:,:)
            integer :: n
        
        
            IF ( ALLOCATED(ref_var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3),SIZE(var,dim=4)))
                allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3),SIZE(var,dim=4)))
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
                    n = count(var/=ref_var)
                    where(ref_var .NE. 0)
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
                END IF
                deallocate(temp,temp2)
            END IF
        end subroutine

        END SUBROUTINE remap1

        !=======================================================================================================!
        !This uses the exact same model and reference grids and data as remap_Q, but it interpolates
        !using PPM instead of splines.

        SUBROUTINE remap_q_ppm(qdp, nx, qsize, dp1, dp2)
            ! remap 1 field
            ! input:  Qdp   field to be remapped (NOTE: MASS, not MIXING RATIO)
            !         dp1   layer thickness (source)
            !         dp2   layer thickness (target)
            !
            ! output: remaped Qdp, conserving mass
            !
            USE control_mod, ONLY: vert_remap_q_alg
            IMPLICIT NONE
            INTEGER, intent(in) :: nx, qsize
            REAL(KIND=real_kind), intent(inout) :: qdp(nx,nx,nlev,qsize)
            REAL(KIND=real_kind), intent(in) :: dp1(nx,nx,nlev), dp2(nx,nx,nlev)
            ! Local Variables
            INTEGER, parameter :: gs = 2 !Number of cells to place in the ghost region
            REAL(KIND=real_kind), dimension(nlev+2) :: pio !Pressure at interfaces for old grid
            REAL(KIND=real_kind), dimension(nlev+1) :: pin !Pressure at interfaces for new grid
            REAL(KIND=real_kind), dimension(nlev+1) :: masso !Accumulate mass up to each interface
            REAL(KIND=real_kind), dimension(1-gs:nlev+gs) :: ao !Tracer value on old grid
            REAL(KIND=real_kind), dimension(1-gs:nlev+gs) :: dpo !change in pressure over a cell for old grid
            REAL(KIND=real_kind), dimension(1-gs:nlev+gs) :: dpn !change in pressure over a cell for old grid
            REAL(KIND=real_kind), dimension(3,     nlev) :: coefs !PPM coefficients within each cell
            REAL(KIND=real_kind), dimension(       nlev   ) :: z1, z2
            REAL(KIND=real_kind) :: ppmdx(10,0:nlev+1) !grid spacings
            REAL(KIND=real_kind) :: mymass, massn1, massn2
            INTEGER :: i, j, k, q, kk, kid(nlev)
            CALL t_startf('remap_Q_ppm')
            DO j = 1 , nx
                DO i = 1 , nx
                    pin(1) = 0
                    pio(1) = 0
                    DO k=1,nlev
                        dpn(k) = dp2(i,j,k)
                        dpo(k) = dp1(i,j,k)
                        pin(k+1) = pin(k)+dpn(k)
                        pio(k+1) = pio(k)+dpo(k)
                    END DO 
                    pio(nlev+2) = pio(nlev+1) + 1. !This is here to allow an entire block of k threads to run in the remapping phase.
                    !It makes sure there's an old interface value below the domain that is larger.
                    pin(nlev+1) = pio(nlev+1) !The total mass in a column does not change.
                    !Therefore, the pressure of that mass cannot either.
                    !Fill in the ghost regions with mirrored values. if vert_remap_q_alg is defined, this is of no consequence.
                    DO k = 1 , gs
                        dpo(1   -k) = dpo(       k)
                        dpo(nlev+k) = dpo(nlev+1-k)
                    END DO 
                    !Compute remapping intervals once for all tracers. Find the old grid cell index in which the
                    !k-th new cell interface resides. Then integrate from the bottom of that old cell to the new
                    !interface location. In practice, the grid never deforms past one cell, so the search can be
                    !simplified by this. Also, the interval of integration is usually of magnitude close to zero
                    !or close to dpo because of minimial deformation.
                    !Numerous tests confirmed that the bottom and top of the grids match to machine precision, so
                    !I set them equal to each other.
                    DO k = 1 , nlev
                        kk = k !Keep from an order n^2 search operation by assuming the old cell index is close.
                        !Find the index of the old grid cell in which this new cell's bottom interface resides.
                        DO while (pio(kk) <= pin(k+1))
                            kk = kk + 1
                        END DO 
                        kk = kk - 1 !kk is now the cell index we're integrating over.
                        IF (kk == nlev+1) kk = nlev !This is to keep the indices in bounds.
                        !Top bounds match anyway, so doesn't matter what coefficients are used
                        kid(k) = kk !Save for reuse
                        z1(k) = -0.5d0 !This remapping assumes we're starting from the left interface of an old grid cell
                        !In fact, we're usually integrating very little or almost all of the cell in question
                        z2(k) = (pin(k+1) - ( pio(kk) + pio(kk+1) ) * 0.5) / dpo(kk) !PPM interpolants are normalized to an independent
                        !coordinate domain [-0.5,0.5].
                    END DO 
                    !This turned out a big optimization, remembering that only parts of the PPM algorithm depends on the data, 
                    ! namely the
                    !limiting. So anything that depends only on the grid is pre-computed outside the tracer loop.
                    ppmdx(:,:) = compute_ppm_grids( dpo )
                    !From here, we loop over tracers for only those portions which depend on tracer data, which includes PPM 
                    ! limiting and
                    !mass accumulation
                    DO q = 1 , qsize
                        !Accumulate the old mass up to old grid cell interface locations to simplify integration
                        !during remapping. Also, divide out the grid spacing so we're working with actual tracer
                        !values and can conserve mass. The option for ifndef ZEROHORZ I believe is there to ensure
                        !tracer consistency for an initially uniform field. I copied it from the old remap routine.
                        masso(1) = 0.
                        DO k = 1 , nlev
                            ao(k) = qdp(i,j,k,q)
                            masso(k+1) = masso(k) + ao(k) !Accumulate the old mass. This will simplify the remapping
                            ao(k) = ao(k) / dpo(k) !Divide out the old grid spacing because we want the tracer mixing ratio, not mass.
                        END DO 
                        !Fill in ghost values. Ignored if vert_remap_q_alg == 2
                        DO k = 1 , gs
                            ao(1   -k) = ao(       k)
                            ao(nlev+k) = ao(nlev+1-k)
                        END DO 
                        !Compute monotonic and conservative PPM reconstruction over every cell
                        coefs(:,:) = compute_ppm(ao , ppmdx)
                        !Compute tracer values on the new grid by integrating from the old cell bottom to the new
                        !cell interface to form a new grid mass accumulation. Taking the difference between
                        !accumulation at successive interfaces gives the mass inside each cell. Since Qdp is
                        !supposed to hold the full mass this needs no normalization.
                        massn1 = 0.
                        DO k = 1 , nlev
                            kk = kid(k)
                            massn2 = masso(kk) + integrate_parabola(coefs(:,kk) , z1(k) , z2(k)) * dpo(kk)
                            qdp(i,j,k,q) = massn2 - massn1
                            massn1 = massn2
                        END DO 
                    END DO 
                END DO 
            END DO 
            CALL t_stopf('remap_Q_ppm')
        END SUBROUTINE remap_q_ppm
        !=======================================================================================================!
        !THis compute grid-based coefficients from Collela & Woodward 1984.

        FUNCTION compute_ppm_grids(dx) RESULT ( rslt )
            USE control_mod, ONLY: vert_remap_q_alg
            IMPLICIT NONE
            REAL(KIND=real_kind), intent(in) :: dx(-1:nlev+2) !grid spacings
            REAL(KIND=real_kind) :: rslt(10,0:nlev+1) !grid spacings
            INTEGER :: j
            INTEGER :: indb, inde
            !Calculate grid-based coefficients for stage 1 of compute_ppm
            IF (vert_remap_q_alg == 2) THEN
                indb = 2
                inde = nlev-1
                ELSE
                indb = 0
                inde = nlev+1
            END IF 
            DO j = indb , inde
                rslt(1,j) = dx(j) / (dx(j-1) + dx(j) + dx(j+1))
                rslt(2,j) = (2.*dx(j-1) + dx(j)) / (dx(j+1) + dx(j))
                rslt(3,j) = (dx(j) + 2.*dx(j+1)) / (dx(j-1) + dx(j))
            END DO 
            !Caculate grid-based coefficients for stage 2 of compute_ppm
            IF (vert_remap_q_alg == 2) THEN
                indb = 2
                inde = nlev-2
                ELSE
                indb = 0
                inde = nlev
            END IF 
            DO j = indb , inde
                rslt(4,j) = dx(j) / (dx(j) + dx(j+1))
                rslt(5,j) = 1. / sum(dx(j-1:j+2))
                rslt(6,j) = (2. * dx(j+1) * dx(j)) / (dx(j) + dx(j+1 ))
                rslt(7,j) = (dx(j-1) + dx(j  )) / (2. * dx(j  ) + dx(j+1))
                rslt(8,j) = (dx(j+2) + dx(j+1)) / (2. * dx(j+1) + dx(j  ))
                rslt(9,j) = dx(j  ) * (dx(j-1) + dx(j  )) / (2.*dx(j  ) +    dx(j+1))
                rslt(10,j) = dx(j+1) * (dx(j+1) + dx(j+2)) / (dx(j  ) + 2.*dx(j+1))
            END DO 
        END FUNCTION compute_ppm_grids
        !=======================================================================================================!
        !This computes a limited parabolic interpolant using a net 5-cell stencil, but the stages of computation are broken up 
        ! into 3 stages

        FUNCTION compute_ppm(a, dx) RESULT ( coefs )
            USE control_mod, ONLY: vert_remap_q_alg
            IMPLICIT NONE
            REAL(KIND=real_kind), intent(in) :: a    (-1:nlev+2) !Cell-mean values
            REAL(KIND=real_kind), intent(in) :: dx   (10,  0:nlev+1) !grid spacings
            REAL(KIND=real_kind) :: coefs(0:2,   nlev) !PPM coefficients (for parabola)
            REAL(KIND=real_kind) :: ai (0:nlev) !fourth-order accurate, then limited interface values
            REAL(KIND=real_kind) :: dma(0:nlev+1) !An expression from Collela's '84 publication
            REAL(KIND=real_kind) :: da !Ditto
            ! Hold expressions based on the grid (which are cumbersome).
            REAL(KIND=real_kind) :: dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10
            REAL(KIND=real_kind) :: al, ar !Left and right interface values for cell-local limiting
            INTEGER :: j
            INTEGER :: indb, inde
            ! Stage 1: Compute dma for each cell, allowing a 1-cell ghost stencil below and above the domain
            IF (vert_remap_q_alg == 2) THEN
                indb = 2
                inde = nlev-1
                ELSE
                indb = 0
                inde = nlev+1
            END IF 
            DO j = indb , inde
                da = dx(1,j) * (dx(2,j) * ( a(j+1) - a(j) ) + dx(3,j) * ( a(j) - a(j-1) ))
                dma(j) = minval((/ abs(da) , 2. * abs( a(j) - a(j-1) ) , 2. * abs( a(j+1) - a(j) ) /)) * sign(1.d0,da)
                IF (( a(j+1) - a(j) ) * ( a(j) - a(j-1) ) <= 0.) dma(j) = 0.
            END DO 
            ! Stage 2: Compute ai for each cell interface in the physical domain (dimension nlev+1)
            IF (vert_remap_q_alg == 2) THEN
                indb = 2
                inde = nlev-2
                ELSE
                indb = 0
                inde = nlev
            END IF 
            DO j = indb , inde
                ai(j) = a(j) + dx(4,j) * (a(j+1) - a(j)) + dx(5,j) * (dx(6,j) * ( dx(7,j) - dx(8,j) )          * ( a(j+1) - a(j) )&
                 - dx(9,j) * dma(j+1) + dx(10,j) * dma(j))
            END DO 
            ! Stage 3: Compute limited PPM interpolant over each cell in the physical domain
            ! (dimension nlev) using ai on either side and ao within the cell.
            IF (vert_remap_q_alg == 2) THEN
                indb = 3
                inde = nlev-2
                ELSE
                indb = 1
                inde = nlev
            END IF 
            DO j = indb , inde
                al = ai(j-1)
                ar = ai(j  )
                IF ((ar - a(j)) * (a(j) - al) <= 0.) THEN
                    al = a(j)
                    ar = a(j)
                END IF 
                IF ((ar - al) * (a(j) - (al + ar)/2.) >  (ar - al)**2/6.) al = 3.*a(j) - 2. * ar
                IF ((ar - al) * (a(j) - (al + ar)/2.) < -(ar - al)**2/6.) ar = 3.*a(j) - 2. * al
                !Computed these coefficients from the edge values and cell mean in Maple. Assumes normalized coordinates: xi=(
                ! x-x0)/dx
                coefs(0,j) = 1.5 * a(j) - (al + ar) / 4.
                coefs(1,j) = ar - al
                coefs(2,j) = -6. * a(j) + 3. * (al + ar)
            END DO 
            !If we're not using a mirrored boundary condition, then make the two cells bordering the top and bottom
            !material boundaries piecewise constant. Zeroing out the first and second moments, and setting the zeroth
            !moment to the cell mean is sufficient to maintain conservation.
            IF (vert_remap_q_alg == 2) THEN
                coefs(0,1:2) = a(1:2)
                coefs(1:2,1:2) = 0.
                coefs(0,nlev-1:nlev) = a(nlev-1:nlev)
                coefs(1:2,nlev-1:nlev) = 0.d0
            END IF 
        END FUNCTION compute_ppm
        !=======================================================================================================!
        !Simple function computes the definite integral of a parabola in normalized coordinates, xi=(x-x0)/dx,
        !given two bounds. Make sure this gets inlined during compilation.

        FUNCTION integrate_parabola(a, x1, x2) RESULT ( mass )
            IMPLICIT NONE
            REAL(KIND=real_kind), intent(in) :: a(0:2) !Coefficients of the parabola
            REAL(KIND=real_kind), intent(in) :: x1 !lower domain bound for integration
            REAL(KIND=real_kind), intent(in) :: x2 !upper domain bound for integration
            REAL(KIND=real_kind) :: mass
            mass = a(0) * (x2 - x1) + a(1) * (x2 ** 2 - x1 ** 2) / 0.2d1 + a(2) * (x2 ** 3 - x1 ** 3) / 0.3d1
        END FUNCTION integrate_parabola
        !=============================================================================================!
    END MODULE vertremap_mod





























