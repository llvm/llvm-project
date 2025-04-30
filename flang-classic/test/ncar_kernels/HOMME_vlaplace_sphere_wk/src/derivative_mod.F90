
! KGEN-generated Fortran source file
!
! Filename    : derivative_mod.F90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9



    MODULE derivative_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE kinds, ONLY: real_kind
        USE dimensions_mod, ONLY: np
        USE dimensions_mod, ONLY: nc
        USE dimensions_mod, ONLY: nep
        USE parallel_mod, ONLY: abortmp
        ! needed for spherical differential operators:
        USE physical_constants, ONLY: rrearth
        USE element_mod, ONLY: element_t
        USE control_mod, ONLY: hypervis_scaling
        USE control_mod, ONLY: hypervis_power
        IMPLICIT NONE
        PRIVATE
        TYPE, public :: derivative_t
            REAL(KIND=real_kind) :: dvv(np,np)
            REAL(KIND=real_kind) :: dvv_diag(np,np)
            REAL(KIND=real_kind) :: dvv_twt(np,np)
            REAL(KIND=real_kind) :: mvv_twt(np,np) ! diagonal matrix of GLL weights
            REAL(KIND=real_kind) :: mfvm(np,nc+1)
            REAL(KIND=real_kind) :: cfvm(np,nc)
            REAL(KIND=real_kind) :: sfvm(np,nep)
            REAL(KIND=real_kind) :: legdg(np,np)
        END TYPE derivative_t
        ! ======================================
        ! Public Interfaces
        ! ======================================



        ! these routines compute spherical differential operators as opposed to
        ! the gnomonic coordinate operators above.  Vectors (input or output)
        ! are always expressed in lat-lon coordinates
        !
        ! note that weak derivatives (integrated by parts form) can be defined using
        ! contra or co-variant test functions, so
        !
        PUBLIC gradient_sphere
        PUBLIC gradient_sphere_wk_testcov
        ! only used for debugging
        PUBLIC vorticity_sphere
        PUBLIC divergence_sphere
        PUBLIC curl_sphere_wk_testcov
        !  public  :: curl_sphere_wk_testcontra  ! not coded
        PUBLIC divergence_sphere_wk
        PUBLIC laplace_sphere_wk
        PUBLIC vlaplace_sphere_wk

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_derivative_t
        END INTERFACE kgen_read

        PUBLIC kgen_verify
        INTERFACE kgen_verify
            MODULE PROCEDURE kgen_verify_derivative_t
        END INTERFACE kgen_verify

        CONTAINS

        ! write subroutines
        ! No module extern variables
        SUBROUTINE kgen_read_derivative_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(derivative_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%dvv
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dvv **", var%dvv
            END IF
            READ(UNIT=kgen_unit) var%dvv_diag
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dvv_diag **", var%dvv_diag
            END IF
            READ(UNIT=kgen_unit) var%dvv_twt
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dvv_twt **", var%dvv_twt
            END IF
            READ(UNIT=kgen_unit) var%mvv_twt
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%mvv_twt **", var%mvv_twt
            END IF
            READ(UNIT=kgen_unit) var%mfvm
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%mfvm **", var%mfvm
            END IF
            READ(UNIT=kgen_unit) var%cfvm
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%cfvm **", var%cfvm
            END IF
            READ(UNIT=kgen_unit) var%sfvm
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%sfvm **", var%sfvm
            END IF
            READ(UNIT=kgen_unit) var%legdg
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%legdg **", var%legdg
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_derivative_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(derivative_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_real_real_kind_dim2("dvv", dtype_check_status, var%dvv, ref_var%dvv)
            CALL kgen_verify_real_real_kind_dim2("dvv_diag", dtype_check_status, var%dvv_diag, ref_var%dvv_diag)
            CALL kgen_verify_real_real_kind_dim2("dvv_twt", dtype_check_status, var%dvv_twt, ref_var%dvv_twt)
            CALL kgen_verify_real_real_kind_dim2("mvv_twt", dtype_check_status, var%mvv_twt, ref_var%mvv_twt)
            CALL kgen_verify_real_real_kind_dim2("mfvm", dtype_check_status, var%mfvm, ref_var%mfvm)
            CALL kgen_verify_real_real_kind_dim2("cfvm", dtype_check_status, var%cfvm, ref_var%cfvm)
            CALL kgen_verify_real_real_kind_dim2("sfvm", dtype_check_status, var%sfvm, ref_var%sfvm)
            CALL kgen_verify_real_real_kind_dim2("legdg", dtype_check_status, var%legdg, ref_var%legdg)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
            SUBROUTINE kgen_verify_real_real_kind_dim2( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in), DIMENSION(:,:) :: var, ref_var
                real(KIND=real_kind) :: nrmsdiff, rmsdiff
                real(KIND=real_kind), allocatable, DIMENSION(:,:) :: temp, temp2
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
            END SUBROUTINE kgen_verify_real_real_kind_dim2

        ! ==========================================
        ! derivinit:
        !
        ! Initialize the matrices for taking
        ! derivatives and interpolating
        ! ==========================================


        ! =======================================
        ! dmatinit:
        !
        ! Compute rectangular v->p
        ! derivative matrix (dmat)
        ! =======================================

        ! =======================================
        ! dpvinit:
        !
        ! Compute rectangular p->v
        ! derivative matrix (dmat)
        ! for strong gradients
        ! =======================================

        ! =======================================
        ! v2pinit:
        ! Compute interpolation matrix from gll(1:n1) -> gs(1:n2)
        ! =======================================

        ! =======================================
        ! dvvinit:
        !
        ! Compute rectangular v->v
        ! derivative matrix (dvv)
        ! =======================================

        !  ================================================
        !  divergence_stag:
        !
        !  Compute divergence (maps v grid -> p grid)
        !  ================================================

        !  ================================================
        !  divergence_nonstag:
        !
        !  Compute divergence (maps v->v)
        !  ================================================

        !  ================================================
        !  gradient_wk_stag:
        !
        !  Compute the weak form gradient:
        !  maps scalar field on the pressure grid to the
        !  velocity grid
        !  ================================================

        !  ================================================
        !  gradient_wk_nonstag:
        !
        !  Compute the weak form gradient:
        !  maps scalar field on the Gauss-Lobatto grid to the
        !  weak gradient on the Gauss-Lobbatto grid
        !  ================================================

        !  ================================================
        !  gradient_str_stag:
        !
        !  Compute the *strong* form gradient:
        !  maps scalar field on the pressure grid to the
        !  velocity grid
        !  ================================================

        !  ================================================
        !  gradient_str_nonstag:
        !
        !  Compute the *strong* gradient on the velocity grid
        !  of a scalar field on the velocity grid
        !  ================================================

        !  ================================================
        !  vorticity:
        !
        !  Compute the vorticity of the velocity field on the
        !  velocity grid
        !  ================================================

        !  ================================================
        !  interpolate_gll2fvm_points:
        !
        !  shape funtion interpolation from data on GLL grid to cellcenters on physics grid
        !  Author: Christoph Erath
        !  ================================================

        !  ================================================
        !  interpolate_gll2spelt_points:
        !
        !  shape function interpolation from data on GLL grid the spelt grid
        !  Author: Christoph Erath
        !  ================================================

        !  ================================================
        !  interpolate_gll2fvm_corners:
        !
        !  shape funtion interpolation from data on GLL grid to physics grid
        !
        !  ================================================

        !  ================================================
        !  remap_phys2gll:
        !
        !  interpolate to an equally spaced (in reference element coordinate system)
        !  "physics" grid to the GLL grid
        !
        !  1st order, monotone, conservative
        !  MT initial version 2013
        !  ================================================

        !----------------------------------------------------------------

        FUNCTION gradient_sphere(s, deriv, dinv) RESULT ( ds )
            !
            !   input s:  scalar
            !   output  ds: spherical gradient of s, lat-lon coordinates
            !
            TYPE(derivative_t), intent(in) :: deriv
            REAL(KIND=real_kind), intent(in), dimension(2,2,np,np) :: dinv
            REAL(KIND=real_kind), intent(in) :: s(np,np)
            REAL(KIND=real_kind) :: ds(np,np,2)
            INTEGER :: i
            INTEGER :: j
            INTEGER :: l
            REAL(KIND=real_kind) :: dsdx00
            REAL(KIND=real_kind) :: dsdy00
            REAL(KIND=real_kind) :: v1(np,np)
            REAL(KIND=real_kind) :: v2(np,np)
            DO j=1,np
                DO l=1,np
                    dsdx00 = 0.0d0
                    dsdy00 = 0.0d0
                    DO i=1,np
                        dsdx00 = dsdx00 + deriv%dvv(i,l)*s(i,j)
                        dsdy00 = dsdy00 + deriv%dvv(i,l)*s(j  ,i)
                    END DO 
                    v1(l  ,j) = dsdx00*rrearth
                    v2(j  ,l) = dsdy00*rrearth
                END DO 
            END DO 
            ! convert covarient to latlon
            DO j=1,np
                DO i=1,np
                    ds(i,j,1) = dinv(1,1,i,j)*v1(i,j) + dinv(2,1,i,j)*v2(i,j)
                    ds(i,j,2) = dinv(1,2,i,j)*v1(i,j) + dinv(2,2,i,j)*v2(i,j)
                END DO 
            END DO 
        END FUNCTION gradient_sphere

        FUNCTION curl_sphere_wk_testcov(s, deriv, elem) RESULT ( ds )
            !
            !   integrated-by-parts gradient, w.r.t. COVARIANT test functions
            !   input s:  scalar  (assumed to be s*khat)
            !   output  ds: weak curl, lat/lon coordinates
            !
            ! starting with:
            !   PHIcov1 = (PHI,0)  covariant vector
            !   PHIcov2 = (0,PHI)  covariant vector
            !
            !   ds1 = integral[ PHIcov1 dot curl(s*khat) ]
            !   ds2 = integral[ PHIcov2 dot curl(s*khat) ]
            ! integrate by parts:
            !   ds1 = integral[ vor(PHIcov1) * s ]
            !   ds2 = integral[ vor(PHIcov1) * s ]
            !
            !     PHIcov1 = (PHI^mn,0)
            !     PHIcov2 = (0,PHI^mn)
            !  vorticity() acts on covariant vectors:
            !   ds1 = sum wij g  s_ij 1/g (  (PHIcov1_2)_x  - (PHIcov1_1)_y )
            !       = -sum wij s_ij  d/dy (PHI^mn )
            ! for d/dy component, only sum over i=m
            !       = -sum  w_mj s_mj   d( PHI^n)(j)
            !           j
            !
            !   ds2 = sum wij g  s_ij 1/g (  (PHIcov2_2)_x  - (PHIcov2_1)_y )
            !       = +sum wij s_ij  d/dx (PHI^mn )
            ! for d/dx component, only sum over j=n
            !       = +sum  w_in s_in  d( PHI^m)(i)
            !           i
            !
            TYPE(derivative_t), intent(in) :: deriv
            TYPE(element_t), intent(in) :: elem
            REAL(KIND=real_kind), intent(in) :: s(np,np)
            REAL(KIND=real_kind) :: ds(np,np,2)
            INTEGER :: n
            INTEGER :: m
            INTEGER :: j
            INTEGER :: i
            REAL(KIND=real_kind) :: dscontra(np,np,2)
            dscontra = 0
            DO n=1,np
                DO m=1,np
                    DO j=1,np
                        ! phi(n)_y  sum over second index, 1st index fixed at m
                        dscontra(m,n,1) = dscontra(m,n,1)-(elem%mp(m,j)*s(m,j)*deriv%dvv(n,j))*rrearth
                        ! phi(m)_x  sum over first index, second index fixed at n
                        dscontra(m,n,2) = dscontra(m,n,2)+(elem%mp(j,n)*s(j,n)*deriv%dvv(m,j))*rrearth
                    END DO 
                END DO 
            END DO 
            ! convert contra -> latlon
            DO j=1,np
                DO i=1,np
                    ds(i,j,1) = (elem%d(1,1,i,j)*dscontra(i,j,1) + elem%d(1,2,i,j)*dscontra(i,j,2))
                    ds(i,j,2) = (elem%d(2,1,i,j)*dscontra(i,j,1) + elem%d(2,2,i,j)*dscontra(i,j,2))
                END DO 
            END DO 
        END FUNCTION curl_sphere_wk_testcov

        FUNCTION gradient_sphere_wk_testcov(s, deriv, elem) RESULT ( ds )
            !
            !   integrated-by-parts gradient, w.r.t. COVARIANT test functions
            !   input s:  scalar
            !   output  ds: weak gradient, lat/lon coordinates
            !   ds = - integral[ div(PHIcov) s ]
            !
            !     PHIcov1 = (PHI^mn,0)
            !     PHIcov2 = (0,PHI^mn)
            !   div() acts on contra components, so convert test function to contra:
            !     PHIcontra1 =  metinv PHIcov1  = (a^mn,b^mn)*PHI^mn
            !                                     a = metinv(1,1)  b=metinv(2,1)
            !
            !   ds1 = sum wij g  s_ij 1/g ( g a PHI^mn)_x  + ( g b PHI^mn)_y )
            !       = sum  wij s_ij  ag(m,n)  d/dx( PHI^mn ) + bg(m,n) d/dy( PHI^mn)
            !          i,j
            ! for d/dx component, only sum over j=n
            !       = sum  w_in s_in  ag(m,n)  d( PHI^m)(i)
            !          i
            ! for d/dy component, only sum over i=m
            !       = sum  w_mj s_mj  bg(m,n)  d( PHI^n)(j)
            !          j
            !
            !
            ! This formula is identical to gradient_sphere_wk_testcontra, except that
            !    g(m,n) is replaced by a(m,n)*g(m,n)
            !  and we have two terms for each componet of ds
            !
            !
            TYPE(derivative_t), intent(in) :: deriv
            TYPE(element_t), intent(in) :: elem
            REAL(KIND=real_kind), intent(in) :: s(np,np)
            REAL(KIND=real_kind) :: ds(np,np,2)
            INTEGER :: n
            INTEGER :: m
            INTEGER :: j
            INTEGER :: i
            REAL(KIND=real_kind) :: dscontra(np,np,2)
            dscontra = 0
            DO n=1,np
                DO m=1,np
                    DO j=1,np
                        dscontra(m,n,1) = dscontra(m,n,1)-((elem%mp(j,n)*elem%metinv(1,1,m,n)*elem%metdet(m,n)*s(j,n)*deriv%dvv(m,&
                        j) ) +                  (elem%mp(m,j)*elem%metinv(2,1,m,n)*elem%metdet(m,n)*s(m,j)*deriv%dvv(n,j) )) *rrearth
                        dscontra(m,n,2) = dscontra(m,n,2)-((elem%mp(j,n)*elem%metinv(1,2,m,n)*elem%metdet(m,n)*s(j,n)*deriv%dvv(m,&
                        j) ) +                  (elem%mp(m,j)*elem%metinv(2,2,m,n)*elem%metdet(m,n)*s(m,j)*deriv%dvv(n,j) )) *rrearth
                    END DO 
                END DO 
            END DO 
            ! convert contra -> latlon
            DO j=1,np
                DO i=1,np
                    ds(i,j,1) = (elem%d(1,1,i,j)*dscontra(i,j,1) + elem%d(1,2,i,j)*dscontra(i,j,2))
                    ds(i,j,2) = (elem%d(2,1,i,j)*dscontra(i,j,1) + elem%d(2,2,i,j)*dscontra(i,j,2))
                END DO 
            END DO 
        END FUNCTION gradient_sphere_wk_testcov



        !--------------------------------------------------------------------------

        FUNCTION divergence_sphere_wk(v, deriv, elem) RESULT ( div )
            !
            !   input:  v = velocity in lat-lon coordinates
            !   ouput:  div(v)  spherical divergence of v, integrated by parts
            !
            !   Computes  -< grad(psi) dot v >
            !   (the integrated by parts version of < psi div(v) > )
            !
            !   note: after DSS, divergence_sphere() and divergence_sphere_wk()
            !   are identical to roundoff, as theory predicts.
            !
            REAL(KIND=real_kind), intent(in) :: v(np,np,2) ! in lat-lon coordinates
            TYPE(derivative_t), intent(in) :: deriv
            TYPE(element_t), intent(in) :: elem
            REAL(KIND=real_kind) :: div(np,np)
            ! Local
            INTEGER :: j
            INTEGER :: i
            INTEGER :: n
            INTEGER :: m
            REAL(KIND=real_kind) :: vtemp(np,np,2)
            ! latlon- > contra
            DO j=1,np
                DO i=1,np
                    vtemp(i,j,1) = (elem%dinv(1,1,i,j)*v(i,j,1) + elem%dinv(1,2,i,j)*v(i,j,2))
                    vtemp(i,j,2) = (elem%dinv(2,1,i,j)*v(i,j,1) + elem%dinv(2,2,i,j)*v(i,j,2))
                END DO 
            END DO 
            DO n=1,np
                DO m=1,np
                    div(m,n) = 0
                    DO j=1,np
                        div(m,n) = div(m,n)-(elem%spheremp(j,n)*vtemp(j,n,1)*deriv%dvv(m,j)                               &
                        +elem%spheremp(m,j)*vtemp(m,j,2)*deriv%dvv(n,j))                               * rrearth
                    END DO 
                END DO 
            END DO 
        END FUNCTION divergence_sphere_wk



        FUNCTION vorticity_sphere(v, deriv, elem) RESULT ( vort )
            !
            !   input:  v = velocity in lat-lon coordinates
            !   ouput:  spherical vorticity of v
            !
            TYPE(derivative_t), intent(in) :: deriv
            TYPE(element_t), intent(in) :: elem
            REAL(KIND=real_kind), intent(in) :: v(np,np,2)
            REAL(KIND=real_kind) :: vort(np,np)
            INTEGER :: i
            INTEGER :: j
            INTEGER :: l
            REAL(KIND=real_kind) :: dvdx00
            REAL(KIND=real_kind) :: dudy00
            REAL(KIND=real_kind) :: vco(np,np,2)
            REAL(KIND=real_kind) :: vtemp(np,np)
            ! convert to covariant form
            DO j=1,np
                DO i=1,np
                    vco(i,j,1) = (elem%d(1,1,i,j)*v(i,j,1) + elem%d(2,1,i,j)*v(i,j,2))
                    vco(i,j,2) = (elem%d(1,2,i,j)*v(i,j,1) + elem%d(2,2,i,j)*v(i,j,2))
                END DO 
            END DO 
            DO j=1,np
                DO l=1,np
                    dudy00 = 0.0d0
                    dvdx00 = 0.0d0
                    DO i=1,np
                        dvdx00 = dvdx00 + deriv%dvv(i,l)*vco(i,j  ,2)
                        dudy00 = dudy00 + deriv%dvv(i,l)*vco(j  ,i,1)
                    END DO 
                    vort(l  ,j) = dvdx00
                    vtemp(j  ,l) = dudy00
                END DO 
            END DO 
            DO j=1,np
                DO i=1,np
                    vort(i,j) = (vort(i,j)-vtemp(i,j))*(elem%rmetdet(i,j)*rrearth)
                END DO 
            END DO 
        END FUNCTION vorticity_sphere


        FUNCTION divergence_sphere(v, deriv, elem) RESULT ( div )
            !
            !   input:  v = velocity in lat-lon coordinates
            !   ouput:  div(v)  spherical divergence of v
            !
            REAL(KIND=real_kind), intent(in) :: v(np,np,2) ! in lat-lon coordinates
            TYPE(derivative_t), intent(in) :: deriv
            TYPE(element_t), intent(in) :: elem
            REAL(KIND=real_kind) :: div(np,np)
            ! Local
            INTEGER :: i
            INTEGER :: j
            INTEGER :: l
            REAL(KIND=real_kind) :: dudx00
            REAL(KIND=real_kind) :: dvdy00
            REAL(KIND=real_kind) :: gv(np,np,2)
            REAL(KIND=real_kind) :: vvtemp(np,np)
            ! convert to contra variant form and multiply by g
            DO j=1,np
                DO i=1,np
                    gv(i,j,1) = elem%metdet(i,j)*(elem%dinv(1,1,i,j)*v(i,j,1) + elem%dinv(1,2,i,j)*v(i,j,2))
                    gv(i,j,2) = elem%metdet(i,j)*(elem%dinv(2,1,i,j)*v(i,j,1) + elem%dinv(2,2,i,j)*v(i,j,2))
                END DO 
            END DO 
            ! compute d/dx and d/dy
            DO j=1,np
                DO l=1,np
                    dudx00 = 0.0d0
                    dvdy00 = 0.0d0
                    DO i=1,np
                        dudx00 = dudx00 + deriv%dvv(i,l)*gv(i,j  ,1)
                        dvdy00 = dvdy00 + deriv%dvv(i,l)*gv(j  ,i,2)
                    END DO 
                    div(l  ,j) = dudx00
                    vvtemp(j  ,l) = dvdy00
                END DO 
            END DO 
            DO j=1,np
                DO i=1,np
                    div(i,j) = (div(i,j)+vvtemp(i,j))*(elem%rmetdet(i,j)*rrearth)
                END DO 
            END DO 
        END FUNCTION divergence_sphere

        FUNCTION laplace_sphere_wk(s, deriv, elem, var_coef) RESULT ( laplace )
            !
            !   input:  s = scalar
            !   ouput:  -< grad(PHI), grad(s) >   = weak divergence of grad(s)
            !     note: for this form of the operator, grad(s) does not need to be made C0
            !
            REAL(KIND=real_kind), intent(in) :: s(np,np)
            LOGICAL, intent(in) :: var_coef
            TYPE(derivative_t), intent(in) :: deriv
            TYPE(element_t), intent(in) :: elem
            REAL(KIND=real_kind) :: laplace(np,np)
            INTEGER :: j
            INTEGER :: i
            ! Local
            REAL(KIND=real_kind) :: grads(np,np,2)
            REAL(KIND=real_kind) :: oldgrads(np,np,2)
            grads = gradient_sphere(s,deriv,elem%dinv)
            IF (var_coef) THEN
                IF (hypervis_power/=0) THEN
                    ! scalar viscosity with variable coefficient
                    grads(:,:,1) = grads(:,:,1)*elem%variable_hyperviscosity(:,:)
                    grads(:,:,2) = grads(:,:,2)*elem%variable_hyperviscosity(:,:)
                    ELSE IF (hypervis_scaling /=0) THEN
                    ! tensor hv, (3)
                    oldgrads = grads
                    DO j=1,np
                        DO i=1,np
                            grads(i,j,1) = sum(oldgrads(i,j,:)*elem%tensorvisc(1,:,i,j))
                            grads(i,j,2) = sum(oldgrads(i,j,:)*elem%tensorvisc(2,:,i,j))
                        END DO 
                    END DO 
                    ELSE
                    ! do nothing: constant coefficient viscsoity
                END IF 
            END IF 
            ! note: divergnece_sphere and divergence_sphere_wk are identical *after* bndry_exchange
            ! if input is C_0.  Here input is not C_0, so we should use divergence_sphere_wk().
            laplace = divergence_sphere_wk(grads,deriv,elem)
        END FUNCTION laplace_sphere_wk

        FUNCTION vlaplace_sphere_wk(v, deriv, elem, var_coef, nu_ratio) RESULT ( laplace )
            !
            !   input:  v = vector in lat-lon coordinates
            !   ouput:  weak laplacian of v, in lat-lon coordinates
            !
            !   logic:
            !      tensorHV:     requires cartesian
            !      nu_div/=nu:   requires contra formulatino
            !
            !   One combination NOT supported:  tensorHV and nu_div/=nu then abort
            !
            REAL(KIND=real_kind), intent(in) :: v(np,np,2)
            LOGICAL, intent(in) :: var_coef
            TYPE(derivative_t), intent(in) :: deriv
            TYPE(element_t), intent(in) :: elem
            REAL(KIND=real_kind), optional :: nu_ratio
            REAL(KIND=real_kind) :: laplace(np,np,2)
            IF (hypervis_scaling/=0 .and. var_coef) THEN
                ! tensorHV is turned on - requires cartesian formulation
                IF (present(nu_ratio)) THEN
                    IF (nu_ratio /= 1) THEN
                        CALL abortmp('ERROR: tensorHV can not be used with nu_div/=nu')
                    END IF 
                END IF 
                laplace = vlaplace_sphere_wk_cartesian(v,deriv,elem,var_coef)
                ELSE
                ! all other cases, use contra formulation:
                laplace = vlaplace_sphere_wk_contra(v,deriv,elem,var_coef,nu_ratio)
            END IF 
        END FUNCTION vlaplace_sphere_wk

        FUNCTION vlaplace_sphere_wk_cartesian(v, deriv, elem, var_coef) RESULT ( laplace )
            !
            !   input:  v = vector in lat-lon coordinates
            !   ouput:  weak laplacian of v, in lat-lon coordinates
            REAL(KIND=real_kind), intent(in) :: v(np,np,2)
            LOGICAL :: var_coef
            TYPE(derivative_t), intent(in) :: deriv
            TYPE(element_t), intent(in) :: elem
            REAL(KIND=real_kind) :: laplace(np,np,2)
            ! Local
            INTEGER :: component
            REAL(KIND=real_kind) :: dum_cart(np,np,3)
            ! latlon -> cartesian
            DO component=1,3
                dum_cart(:,:,component) = sum(elem%vec_sphere2cart(:,:,component,:)*v(:,:,:) ,3)
            END DO 
            ! Do laplace on cartesian comps
            DO component=1,3
                dum_cart(:,:,component) = laplace_sphere_wk(dum_cart(:,:,component),deriv,elem,var_coef)
            END DO 
            ! cartesian -> latlon
            DO component=1,2
                ! vec_sphere2cart is its own pseudoinverse.
                laplace(:,:,component) = sum(dum_cart(:,:,:)*elem%vec_sphere2cart(:,:,:,component) ,3)
            END DO 
        END FUNCTION vlaplace_sphere_wk_cartesian

        FUNCTION vlaplace_sphere_wk_contra(v, deriv, elem, var_coef, nu_ratio) RESULT ( laplace )
            !
            !   input:  v = vector in lat-lon coordinates
            !   ouput:  weak laplacian of v, in lat-lon coordinates
            !
            !   du/dt = laplace(u) = grad(div) - curl(vor)
            !   < PHI du/dt > = < PHI laplace(u) >        PHI = covariant, u = contravariant
            !                 = < PHI grad(div) >  - < PHI curl(vor) >
            !                 = grad_wk(div) - curl_wk(vor)
            !
            REAL(KIND=real_kind), intent(in) :: v(np,np,2)
            LOGICAL, intent(in) :: var_coef
            TYPE(derivative_t), intent(in) :: deriv
            TYPE(element_t), intent(in) :: elem
            REAL(KIND=real_kind) :: laplace(np,np,2)
            REAL(KIND=real_kind), optional :: nu_ratio
            ! Local
            INTEGER :: n
            INTEGER :: m
            REAL(KIND=real_kind) :: div(np,np)
            REAL(KIND=real_kind) :: vor(np,np)
            div = divergence_sphere(v,deriv,elem)
            vor = vorticity_sphere(v,deriv,elem)
            IF (var_coef .and. hypervis_power/=0) THEN
                ! scalar viscosity with variable coefficient
                div = div*elem%variable_hyperviscosity(:,:)
                vor = vor*elem%variable_hyperviscosity(:,:)
            END IF 
            IF (present(nu_ratio)) div = nu_ratio*div
            laplace = gradient_sphere_wk_testcov(div,deriv,elem) -          curl_sphere_wk_testcov(vor,deriv,elem)
            DO n=1,np
                DO m=1,np
                    ! add in correction so we dont damp rigid rotation
                    laplace(m,n,1) = laplace(m,n,1) + 2*elem%spheremp(m,n)*v(m,n,1)*(rrearth**2)
                    laplace(m,n,2) = laplace(m,n,2) + 2*elem%spheremp(m,n)*v(m,n,2)*(rrearth**2)
                END DO 
            END DO 
        END FUNCTION vlaplace_sphere_wk_contra


        ! Given a field defined on the unit element, [-1,1]x[-1,1]
        ! sample values, sampled_val, and integration weights, metdet,
        ! at a number, np, of Gauss-Lobatto-Legendre points. Divide
        ! the square up into intervals by intervals sub-squares so that
        ! there are now intervals**2 sub-cells.  Integrate the
        ! function defined by sampled_val and metdet over each of these
        ! sub-cells and return the integrated values as an
        ! intervals by intervals matrix.
        !
        ! Efficiency is obtained by computing and caching the appropriate
        ! integration matrix the first time the function is called.

        ! Helper subroutine that will fill in a matrix needed to
        ! integrate a function defined on the GLL points of a unit
        ! square on sub-cells.  So np is the number of integration
        ! GLL points defined on the unit square (actually [-1,1]x[-1,1])
        ! and intervals is the number to cut it up into, say a 3 by 3
        ! set of uniform sub-cells.  This function will fill the
        ! subcell_integration matrix with the correct coefficients
        ! to integrate over each subcell.

    END MODULE derivative_mod
