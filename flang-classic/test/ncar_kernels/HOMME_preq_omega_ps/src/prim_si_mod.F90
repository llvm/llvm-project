
! KGEN-generated Fortran source file
!
! Filename    : prim_si_mod.F90
! Generated at: 2015-03-16 09:25:31
! KGEN version: 0.4.5



    MODULE prim_si_mod
        IMPLICIT NONE
        PRIVATE
        PUBLIC preq_omega_ps
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        ! ==========================================================
        ! Implicit system for semi-implicit primitive equations.
        ! ==========================================================

        !-----------------------------------------------------------------------
        ! preq_omegap:
        ! Purpose:
        ! Calculate (omega/p) needed for the Thermodynamics Equation
        !
        ! Method:
        ! Simplified version in CAM2 for clarity
        !
        ! Author: Modified by Rich Loft for use in HOMME.
        !
        !-----------------------------------------------------------------------

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        !
        !  compute omega/p using ps, modeled after CCM3 formulas
        !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

        SUBROUTINE preq_omega_ps(omega_p, hvcoord, p, vgrad_p, divdp)
            USE kinds, ONLY: real_kind
            USE dimensions_mod, ONLY: np
            USE dimensions_mod, ONLY: nlev
            USE hybvcoord_mod, ONLY: hvcoord_t
            IMPLICIT NONE
            !------------------------------Arguments---------------------------------------------------------------
            REAL(KIND=real_kind), intent(in) :: divdp(np,np,nlev) ! divergence
            REAL(KIND=real_kind), intent(in) :: vgrad_p(np,np,nlev) ! v.grad(p)
            REAL(KIND=real_kind), intent(in) :: p(np,np,nlev) ! layer thicknesses (pressure)
            TYPE(hvcoord_t), intent(in) :: hvcoord
            REAL(KIND=real_kind), intent(out) :: omega_p(np,np,nlev) ! vertical pressure velocity
            !------------------------------------------------------------------------------------------------------
            !---------------------------Local workspace-----------------------------
            INTEGER :: j
            INTEGER :: i
            INTEGER :: k ! longitude, level indices
            REAL(KIND=real_kind) :: term ! one half of basic term in omega/p summation
            REAL(KIND=real_kind) :: ckk
            REAL(KIND=real_kind) :: ckl ! diagonal term of energy conversion matrix
            REAL(KIND=real_kind) :: suml(np,np) ! partial sum over l = (1, k-1)
            !-----------------------------------------------------------------------
            DO j=1,np !   Loop inversion (AAM)
                DO i=1,np
                    ckk = 0.5d0/p(i,j,1)
                    term = divdp(i,j,1)
                    !             omega_p(i,j,1) = hvcoord%hybm(1)*vgrad_ps(i,j,1)/p(i,j,1)
                    omega_p(i,j,1) = vgrad_p(i,j,1)/p(i,j,1)
                    omega_p(i,j,1) = omega_p(i,j,1) - ckk*term
                    suml(i,j) = term
                END DO 
                DO k=2,nlev-1
                    DO i=1,np
                        ckk = 0.5d0/p(i,j,k)
                        ckl = 2*ckk
                        term = divdp(i,j,k)
                        !                omega_p(i,j,k) = hvcoord%hybm(k)*vgrad_ps(i,j,k)/p(i,j,k)
                        omega_p(i,j,k) = vgrad_p(i,j,k)/p(i,j,k)
                        omega_p(i,j,k) = omega_p(i,j,k) - ckl*suml(i,j) - ckk*term
                        suml(i,j) = suml(i,j) + term
                    END DO 
                END DO 
                DO i=1,np
                    ckk = 0.5d0/p(i,j,nlev)
                    ckl = 2*ckk
                    term = divdp(i,j,nlev)
                    !             omega_p(i,j,nlev) = hvcoord%hybm(nlev)*vgrad_ps(i,j,nlev)/p(i,j,nlev)
                    omega_p(i,j,nlev) = vgrad_p(i,j,nlev)/p(i,j,nlev)
                    omega_p(i,j,nlev) = omega_p(i,j,nlev) - ckl*suml(i,j) - ckk*term
                END DO 
            END DO 
        END SUBROUTINE preq_omega_ps
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        !
        !  compute omega/p using lnps
        !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        !
        !  CCM3 hydrostatic integral
        !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

        !
        !  The hydrostatic routine from CAM physics.
        !  (FV stuff removed)
        !  t,q input changed to take t_v
        !  removed gravit, so this routine returns PHI, not zm

        !-----------------------------------------------------------------------
        ! preq_pressure:
        !
        ! Purpose:
        ! Define the pressures of the interfaces and midpoints from the
        ! coordinate definitions and the surface pressure. Originally plevs0!
        !
        ! Method:
        !
        ! Author: B. Boville/ Adapted for HOMME by Rich Loft
        !
        !-----------------------------------------------------------------------
        !
        ! $Id: prim_si_mod.F90,v 2.10 2005/10/14 20:17:22 jedwards Exp $
        ! $Author: jedwards $
        !
        !-----------------------------------------------------------------------

    END MODULE prim_si_mod
