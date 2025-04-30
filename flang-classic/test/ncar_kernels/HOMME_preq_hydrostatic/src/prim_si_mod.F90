
! KGEN-generated Fortran source file
!
! Filename    : prim_si_mod.F90
! Generated at: 2015-04-12 19:37:50
! KGEN version: 0.4.9



    MODULE prim_si_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        IMPLICIT NONE
        PRIVATE
        PUBLIC preq_hydrostatic
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

        SUBROUTINE preq_hydrostatic(phi, phis, t_v, p, dp)
            USE kinds, ONLY: real_kind
            USE dimensions_mod, ONLY: np
            USE dimensions_mod, ONLY: nlev
            USE physical_constants, ONLY: rgas
            !    use hybvcoord_mod, only : hvcoord_t
            IMPLICIT NONE
            !------------------------------Arguments---------------------------------------------------------------
            REAL(KIND=real_kind), intent(out) :: phi(np,np,nlev)
            REAL(KIND=real_kind), intent(in) :: phis(np,np)
            REAL(KIND=real_kind), intent(in) :: t_v(np,np,nlev)
            REAL(KIND=real_kind), intent(in) :: p(np,np,nlev)
            REAL(KIND=real_kind), intent(in) :: dp(np,np,nlev)
            !   type (hvcoord_t),     intent(in) :: hvcoord
            !------------------------------------------------------------------------------------------------------
            !---------------------------Local workspace-----------------------------
            INTEGER :: j
            INTEGER :: i
            INTEGER :: k ! longitude, level indices
            REAL(KIND=real_kind) :: hkk
            REAL(KIND=real_kind) :: hkl ! diagonal term of energy conversion matrix
            REAL(KIND=real_kind), dimension(np,np,nlev) :: phii ! Geopotential at interfaces
            !-----------------------------------------------------------------------
            DO j=1,np !   Loop inversion (AAM)
                DO i=1,np
                    hkk = dp(i,j,nlev)*0.5d0/p(i,j,nlev)
                    hkl = 2*hkk
                    phii(i,j,nlev) = rgas*t_v(i,j,nlev)*hkl
                    phi(i,j,nlev) = phis(i,j) + rgas*t_v(i,j,nlev)*hkk
                END DO 
                DO k=nlev-1,2,-1
                    DO i=1,np
                        ! hkk = dp*ckk
                        hkk = dp(i,j,k)*0.5d0/p(i,j,k)
                        hkl = 2*hkk
                        phii(i,j,k) = phii(i,j,k+1) + rgas*t_v(i,j,k)*hkl
                        phi(i,j,k) = phis(i,j) + phii(i,j,k+1) + rgas*t_v(i,j,k)*hkk
                    END DO 
                END DO 
                DO i=1,np
                    ! hkk = dp*ckk
                    hkk = 0.5d0*dp(i,j,1)/p(i,j,1)
                    phi(i,j,1) = phis(i,j) + phii(i,j,2) + rgas*t_v(i,j,1)*hkk
                END DO 
            END DO 
        END SUBROUTINE preq_hydrostatic
        !
        !  The hydrostatic routine from 1 physics.
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
