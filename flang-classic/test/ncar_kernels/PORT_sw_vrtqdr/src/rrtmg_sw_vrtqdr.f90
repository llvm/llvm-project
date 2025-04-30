
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_sw_vrtqdr.f90
! Generated at: 2015-07-31 21:01:00
! KGEN version: 0.4.13



    MODULE rrtmg_sw_vrtqdr
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
        !      use parkind, only: jpim, jprb
        !      use parrrsw, only: ngptsw
        IMPLICIT NONE
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        ! --------------------------------------------------------------------------

        SUBROUTINE vrtqdr_sw(ncol, klev, kw, pref, prefd, ptra, ptrad, pdbt, prdnd, prup, prupd, ptdbt, pfd, pfu)
            ! --------------------------------------------------------------------------
            ! Purpose: This routine performs the vertical quadrature integration
            !
            ! Interface:  *vrtqdr_sw* is called from *spcvrt_sw* and *spcvmc_sw*
            !
            ! Modifications.
            !
            ! Original: H. Barker
            ! Revision: Integrated with rrtmg_sw, J.-J. Morcrette, ECMWF, Oct 2002
            ! Revision: Reformatted for consistency with rrtmg_lw: MJIacono, AER, Jul 2006
            !
            !-----------------------------------------------------------------------
            ! ------- Declarations -------
            ! Input
            INTEGER, intent (in) :: ncol
            INTEGER, intent (in) :: klev ! number of model layers
            INTEGER, intent (in) :: kw(ncol) ! g-point index
            REAL(KIND=r8), intent(in) :: pref(:,:) ! direct beam reflectivity
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(in) :: prefd(:,:) ! diffuse beam reflectivity
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(in) :: ptra(:,:) ! direct beam transmissivity
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(in) :: ptrad(:,:) ! diffuse beam transmissivity
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(in) :: pdbt(:,:)
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(in) :: ptdbt(:,:)
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(inout) :: prdnd(:,:)
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(inout) :: prup(:,:)
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(inout) :: prupd(:,:)
            !   Dimensions: (nlayers+1)
            ! Output
            REAL(KIND=r8), intent(out) :: pfd(:,:,:) ! downwelling flux (W/m2)
            !   Dimensions: (nlayers+1,ngptsw)
            ! unadjusted for earth/sun distance or zenith angle
            REAL(KIND=r8), intent(out) :: pfu(:,:,:) ! upwelling flux (W/m2)
            !   Dimensions: (nlayers+1,ngptsw)
            ! unadjusted for earth/sun distance or zenith angle
            ! Local
            INTEGER :: jk
            INTEGER :: ikp
            INTEGER :: icol
            INTEGER :: ikx
            REAL(KIND=r8) :: zreflect
            REAL(KIND=r8) :: ztdn(klev+1)
            ! Definitions
            !
            ! pref(icol,jk)   direct reflectance
            ! prefd(icol,jk)  diffuse reflectance
            ! ptra(icol,jk)   direct transmittance
            ! ptrad(icol,jk)  diffuse transmittance
            !
            ! pdbt(icol,jk)   layer mean direct beam transmittance
            ! ptdbt(icol,jk)  total direct beam transmittance at levels
            !
            !-----------------------------------------------------------------------------
            ! Link lowest layer with surface
			do icol=1,ncol
      zreflect = 1._r8 / (1._r8 - prefd(icol,klev+1) * prefd(icol,klev))
      prup(icol,klev) = pref(icol,klev) + (ptrad(icol,klev) * &
                 ((ptra(icol,klev) - pdbt(icol,klev)) * prefd(icol,klev+1) + &
                   pdbt(icol,klev) * pref(icol,klev+1))) * zreflect
      prupd(icol,klev) = prefd(icol,klev) + ptrad(icol,klev) * ptrad(icol,klev) * &
                    prefd(icol,klev+1) * zreflect
                ! Pass from bottom to top
      do jk = 1,klev-1
         ikp = klev+1-jk                       
         ikx = ikp-1
         zreflect = 1._r8 / (1._r8 -prupd(icol,ikp) * prefd(icol,ikx))
         prup(icol,ikx) = pref(icol,ikx) + (ptrad(icol,ikx) * &
                   ((ptra(icol,ikx) - pdbt(icol,ikx)) * prupd(icol,ikp) + &
                     pdbt(icol,ikx) * prup(icol,ikp))) * zreflect
         prupd(icol,ikx) = prefd(icol,ikx) + ptrad(icol,ikx) * ptrad(icol,ikx) * &
                      prupd(icol,ikp) * zreflect
      enddo
                ! Upper boundary conditions
      ztdn(1) = 1._r8
      prdnd(icol,1) = 0._r8
      ztdn(2) = ptra(icol,1)
      prdnd(icol,2) = prefd(icol,1)
                ! Pass from top to bottom
      do jk = 2,klev
         ikp = jk+1
         zreflect = 1._r8 / (1._r8 - prefd(icol,jk) * prdnd(icol,jk))
         ztdn(ikp) = ptdbt(icol,jk) * ptra(icol,jk) + &
                    (ptrad(icol,jk) * ((ztdn(jk) - ptdbt(icol,jk)) + &
                     ptdbt(icol,jk) * pref(icol,jk) * prdnd(icol,jk))) * zreflect
         prdnd(icol,ikp) = prefd(icol,jk) + ptrad(icol,jk) * ptrad(icol,jk) * &
                      prdnd(icol,jk) * zreflect
      enddo
                ! Up and down-welling fluxes at levels
      do jk = 1,klev+1
         zreflect = 1._r8 / (1._r8 - prdnd(icol,jk) * prupd(icol,jk))
         pfu(icol,jk,kw(icol)) = (ptdbt(icol,jk) * prup(icol,jk) + &
                      (ztdn(jk) - ptdbt(icol,jk)) * prupd(icol,jk)) * zreflect
         pfd(icol,jk,kw(icol)) = ptdbt(icol,jk) + (ztdn(jk) - ptdbt(icol,jk)+ &
                      ptdbt(icol,jk) * prup(icol,jk) * prdnd(icol,jk)) * zreflect
      enddo
	end do
        END SUBROUTINE vrtqdr_sw
    END MODULE rrtmg_sw_vrtqdr
