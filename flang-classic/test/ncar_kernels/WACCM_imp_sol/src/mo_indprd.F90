
! KGEN-generated Fortran source file
!
! Filename    : mo_indprd.F90
! Generated at: 2015-05-13 11:02:23
! KGEN version: 0.4.10



    MODULE mo_indprd
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        PRIVATE
        PUBLIC indprd
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables

        SUBROUTINE indprd(class, prod, nprod, y, extfrc, rxt, ncol)
            USE chem_mods, ONLY: gas_pcnst
            USE chem_mods, ONLY: extcnt
            USE chem_mods, ONLY: rxntot
            USE ppgrid, ONLY: pver
            IMPLICIT NONE
            !--------------------------------------------------------------------
            ! ... dummy arguments
            !--------------------------------------------------------------------
            INTEGER, intent(in) :: class
            INTEGER, intent(in) :: ncol
            INTEGER, intent(in) :: nprod
            REAL(KIND=r8), intent(in) :: y(ncol,pver,gas_pcnst)
            REAL(KIND=r8), intent(in) :: rxt(ncol,pver,rxntot)
            REAL(KIND=r8), intent(in) :: extfrc(ncol,pver,extcnt)
            REAL(KIND=r8), intent(inout) :: prod(ncol,pver,nprod)
            !--------------------------------------------------------------------
            ! ... "independent" production for Explicit species
            !--------------------------------------------------------------------
            IF (class == 1) THEN
                prod(:,:,1) = .080_r8*rxt(:,:,314)*y(:,:,48)*y(:,:,1)
                prod(:,:,2) = rxt(:,:,187)*y(:,:,7)*y(:,:,5)
                prod(:,:,3) = 0._r8
                prod(:,:,4) = 0._r8
                prod(:,:,5) = 0._r8
                prod(:,:,6) = 0._r8
                prod(:,:,7) = 0._r8
                prod(:,:,8) = 0._r8
                prod(:,:,9) = 0._r8
                prod(:,:,10) = 0._r8
                prod(:,:,11) = 0._r8
                prod(:,:,12) = 0._r8
                prod(:,:,13) = 0._r8
                prod(:,:,14) = 0._r8
                prod(:,:,15) = 0._r8
                prod(:,:,16) = 0._r8
                prod(:,:,17) = 0._r8
                prod(:,:,18) = 0._r8
                prod(:,:,19) = 0._r8
                prod(:,:,20) = 0._r8
                prod(:,:,21) = (rxt(:,:,267)*y(:,:,17) +rxt(:,:,268)*y(:,:,17) +                  rxt(:,:,279)*y(:,:,99) +rxt(:,:,&
                294)*y(:,:,40) +                  .500_r8*rxt(:,:,307)*y(:,:,45) +.800_r8*rxt(:,:,308)*y(:,:,43) +                &
                  rxt(:,:,309)*y(:,:,44) +.500_r8*rxt(:,:,358)*y(:,:,63))*y(:,:,129)                   + (rxt(:,:,302)*y(:,:,6) &
                +.900_r8*rxt(:,:,305)*y(:,:,13) +                  2.000_r8*rxt(:,:,306)*y(:,:,133) +2.000_r8*rxt(:,:,354)*y(:,:,&
                141) +                  rxt(:,:,382)*y(:,:,145))*y(:,:,133) + (rxt(:,:,353)*y(:,:,13) +                  &
                2.000_r8*rxt(:,:,355)*y(:,:,141))*y(:,:,141) +rxt(:,:,63)*y(:,:,45)                   +.400_r8*rxt(:,:,64)*y(:,:,&
                47)
                prod(:,:,22) = 0._r8
                prod(:,:,23) = 0._r8
                !--------------------------------------------------------------------
                ! ... "independent" production for Implicit species
                !--------------------------------------------------------------------
                ELSE IF (class == 4) THEN
                prod(:,:,123) = 0._r8
                prod(:,:,121) = (rxt(:,:,58) +rxt(:,:,117))*y(:,:,97) +.180_r8*rxt(:,:,60)                  *y(:,:,12)
                prod(:,:,122) = rxt(:,:,5)*y(:,:,4)
                prod(:,:,120) = 0._r8
                prod(:,:,28) = 0._r8
                prod(:,:,27) = 0._r8
                prod(:,:,108) = 1.440_r8*rxt(:,:,60)*y(:,:,12)
                prod(:,:,103) = (rxt(:,:,58) +rxt(:,:,117))*y(:,:,97) +.380_r8*rxt(:,:,60)                  *y(:,:,12) + extfrc(:,&
                :,3)
                prod(:,:,92) = (rxt(:,:,101) +.800_r8*rxt(:,:,104) +rxt(:,:,113) +                  .800_r8*rxt(:,:,116)) + &
                extfrc(:,:,16)
                prod(:,:,129) = + extfrc(:,:,1)
                prod(:,:,130) = + extfrc(:,:,2)
                prod(:,:,131) = .660_r8*rxt(:,:,60)*y(:,:,12) + extfrc(:,:,18)
                prod(:,:,132) = 0._r8
                prod(:,:,133) = 0._r8
                prod(:,:,60) = 0._r8
                prod(:,:,40) = 0._r8
                prod(:,:,119) = rxt(:,:,59)*y(:,:,12) +rxt(:,:,37)*y(:,:,79) +rxt(:,:,48)                  *y(:,:,80)
                prod(:,:,50) = 0._r8
                prod(:,:,30) = 0._r8
                prod(:,:,17) = 0._r8
                prod(:,:,135) = .180_r8*rxt(:,:,60)*y(:,:,12)
                prod(:,:,127) = rxt(:,:,59)*y(:,:,12)
                prod(:,:,125) = 0._r8
                prod(:,:,74) = 0._r8
                prod(:,:,134) = .050_r8*rxt(:,:,60)*y(:,:,12)
                prod(:,:,126) = rxt(:,:,37)*y(:,:,79) +2.000_r8*rxt(:,:,40)*y(:,:,81)                   +2.000_r8*rxt(:,:,41)*y(:,&
                :,82) +2.000_r8*rxt(:,:,42)*y(:,:,83)                   +rxt(:,:,45)*y(:,:,84) +4.000_r8*rxt(:,:,38)*y(:,:,85)    &
                               +3.000_r8*rxt(:,:,39)*y(:,:,86) +rxt(:,:,50)*y(:,:,88) +rxt(:,:,46)                  *y(:,:,89) &
                +rxt(:,:,47)*y(:,:,90) +2.000_r8*rxt(:,:,43)*y(:,:,91)                   +rxt(:,:,44)*y(:,:,92)
                prod(:,:,29) = 0._r8
                prod(:,:,124) = 0._r8
                prod(:,:,46) = 0._r8
                prod(:,:,18) = 0._r8
                prod(:,:,117) = 0._r8
                prod(:,:,93) = 0._r8
                prod(:,:,100) = 0._r8
                prod(:,:,33) = 0._r8
                prod(:,:,118) = rxt(:,:,48)*y(:,:,80) +rxt(:,:,49)*y(:,:,87) +rxt(:,:,50)                  *y(:,:,88) &
                +2.000_r8*rxt(:,:,53)*y(:,:,93) +2.000_r8*rxt(:,:,54)                  *y(:,:,94) +3.000_r8*rxt(:,:,51)*y(:,:,95) &
                +2.000_r8*rxt(:,:,52)                  *y(:,:,96)
                prod(:,:,128) = 0._r8
                prod(:,:,90) = 0._r8
                prod(:,:,84) = 0._r8
                prod(:,:,70) = 0._r8
                prod(:,:,78) = (rxt(:,:,97) +rxt(:,:,109)) + extfrc(:,:,14)
                prod(:,:,85) = + extfrc(:,:,12)
                prod(:,:,58) = (rxt(:,:,101) +rxt(:,:,102) +rxt(:,:,113) +rxt(:,:,114))                   + extfrc(:,:,13)
                prod(:,:,72) = + extfrc(:,:,11)
                prod(:,:,86) = 0._r8
                prod(:,:,61) = (rxt(:,:,102) +1.200_r8*rxt(:,:,104) +rxt(:,:,114) +                  1.200_r8*rxt(:,:,116)) + &
                extfrc(:,:,15)
                prod(:,:,87) = (rxt(:,:,97) +rxt(:,:,101) +rxt(:,:,102) +rxt(:,:,109) +                  rxt(:,:,113) +rxt(:,:,&
                114)) + extfrc(:,:,17)
                prod(:,:,102) = 0._r8
                prod(:,:,94) = 0._r8
                prod(:,:,89) = 0._r8
                prod(:,:,104) = 0._r8
                prod(:,:,75) = 0._r8
                prod(:,:,67) = 0._r8
                prod(:,:,115) = 0._r8
                prod(:,:,62) = 0._r8
                prod(:,:,57) = 0._r8
                prod(:,:,49) = 0._r8
                prod(:,:,37) = 0._r8
                prod(:,:,63) = 0._r8
                prod(:,:,19) = 0._r8
                prod(:,:,71) = 0._r8
                prod(:,:,20) = 0._r8
                prod(:,:,41) = 0._r8
                prod(:,:,79) = 0._r8
                prod(:,:,76) = 0._r8
                prod(:,:,55) = 0._r8
                prod(:,:,77) = 0._r8
                prod(:,:,42) = 0._r8
                prod(:,:,22) = 0._r8
                prod(:,:,23) = 0._r8
                prod(:,:,65) = 0._r8
                prod(:,:,51) = 0._r8
                prod(:,:,31) = 0._r8
                prod(:,:,98) = 0._r8
                prod(:,:,59) = 0._r8
                prod(:,:,66) = 0._r8
                prod(:,:,81) = 0._r8
                prod(:,:,111) = 0._r8
                prod(:,:,113) = 0._r8
                prod(:,:,107) = 0._r8
                prod(:,:,112) = 0._r8
                prod(:,:,43) = 0._r8
                prod(:,:,114) = 0._r8
                prod(:,:,91) = 0._r8
                prod(:,:,44) = 0._r8
                prod(:,:,73) = 0._r8
                prod(:,:,21) = 0._r8
                prod(:,:,96) = 0._r8
                prod(:,:,52) = 0._r8
                prod(:,:,80) = 0._r8
                prod(:,:,53) = 0._r8
                prod(:,:,68) = 0._r8
                prod(:,:,35) = 0._r8
                prod(:,:,95) = 0._r8
                prod(:,:,105) = 0._r8
                prod(:,:,83) = 0._r8
                prod(:,:,56) = 0._r8
                prod(:,:,24) = 0._r8
                prod(:,:,47) = 0._r8
                prod(:,:,106) = 0._r8
                prod(:,:,109) = 0._r8
                prod(:,:,101) = 0._r8
                prod(:,:,97) = 0._r8
                prod(:,:,110) = 0._r8
                prod(:,:,45) = 0._r8
                prod(:,:,69) = 0._r8
                prod(:,:,38) = 0._r8
                prod(:,:,64) = 0._r8
                prod(:,:,54) = 0._r8
                prod(:,:,25) = rxt(:,:,41)*y(:,:,82) +rxt(:,:,42)*y(:,:,83) +rxt(:,:,45)                  *y(:,:,84) +rxt(:,:,49)&
                *y(:,:,87) +rxt(:,:,50)*y(:,:,88) +rxt(:,:,47)                  *y(:,:,90) +2.000_r8*rxt(:,:,43)*y(:,:,91) &
                +2.000_r8*rxt(:,:,44)                  *y(:,:,92) +rxt(:,:,53)*y(:,:,93) +2.000_r8*rxt(:,:,54)*y(:,:,94)
                prod(:,:,32) = rxt(:,:,40)*y(:,:,81) +rxt(:,:,42)*y(:,:,83) +rxt(:,:,46)                  *y(:,:,89)
                prod(:,:,34) = 0._r8
                prod(:,:,88) = rxt(:,:,49)*y(:,:,87) +rxt(:,:,44)*y(:,:,92)
                prod(:,:,99) = + extfrc(:,:,4)
                prod(:,:,39) = 0._r8
                prod(:,:,48) = 0._r8
                prod(:,:,82) = 0._r8
                prod(:,:,116) = 0._r8
                prod(:,:,36) = 0._r8
                prod(:,:,26) = 0._r8
                prod(:,:,1) = 0._r8
                prod(:,:,2) = + extfrc(:,:,5)
                prod(:,:,3) = + extfrc(:,:,7)
                prod(:,:,4) = 0._r8
                prod(:,:,5) = + extfrc(:,:,8)
                prod(:,:,6) = 0._r8
                prod(:,:,7) = 0._r8
                prod(:,:,8) = + extfrc(:,:,9)
                prod(:,:,9) = + extfrc(:,:,6)
                prod(:,:,10) = 0._r8
                prod(:,:,11) = 0._r8
                prod(:,:,12) = + extfrc(:,:,10)
                prod(:,:,13) = 0._r8
                prod(:,:,14) = 0._r8
                prod(:,:,15) = 0._r8
                prod(:,:,16) = 0._r8
            END IF 
        END SUBROUTINE indprd
    END MODULE mo_indprd
