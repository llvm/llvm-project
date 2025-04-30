
! KGEN-generated Fortran source file
!
! Filename    : cmparray_mod.F90
! Generated at: 2015-07-07 00:48:24
! KGEN version: 0.4.13



    MODULE cmparray_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        IMPLICIT NONE
        PRIVATE
        PUBLIC cmpdaynite, expdaynite

        INTERFACE cmpdaynite
            MODULE PROCEDURE cmpdaynite_1d_r
            MODULE PROCEDURE cmpdaynite_2d_r
            MODULE PROCEDURE cmpdaynite_3d_r
            MODULE PROCEDURE cmpdaynite_1d_r_copy
            MODULE PROCEDURE cmpdaynite_2d_r_copy
            MODULE PROCEDURE cmpdaynite_3d_r_copy
            MODULE PROCEDURE cmpdaynite_1d_i
            MODULE PROCEDURE cmpdaynite_2d_i
            MODULE PROCEDURE cmpdaynite_3d_i
        END INTERFACE  ! CmpDayNite

        INTERFACE expdaynite
            MODULE PROCEDURE expdaynite_1d_r
            MODULE PROCEDURE expdaynite_2d_r
            MODULE PROCEDURE expdaynite_3d_r
            MODULE PROCEDURE expdaynite_1d_i
            MODULE PROCEDURE expdaynite_2d_i
            MODULE PROCEDURE expdaynite_3d_i
        END INTERFACE  ! ExpDayNite

        ! cmparray

        ! chksum
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables

        SUBROUTINE cmpdaynite_1d_r(array, nday, idxday, nnite, idxnite, il1, iu1)
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            REAL(KIND=r8), intent(inout), dimension(il1:iu1) :: array
    call CmpDayNite_3d_R(Array, Nday, IdxDay, Nnite, IdxNite, il1, iu1, 1, 1, 1, 1)
    return
        END SUBROUTINE cmpdaynite_1d_r

        SUBROUTINE cmpdaynite_2d_r(array, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            REAL(KIND=r8), intent(inout), dimension(il1:iu1,il2:iu2) :: array
    call CmpDayNite_3d_R(Array, Nday, IdxDay, Nnite, IdxNite, il1, iu1, il2, iu2, 1, 1)
    return
        END SUBROUTINE cmpdaynite_2d_r

        SUBROUTINE cmpdaynite_3d_r(array, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2, il3, iu3)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: il3
            INTEGER, intent(in) :: iu3
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            REAL(KIND=r8), intent(inout), dimension(il1:iu1,il2:iu2,il3:iu3) :: array
            REAL(KIND=r8), dimension(il1:iu1) :: tmp
            INTEGER :: k
            INTEGER :: j
    do k = il3, iu3
      do j = il2, iu2
        tmp(1:Nnite) = Array(IdxNite(1:Nnite),j,k)
        Array(il1:il1+Nday-1,j,k) = Array(IdxDay(1:Nday),j,k)
        Array(il1+Nday:il1+Nday+Nnite-1,j,k) = tmp(1:Nnite)
      end do
    end do
    return
        END SUBROUTINE cmpdaynite_3d_r

        SUBROUTINE cmpdaynite_1d_r_copy(inarray, outarray, nday, idxday, nnite, idxnite, il1, iu1)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            REAL(KIND=r8), intent(in), dimension(il1:iu1) :: inarray
            REAL(KIND=r8), intent(out), dimension(il1:iu1) :: outarray
    call CmpDayNite_3d_R_Copy(InArray, OutArray, Nday, IdxDay, Nnite, IdxNite, il1, iu1, 1, 1, 1, 1)
    return
        END SUBROUTINE cmpdaynite_1d_r_copy

        SUBROUTINE cmpdaynite_2d_r_copy(inarray, outarray, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2)
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            REAL(KIND=r8), intent(in), dimension(il1:iu1,il2:iu2) :: inarray
            REAL(KIND=r8), intent(out), dimension(il1:iu1,il2:iu2) :: outarray
    call CmpDayNite_3d_R_Copy(InArray, OutArray, Nday, IdxDay, Nnite, IdxNite, il1, iu1, il2, iu2, 1, 1)
    return
        END SUBROUTINE cmpdaynite_2d_r_copy

        SUBROUTINE cmpdaynite_3d_r_copy(inarray, outarray, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2, il3, iu3)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in) :: il3
            INTEGER, intent(in) :: iu3
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            REAL(KIND=r8), intent(in), dimension(il1:iu1,il2:iu2,il3:iu3) :: inarray
            REAL(KIND=r8), intent(out), dimension(il1:iu1,il2:iu2,il3:iu3) :: outarray
            INTEGER :: k
            INTEGER :: j
            INTEGER :: i
    do k = il3, iu3
      do j = il2, iu2
         do i=il1,il1+Nday-1
            OutArray(i,j,k) = InArray(IdxDay(i-il1+1),j,k)
         enddo
         do i=il1+Nday,il1+Nday+Nnite-1
            OutArray(i,j,k) = InArray(IdxNite(i-(il1+Nday)+1),j,k)
         enddo
      end do
    end do
    return
        END SUBROUTINE cmpdaynite_3d_r_copy

        SUBROUTINE cmpdaynite_1d_i(array, nday, idxday, nnite, idxnite, il1, iu1)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            INTEGER, intent(inout), dimension(il1:iu1) :: array
    call CmpDayNite_3d_I(Array, Nday, IdxDay, Nnite, IdxNite, il1, iu1, 1, 1, 1, 1)
    return
        END SUBROUTINE cmpdaynite_1d_i

        SUBROUTINE cmpdaynite_2d_i(array, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            INTEGER, intent(inout), dimension(il1:iu1,il2:iu2) :: array
    call CmpDayNite_3d_I(Array, Nday, IdxDay, Nnite, IdxNite, il1, iu1, il2, iu2, 1, 1)
    return
        END SUBROUTINE cmpdaynite_2d_i

        SUBROUTINE cmpdaynite_3d_i(array, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2, il3, iu3)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in) :: iu3
            INTEGER, intent(in) :: il3
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            INTEGER, intent(inout), dimension(il1:iu1,il2:iu2,il3:iu3) :: array
            INTEGER, dimension(il1:iu1) :: tmp
            INTEGER :: k
            INTEGER :: j
    do k = il3, iu3
      do j = il2, iu2
        tmp(1:Nnite) = Array(IdxNite(1:Nnite),j,k)
        Array(il1:il1+Nday-1,j,k) = Array(IdxDay(1:Nday),j,k)
        Array(il1+Nday:il1+Nday+Nnite-1,j,k) = tmp(1:Nnite)
      end do
    end do
    return
        END SUBROUTINE cmpdaynite_3d_i

        SUBROUTINE expdaynite_1d_r(array, nday, idxday, nnite, idxnite, il1, iu1)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            REAL(KIND=r8), intent(inout), dimension(il1:iu1) :: array
    call ExpDayNite_3d_R(Array, Nday, IdxDay, Nnite, IdxNite, il1, iu1, 1, 1, 1, 1)
    return
        END SUBROUTINE expdaynite_1d_r

        SUBROUTINE expdaynite_2d_r(array, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            REAL(KIND=r8), intent(inout), dimension(il1:iu1,il2:iu2) :: array
    call ExpDayNite_3d_R(Array, Nday, IdxDay, Nnite, IdxNite, il1, iu1, il2, iu2, 1, 1)
    return
        END SUBROUTINE expdaynite_2d_r

        SUBROUTINE expdaynite_3d_r(array, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2, il3, iu3)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in) :: il3
            INTEGER, intent(in) :: iu3
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            REAL(KIND=r8), intent(inout), dimension(il1:iu1,il2:iu2,il3:iu3) :: array
            REAL(KIND=r8), dimension(il1:iu1) :: tmp
            INTEGER :: k
            INTEGER :: j
    do k = il3, iu3
      do j = il2, iu2
        tmp(1:Nday) = Array(1:Nday,j,k)
        Array(IdxNite(1:Nnite),j,k) = Array(il1+Nday:il1+Nday+Nnite-1,j,k)
        Array(IdxDay(1:Nday),j,k) = tmp(1:Nday)
      end do
    end do
    return
        END SUBROUTINE expdaynite_3d_r

        SUBROUTINE expdaynite_1d_i(array, nday, idxday, nnite, idxnite, il1, iu1)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            INTEGER, intent(inout), dimension(il1:iu1) :: array
    call ExpDayNite_3d_I(Array, Nday, IdxDay, Nnite, IdxNite, il1, iu1, 1, 1, 1, 1)
    return
        END SUBROUTINE expdaynite_1d_i

        SUBROUTINE expdaynite_2d_i(array, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            INTEGER, intent(inout), dimension(il1:iu1,il2:iu2) :: array
    call ExpDayNite_3d_I(Array, Nday, IdxDay, Nnite, IdxNite, il1, iu1, il2, iu2, 1, 1)
    return
        END SUBROUTINE expdaynite_2d_i

        SUBROUTINE expdaynite_3d_i(array, nday, idxday, nnite, idxnite, il1, iu1, il2, iu2, il3, iu3)
            INTEGER, intent(in) :: nday
            INTEGER, intent(in) :: nnite
            INTEGER, intent(in) :: il1
            INTEGER, intent(in) :: iu1
            INTEGER, intent(in) :: iu2
            INTEGER, intent(in) :: il2
            INTEGER, intent(in) :: il3
            INTEGER, intent(in) :: iu3
            INTEGER, intent(in), dimension(nday) :: idxday
            INTEGER, intent(in), dimension(nnite) :: idxnite
            INTEGER, intent(inout), dimension(il1:iu1,il2:iu2,il3:iu3) :: array
            INTEGER, dimension(il1:iu1) :: tmp
            INTEGER :: k
            INTEGER :: j
    do k = il3, iu3
      do j = il2, iu2
        tmp(1:Nday) = Array(1:Nday,j,k)
        Array(IdxNite(1:Nnite),j,k) = Array(il1+Nday:il1+Nday+Nnite-1,j,k)
        Array(IdxDay(1:Nday),j,k) = tmp(1:Nday)
      end do
    end do
    return
        END SUBROUTINE expdaynite_3d_i
        !******************************************************************************!
        !                                                                              !
        !                                 DEBUG                                        !
        !                                                                              !
        !******************************************************************************!









    END MODULE cmparray_mod
