
! KGEN-generated Fortran source file
!
! Filename    : mo_cld_sampling.f90
! Generated at: 2015-02-19 15:30:32
! KGEN version: 0.4.4



    MODULE mo_cld_sampling
        USE mo_kind, ONLY: wp
        USE mo_exception, ONLY: finish
        USE mo_random_numbers, ONLY: get_random
        IMPLICIT NONE
        PRIVATE
        PUBLIC sample_cld_state
        CONTAINS

        ! read subroutines
        !-----------------------------------------------------------------------------
        !>
        !! @brief Returns a sample of the cloud state
        !!
        !! @remarks
        !

        SUBROUTINE sample_cld_state(kproma, kbdim, klev, ksamps, rnseeds, i_overlap, cld_frac, cldy)
            INTEGER, intent(in) :: kbdim
            INTEGER, intent(in) :: klev
            INTEGER, intent(in) :: ksamps
            INTEGER, intent(in) :: kproma !< numbers of columns, levels, samples
            INTEGER, intent(inout) :: rnseeds(:, :) !< Seeds for random number generator (kbdim, :)
            INTEGER, intent(in) :: i_overlap !< 1=max-ran, 2=maximum, 3=random
            REAL(KIND=wp), intent(in) :: cld_frac(kbdim,klev) !< cloud fraction
            LOGICAL, intent(out) :: cldy(kbdim,klev,ksamps) !< Logical: cloud present?
            REAL(KIND=wp) :: rank(kbdim,klev,ksamps)
            INTEGER :: js
            INTEGER :: jk
            ! Here cldy(:,:,1) indicates whether any cloud is present
            !
            cldy(1:kproma,1:klev,1) = cld_frac(1:kproma,1:klev) > 0._wp
            SELECT CASE ( i_overlap )
                CASE ( 1 )
                ! Maximum-random overlap
                DO js = 1, ksamps
                    DO jk = 1, klev
                        ! mask means we compute random numbers only when cloud is present
                        CALL get_random(kproma, kbdim, rnseeds, cldy(:,jk,1), rank(:,jk,js))
                    END DO 
                END DO 
                ! There may be a better way to structure this calculation...
                DO jk = klev-1, 1, -1
                    DO js = 1, ksamps
                        rank(1:kproma,jk,js) = merge(rank(1:kproma,jk+1,js),                                                      &
                                        rank(1:kproma,jk,js) * (1._wp - cld_frac(1:kproma,jk+1)),                                 &
                             rank(1:kproma,jk+1,js) > 1._wp - cld_frac(1:kproma,jk+1))
                        ! Max overlap...
                        ! ... or random overlap in the clear sky portion,
                        ! depending on whether or not you have cloud in the layer above
                    END DO 
                END DO 
                CASE ( 2 )
                !
                !  Max overlap means every cell in a column is identical
                !
                DO js = 1, ksamps
                    CALL get_random(kproma, kbdim, rnseeds, rank(:, 1, js))
                    rank(1:kproma,2:klev,js) = spread(rank(1:kproma,1,js), dim=2, ncopies=(klev-1))
                END DO 
                CASE ( 3 )
                !
                !  Random overlap means every cell is independent
                !
                DO js = 1, ksamps
                    DO jk = 1, klev
                        ! mask means we compute random numbers only when cloud is present
                        CALL get_random(kproma, kbdim, rnseeds, cldy(:,jk,1), rank(:,jk,js))
                    END DO 
                END DO 
                CASE DEFAULT
                CALL finish('In sample_cld_state: unknown overlap assumption')
            END SELECT 
            ! Now cldy indicates whether the sample (ks) is cloudy or not.
            DO js = 1, ksamps
                cldy(1:kproma,1:klev,js) = rank(1:kproma,1:klev,js) > (1. - cld_frac(1:kproma,1:klev))
            END DO 
        END SUBROUTINE sample_cld_state
    END MODULE mo_cld_sampling
