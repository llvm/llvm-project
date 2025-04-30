
! KGEN-generated Fortran source file
!
! Filename    : mo_spec_sampling.f90
! Generated at: 2015-02-19 15:30:31
! KGEN version: 0.4.4



    MODULE mo_spec_sampling
        USE mo_random_numbers, ONLY: get_random
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        PRIVATE
        !
        ! Team choices - Longwave
        !
        !
        ! Team choices - Shortwave
        !
        !
        ! Encapsulate the strategy
        !
        TYPE spec_sampling_strategy
            PRIVATE
            INTEGER, dimension(:, :), pointer :: teams => null()
            INTEGER :: num_gpts_ts ! How many g points at each time step
            LOGICAL :: unique = .false.
        END TYPE spec_sampling_strategy
        PUBLIC spec_sampling_strategy, get_gpoint_set

        ! read interface
        PUBLIC kgen_read_var
        interface kgen_read_var
            module procedure read_var_integer_4_dim2_pointer
            module procedure read_var_spec_sampling_strategy
        end interface kgen_read_var

        CONTAINS
        subroutine read_var_spec_sampling_strategy(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            type(spec_sampling_strategy), intent(out) :: var
        
            call kgen_read_var(var%teams, kgen_unit, .true.)
            READ(UNIT=kgen_unit) var%num_gpts_ts
            READ(UNIT=kgen_unit) var%unique
        end subroutine

        ! read subroutines
        subroutine read_var_integer_4_dim2_pointer(var, kgen_unit, is_pointer)
            integer, intent(in) :: kgen_unit
            logical, intent(in) :: is_pointer
            integer(kind=4), intent(out), dimension(:,:), pointer :: var
            integer, dimension(2,2) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine
        ! -----------------------------------------------------------------------------------------------
        !>
        !! @brief Sets a spectral sampling strategy
        !!
        !! @remarks: Choose a set of g-point teams to use.
        !!   Two end-member choices:
        !!   strategy = 1 : a single team comprising all g-points, i.e. broadband integration
        !!   strategy = 2 : ngpts teams of a single g-point each, i.e. a single randomly chosen g-point
        !!     This can be modified to choose m samples at each time step (with or without replacement, eventually)
        !!   Other strategies must combine n teams of m gpoints each such that m * n = ngpts
        !!   strategy 1 (broadband) is the default
        !!
        !

        ! -----------------------------------------------------------------------------------------------
        ! -----------------------------------------------------------------------------------------------
        !>
        !! @brief Sets a spectral sampling strategy
        !!
        !! @remarks: Choose a set of g-point teams to use.
        !!   Two end-member choices:
        !!   strategy = 1 : a single team comprising all g-points, i.e. broadband integration
        !!   strategy = 2 : ngpts teams of a single g-point each, i.e. a single randomly chosen g-point
        !!     This can be modified to choose m samples at each time step (with or without replacement, eventually)
        !!   Other strategies must combine n teams of m gpoints each such that m * n = ngpts
        !!   strategy 1 (broadband) is the default
        !!
        !

        ! -----------------------------------------------------------------------------------------------
        !>
        !! @brief Returns the number of g-points to compute at each time step
        !!

        ! -----------------------------------------------------------------------------------------------
        !>
        !! @brief Returns one set of g-points consistent with sampling strategy
        !!

        FUNCTION get_gpoint_set(kproma, kbdim, strategy, seeds)
            INTEGER, intent(in) :: kproma
            INTEGER, intent(in) :: kbdim
            TYPE(spec_sampling_strategy), intent(in) :: strategy
            INTEGER, intent(inout) :: seeds(:,:) ! dimensions kbdim, rng seed_size
            INTEGER, dimension(kproma, strategy%num_gpts_ts) :: get_gpoint_set
            REAL(KIND=wp) :: rn(kbdim)
            INTEGER :: team(kbdim)
            INTEGER :: num_teams
            INTEGER :: num_gpts_team
            INTEGER :: jl
            INTEGER :: it
            ! --------
            num_teams = size(strategy%teams, 2)
            num_gpts_team = size(strategy%teams, 1)
            IF (num_teams == 1) THEN
                !
                ! Broadband integration
                !
                get_gpoint_set(1:kproma,:) = spread(strategy%teams(:, 1), dim = 1, ncopies = kproma)
                ELSE IF (num_gpts_team > 1) THEN
                !
                ! Mutiple g-points per team, including broadband integration
                !   Return just one team
                !
                CALL get_random(kproma, kbdim, seeds, rn)
                team(1:kproma) = min(int(rn(1:kproma) * num_teams) + 1, num_teams)
                DO jl = 1, kproma
                    get_gpoint_set(jl, :) = strategy%teams(:,team(jl))
                END DO 
                ELSE
                !
                ! MCSI - return one or more individual points chosen randomly
                !   Need to add option for sampling without replacement
                !
                DO it = 1, strategy%num_gpts_ts
                    CALL get_random(kproma, kbdim, seeds, rn)
                    team(1:kproma) = min(int(rn(1:kproma) * num_teams) + 1, num_teams)
                    get_gpoint_set(1:kproma, it) = strategy%teams(1, team(1:kproma))
                END DO 
            END IF 
        END FUNCTION get_gpoint_set
        ! -----------------------------------------------------------------------------------------------
    END MODULE mo_spec_sampling
