
! KGEN-generated Fortran source file
!
! Filename    : mcica_random_numbers.f90
! Generated at: 2015-07-07 00:48:25
! KGEN version: 0.4.13



    MODULE mersennetwister
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        ! -------------------------------------------------------------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !  use parkind, only : jpim, jprb
        IMPLICIT NONE
        PRIVATE
        ! Algorithm parameters
        ! -------
        ! Period parameters
        INTEGER, parameter :: blocksize = 624
        INTEGER, parameter :: lmask     =  2147483647
        INTEGER, parameter :: umask     = (-lmask) - 1
        INTEGER, parameter :: m         = 397
        INTEGER, parameter :: matrix_a  = -1727483681
        ! constant vector a         (0x9908b0dfUL)
        ! least significant r bits (0x7fffffffUL)
        ! most significant w-r bits (0x80000000UL)
        ! Tempering parameters
        INTEGER, parameter :: tmaskb= -1658038656
        INTEGER, parameter :: tmaskc= -272236544 ! (0x9d2c5680UL)
        ! (0xefc60000UL)
        ! -------
        ! The type containing the state variable
        TYPE randomnumbersequence
            INTEGER :: currentelement ! = blockSize
            INTEGER, dimension(0:blocksize -1) :: state ! = 0
        END TYPE randomnumbersequence

        INTERFACE new_randomnumbersequence
            MODULE PROCEDURE initialize_scalar, initialize_vector
        END INTERFACE new_randomnumbersequence
        PUBLIC randomnumbersequence
        PUBLIC new_randomnumbersequence, getrandomreal, getrandomint
        ! -------------------------------------------------------------

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_randomnumbersequence
        END INTERFACE kgen_read

        PUBLIC kgen_verify
        INTERFACE kgen_verify
            MODULE PROCEDURE kgen_verify_randomnumbersequence
        END INTERFACE kgen_verify

        CONTAINS

        ! write subroutines
        ! No module extern variables
        SUBROUTINE kgen_read_randomnumbersequence(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(randomnumbersequence), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%currentelement
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%currentelement **", var%currentelement
            END IF
            READ(UNIT=kgen_unit) var%state
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%state **", var%state
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_randomnumbersequence(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(randomnumbersequence), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_integer("currentelement", dtype_check_status, var%currentelement, ref_var%currentelement)
            CALL kgen_verify_integer_4_dim1("state", dtype_check_status, var%state, ref_var%state)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
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

            SUBROUTINE kgen_verify_integer_4_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer, intent(in), DIMENSION(:) :: var, ref_var
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
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                    end if
                
                    check_status%numFatal = check_status%numFatal+1
                END IF
            END SUBROUTINE kgen_verify_integer_4_dim1

        ! -------------------------------------------------------------
        ! Private functions
        ! ---------------------------

        FUNCTION mixbits(u, v)
            INTEGER, intent( in) :: u
            INTEGER, intent( in) :: v
            INTEGER :: mixbits
    mixbits = ior(iand(u, UMASK), iand(v, LMASK))
        END FUNCTION mixbits
        ! ---------------------------

        FUNCTION twist(u, v)
            INTEGER, intent( in) :: u
            INTEGER, intent( in) :: v
            INTEGER :: twist
            ! Local variable
            INTEGER, parameter, dimension(0:1) :: t_matrix = (/ 0, matrix_a /)
    twist = ieor(ishft(mixbits(u, v), -1), t_matrix(iand(v, 1)))
    twist = ieor(ishft(mixbits(u, v), -1), t_matrix(iand(v, 1)))
        END FUNCTION twist
        ! ---------------------------

        SUBROUTINE nextstate(twister)
            TYPE(randomnumbersequence), intent(inout) :: twister
            ! Local variables
            INTEGER :: k
    do k = 0, blockSize - M - 1
      twister%state(k) = ieor(twister%state(k + M), &
                              twist(twister%state(k), twister%state(k + 1)))
    end do 
    do k = blockSize - M, blockSize - 2
      twister%state(k) = ieor(twister%state(k + M - blockSize), &
                              twist(twister%state(k), twister%state(k + 1)))
    end do 
    twister%state(blockSize - 1) = ieor(twister%state(M - 1), &
                                        twist(twister%state(blockSize - 1), twister%state(0)))
    twister%currentElement = 0
        END SUBROUTINE nextstate
        ! ---------------------------

        elemental FUNCTION temper(y)
            INTEGER, intent(in) :: y
            INTEGER :: temper
            INTEGER :: x
            ! Tempering
    x      = ieor(y, ishft(y, -11))
    x      = ieor(x, iand(ishft(x,  7), TMASKB))
    x      = ieor(x, iand(ishft(x, 15), TMASKC))
    temper = ieor(x, ishft(x, -18))
        END FUNCTION temper
        ! -------------------------------------------------------------
        ! Public (but hidden) functions
        ! --------------------

        FUNCTION initialize_scalar(seed) RESULT ( twister )
            INTEGER, intent(in   ) :: seed
            TYPE(randomnumbersequence) :: twister
            INTEGER :: i
            ! See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. In the previous versions,
            !   MSBs of the seed affect only MSBs of the array state[].
            !   2002/01/09 modified by Makoto Matsumoto
    twister%state(0) = iand(seed, -1)
    do i = 1,  blockSize - 1 ! ubound(twister%state) ! ubound(twister%state)
       twister%state(i) = 1812433253 * ieor(twister%state(i-1), &
                                            ishft(twister%state(i-1), -30)) + i
       twister%state(i) = iand(twister%state(i), -1) ! for >32 bit machines ! for >32 bit machines
    end do
    twister%currentElement = blockSize
        END FUNCTION initialize_scalar
        ! -------------------------------------------------------------

        FUNCTION initialize_vector(seed) RESULT ( twister )
            INTEGER, dimension(0:), intent(in) :: seed
            TYPE(randomnumbersequence) :: twister
            INTEGER :: nwraps
            INTEGER :: nfirstloop
            INTEGER :: k
            INTEGER :: i
            INTEGER :: j
    nWraps  = 0
    twister = initialize_scalar(19650218)
    nFirstLoop = max(blockSize, size(seed))
    do k = 1, nFirstLoop
       i = mod(k + nWraps, blockSize)
       j = mod(k - 1,      size(seed))
       if(i == 0) then
         twister%state(i) = twister%state(blockSize - 1)
         twister%state(1) = ieor(twister%state(1),                                 &
                                 ieor(twister%state(1-1),                          & 
                                      ishft(twister%state(1-1), -30)) * 1664525) + & 
                            seed(j) + j ! Non-linear
                    ! Non-linear
         twister%state(i) = iand(twister%state(i), -1) ! for >32 bit machines ! for >32 bit machines
         nWraps = nWraps + 1
       else
         twister%state(i) = ieor(twister%state(i),                                 &
                                 ieor(twister%state(i-1),                          & 
                                      ishft(twister%state(i-1), -30)) * 1664525) + & 
                            seed(j) + j ! Non-linear
                    ! Non-linear
         twister%state(i) = iand(twister%state(i), -1) ! for >32 bit machines ! for >32 bit machines
      end if
    end do
            !
            ! Walk through the state array, beginning where we left off in the block above
            !
    do i = mod(nFirstLoop, blockSize) + nWraps + 1, blockSize - 1
      twister%state(i) = ieor(twister%state(i),                                 &
                              ieor(twister%state(i-1),                          & 
                                   ishft(twister%state(i-1), -30)) * 1566083941) - i ! Non-linear
                ! Non-linear
      twister%state(i) = iand(twister%state(i), -1) ! for >32 bit machines ! for >32 bit machines
    end do
    twister%state(0) = twister%state(blockSize - 1) 
    do i = 1, mod(nFirstLoop, blockSize) + nWraps
      twister%state(i) = ieor(twister%state(i),                                 &
                              ieor(twister%state(i-1),                          & 
                                   ishft(twister%state(i-1), -30)) * 1566083941) - i ! Non-linear
                ! Non-linear
      twister%state(i) = iand(twister%state(i), -1) ! for >32 bit machines ! for >32 bit machines
    end do
    twister%state(0) = UMASK 
    twister%currentElement = blockSize
        END FUNCTION initialize_vector
        ! -------------------------------------------------------------
        ! Public functions
        ! --------------------

        FUNCTION getrandomint(twister)
            TYPE(randomnumbersequence), intent(inout) :: twister
            INTEGER :: getrandomint
            ! Generate a random integer on the interval [0,0xffffffff]
            !   Equivalent to genrand_int32 in the C code.
            !   Fortran doesn't have a type that's unsigned like C does,
            !   so this is integers in the range -2**31 - 2**31
            ! All functions for getting random numbers call this one,
            !   then manipulate the result
    if(twister%currentElement >= blockSize) call nextState(twister)
    getRandomInt = temper(twister%state(twister%currentElement))
    twister%currentElement = twister%currentElement + 1
        END FUNCTION getrandomint
        ! --------------------

        ! --------------------
        !! mji - modified Jan 2007, double converted to rrtmg real kind type

        FUNCTION getrandomreal(twister)
            TYPE(randomnumbersequence), intent(inout) :: twister
            !    double precision             :: getRandomReal
            REAL(KIND=r8) :: getrandomreal
            ! Generate a random number on [0,1]
            !   Equivalent to genrand_real1 in the C code
            !   The result is stored as double precision but has 32 bit resolution
            INTEGER :: localint
    localInt = getRandomInt(twister)
    if(localInt < 0) then
                !      getRandomReal = dble(localInt + 2.0d0**32)/(2.0d0**32 - 1.0d0)
      getRandomReal = (localInt + 2.0**32_r8)/(2.0**32_r8 - 1.0_r8)
    else
                !      getRandomReal = dble(localInt            )/(2.0d0**32 - 1.0d0)
      getRandomReal = (localInt            )/(2.0**32_r8 - 1.0_r8)
    end if
        END FUNCTION getrandomreal
        ! --------------------

        ! --------------------
    END MODULE mersennetwister

    MODULE mcica_random_numbers
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        ! Generic module to wrap random number generators.
        !   The module defines a type that identifies the particular stream of random
        !   numbers, and has procedures for initializing it and getting real numbers
        !   in the range 0 to 1.
        ! This version uses the Mersenne Twister to generate random numbers on [0, 1].
        !
        ! The random number engine.
        !! mji
        !!  use time_manager_mod, only: time_type, get_date
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !  use parkind, only : jpim, jprb
        IMPLICIT NONE
        PRIVATE


        !! mji
        !!            initializeRandomNumberStream, getRandomNumbers, &
        !!            constructSeed
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        ! ---------------------------------------------------------
        ! Initialization
        ! ---------------------------------------------------------

        ! ---------------------------------------------------------

        ! ---------------------------------------------------------
        ! Procedures for drawing random numbers
        ! ---------------------------------------------------------

        ! ---------------------------------------------------------

        ! ---------------------------------------------------------

        ! mji
        !  ! ---------------------------------------------------------
        !  ! Constructing a unique seed from grid cell index and model date/time
        !  !   Once we have the GFDL stuff we'll add the year, month, day, hour, minute
        !  ! ---------------------------------------------------------
        !  function constructSeed(i, j, time) result(seed)
        !    integer,         intent( in)  :: i, j
        !    type(time_type), intent( in) :: time
        !    integer, dimension(8) :: seed
        !
        !    ! Local variables
        !    integer :: year, month, day, hour, minute, second
        !
        !
        !    call get_date(time, year, month, day, hour, minute, second)
        !    seed = (/ i, j, year, month, day, hour, minute, second /)
        !  end function constructSeed
    END MODULE mcica_random_numbers
