
! KGEN-generated Fortran source file
!
! Filename    : coordinate_systems_mod.F90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9



    MODULE coordinate_systems_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        ! WARNING:  When using this class be sure that you know if the
        ! cubic coordinates are on the unit cube or the [-\pi/4,\pi/4] cube
        ! and if the spherical longitude is in [0,2\pi] or [-\pi,\pi]
        USE kinds, ONLY: real_kind
        IMPLICIT NONE
        PRIVATE
        TYPE, public :: cartesian2d_t
            REAL(KIND=real_kind) :: x ! x coordinate
            REAL(KIND=real_kind) :: y ! y coordinate
        END TYPE cartesian2d_t
        TYPE, public :: cartesian3d_t
            REAL(KIND=real_kind) :: x ! x coordinate
            REAL(KIND=real_kind) :: y ! y coordinate
            REAL(KIND=real_kind) :: z ! z coordinate
        END TYPE cartesian3d_t
        TYPE, public :: spherical_polar_t
            REAL(KIND=real_kind) :: r ! radius
            REAL(KIND=real_kind) :: lon ! longitude
            REAL(KIND=real_kind) :: lat ! latitude
        END TYPE spherical_polar_t




        ! ==========================================
        ! Public Interfaces
        ! ==========================================
        ! (x,y,z)           -> equal-angle (x,y)
        ! (lat,lon)         ->  (x,y,z)
        ! equal-angle (x,y) ->  (lat,lon)
        ! should be called cubedsphere2spherical
        ! equal-angle (x,y) ->  (x,y,z)
        ! (lat,lon)         ->  equal-angle (x,y)
        ! CE
        !  (x,y,z)          -> gnomonic (x,y)
        !  gnominic (x,y)   -> (lat,lon)
        !private :: spherical_to_cart

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_cartesian2d_t
            MODULE PROCEDURE kgen_read_cartesian3d_t
            MODULE PROCEDURE kgen_read_spherical_polar_t
        END INTERFACE kgen_read

        PUBLIC kgen_verify
        INTERFACE kgen_verify
            MODULE PROCEDURE kgen_verify_cartesian2d_t
            MODULE PROCEDURE kgen_verify_cartesian3d_t
            MODULE PROCEDURE kgen_verify_spherical_polar_t
        END INTERFACE kgen_verify

        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        SUBROUTINE kgen_read_cartesian2d_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(cartesian2d_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%x
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%x **", var%x
            END IF
            READ(UNIT=kgen_unit) var%y
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%y **", var%y
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_read_cartesian3d_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(cartesian3d_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%x
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%x **", var%x
            END IF
            READ(UNIT=kgen_unit) var%y
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%y **", var%y
            END IF
            READ(UNIT=kgen_unit) var%z
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%z **", var%z
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_read_spherical_polar_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(spherical_polar_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%r
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%r **", var%r
            END IF
            READ(UNIT=kgen_unit) var%lon
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%lon **", var%lon
            END IF
            READ(UNIT=kgen_unit) var%lat
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%lat **", var%lat
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_cartesian2d_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(cartesian2d_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_real_real_kind("x", dtype_check_status, var%x, ref_var%x)
            CALL kgen_verify_real_real_kind("y", dtype_check_status, var%y, ref_var%y)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_cartesian3d_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(cartesian3d_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_real_real_kind("x", dtype_check_status, var%x, ref_var%x)
            CALL kgen_verify_real_real_kind("y", dtype_check_status, var%y, ref_var%y)
            CALL kgen_verify_real_real_kind("z", dtype_check_status, var%z, ref_var%z)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_spherical_polar_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(spherical_polar_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_real_real_kind("r", dtype_check_status, var%r, ref_var%r)
            CALL kgen_verify_real_real_kind("lon", dtype_check_status, var%lon, ref_var%lon)
            CALL kgen_verify_real_real_kind("lat", dtype_check_status, var%lat, ref_var%lat)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
            SUBROUTINE kgen_verify_real_real_kind( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in) :: var, ref_var
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
            END SUBROUTINE kgen_verify_real_real_kind

        ! ============================================
        ! copy_cart2d:
        !
        ! Overload assignment operator for cartesian2D_t
        ! ============================================

        ! ============================================
        ! eq_cart2d:
        !
        ! Overload == operator for cartesian2D_t
        ! ============================================

        ! ===================================================
        ! distance_cart2D  : scalar version
        ! distance_cart2D_v: vector version
        !
        ! computes distance between cartesian 2D coordinates
        ! ===================================================


        ! ===================================================
        ! distance_cart3D  : scalar version
        ! distance_cart3D_v: vector version
        ! ===================================================


        ! ===================================================================
        ! spherical_to_cart:
        ! converts spherical polar {lon,lat}  to 3D cartesian {x,y,z}
        ! on unit sphere.  Note: spherical longitude is [0,2\pi]
        ! ===================================================================

        ! ===================================================================
        ! spherical_to_cart_v:
        ! converts spherical polar {lon,lat}  to 3D cartesian {x,y,z}
        ! on unit sphere.  Note: spherical longitude is [0,2\pi]
        ! ===================================================================

        ! ==========================================================================
        ! cart_to_spherical:
        !
        ! converts 3D cartesian {x,y,z} to spherical polar {lon,lat}
        ! on unit sphere. Note: spherical longitude is [0,2\pi]
        ! ==========================================================================
        ! scalar version





        ! Note: Output spherical longitude is [-pi,pi]

        ! takes a 2D point on a face of the cube of size [-\pi/4, \pi/4] and projects it
        ! onto a 3D point on a cube of size [-1,1] in R^3

        ! onto a cube of size [-\pi/2,\pi/2] in R^3
        ! the spherical longitude can be either in [0,2\pi] or [-\pi,\pi]

        ! Go from an arbitrary sized cube in 3D
        ! to a [-\pi/4,\pi/4] sized cube with (face,2d) coordinates.
        !
        !                        Z
        !                        |
        !                        |
        !                        |
        !                        |
        !                        ---------------Y
        !                       /
        !                      /
        !                     /
        !                    /
        !                   X
        !
        ! NOTE: Face 1 =>  X positive constant face of cube
        !       Face 2 =>  Y positive constant face of cube
        !       Face 3 =>  X negative constant face of cube
        !       Face 4 =>  Y negative constant face of cube
        !       Face 5 =>  Z negative constant face of cube
        !       Face 6 =>  Z positive constant face of cube

        ! This function divides three dimentional space up into
        ! six sectors.  These sectors are then considered as the
        ! faces of the cube.  It should work for any (x,y,z) coordinate
        ! if on a sphere or on a cube.

        ! This could be done directly by using the lon, lat coordinates,
        ! but call cube_face_number_from_cart just so that there is one place
        ! to do the conversions and they are all consistant.

        ! CE, need real (cartesian) xy coordinates on the cubed sphere

        ! CE END

        !CE, 5.May 2011
        !INPUT: Points in xy cubed sphere coordinates, counterclockwise
        !OUTPUT: corresponding area on the sphere

    END MODULE coordinate_systems_mod
