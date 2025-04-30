
! KGEN-generated Fortran source file
!
! Filename    : edge_mod.F90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9



    MODULE edge_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE coordinate_systems_mod, ONLY : kgen_read_mod6 => kgen_read
    USE coordinate_systems_mod, ONLY : kgen_verify_mod6 => kgen_verify
        USE kinds, ONLY: int_kind
        USE kinds, ONLY: log_kind
        USE kinds, ONLY: real_kind
        ! _EXTERNAL
        USE coordinate_systems_mod, ONLY: cartesian3d_t
        IMPLICIT NONE
        PRIVATE
        TYPE, public :: rotation_t
            INTEGER :: nbr ! nbr direction: north south east west
            INTEGER :: reverse ! 0 = do not reverse order
            ! 1 = reverse order
            REAL(KIND=real_kind), dimension(:,:,:), pointer :: r => null() ! rotation matrix
        END TYPE rotation_t
        TYPE, public :: edgedescriptor_t
            INTEGER(KIND=int_kind) :: use_rotation
            INTEGER(KIND=int_kind) :: padding
            INTEGER(KIND=int_kind), pointer :: putmapp(:) => null()
            INTEGER(KIND=int_kind), pointer :: getmapp(:) => null()
            INTEGER(KIND=int_kind), pointer :: putmapp_ghost(:) => null()
            INTEGER(KIND=int_kind), pointer :: getmapp_ghost(:) => null()
            INTEGER(KIND=int_kind), pointer :: globalid(:) => null()
            INTEGER(KIND=int_kind), pointer :: loc2buf(:) => null()
            TYPE(cartesian3d_t), pointer :: neigh_corners(:,:) => null()
            INTEGER :: actual_neigh_edges
            LOGICAL(KIND=log_kind), pointer :: reverse(:) => null()
            TYPE(rotation_t), dimension(:), pointer :: rot => null() ! Identifies list of edges
            !  that must be rotated, and how
        END TYPE edgedescriptor_t
        ! NOTE ON ELEMENT ORIENTATION
        !
        ! Element orientation:  index V(i,j)
        !
        !           (1,np) NWEST      (np,np) NEAST
        !
        !           (1,1) SWEST       (np,1) SEAST
        !
        !
        ! for the edge neighbors:
        !    we set the "reverse" flag if two elements who share an edge use a
        !    reverse orientation.  The data is reversed during the *pack* stage
        ! For corner neighbors:
        !    for edge buffers, there is no orientation because two corner neighbors
        !    only share a single point.
        !    For ghost cell data, there is a again two posible orientations. For
        !    this case, we set the "reverse" flag if the corner element is using
        !    the reverse orientation.  In this case, the data is reversed during the
        !    *unpack* stage (not sure why)
        !
        ! The edge orientation is set at startup.  The corner orientation is computed
        ! at run time, via the call to compute_ghost_corner_orientation()
        ! This routine only works for meshes with at most 1 corner element.  It's
        ! not called and the corner orientation flag is not set for unstructured meshes
        !
        !
        ! Mark Taylor
        ! pack/unpack full element of data of size (nx,nx)
        ! user specifies the size when creating the buffer
        ! input/output arrays are cartesian, and will only unpack 1 corner element
        ! (even if there are more when running with an unstructured grid)
        ! This routine is used mostly for testing and to compute the orientation of
        ! an elements corner neighbors
        !
        ! init/free buffers used by pack/unpack full and 3D
        ! same as above, except orientation of element data is preserved
        ! (so boundary data for two adjacent element may not match up)
        !
        ! James Overfelt
        ! pack/unpack user specifed halo region "nhc".
        ! Does not include element edge data (assumes element edge data is C0)
        ! (appropriate for continuous GLL data where the edge data does not need to be sent)
        ! support for unstructed meshes via extra output arrays: sw,se,ne,nw
        ! This routine is currently used by surfaces_mod.F90 to construct the GLL dual grid
        !
        ! pack/unpack specifed halo size (up to 1 element)
        ! should be identical to ghostVpack2d except for
        ! shape of input array
        ! returns v including populating halo region of v
        ! "extra" corner elements are returned in arrays
        ! sw,se,ne,nw
        ! MT TODO: this routine works for unstructed data (where the corner orientation flag is
        ! not set).  So why dont we remove all the "reverse" checks in unpack?
        !
        ! Christoph Erath
        ! pack/unpack partial element of data of size (nx,nx) with user specifed halo size nh
        ! user specifies the sizes when creating the buffer
        ! buffer has 1 extra dimension (as compared to subroutines above) for multiple tracers
        ! input/output arrays are cartesian, and thus assume at most 1 element at each corner
        ! hence currently only supports cube-sphere grids.
        !
        ! TODO: GhostBufferTR (init and type) should be removed - we only need GhostBuffer3D,
        ! if we can fix
        ! ghostVpack2d below to pass vlyr*ntrac_d instead of two seperate arguments
        !
        ! ghostbufferTR_t
        ! ghostbufferTR_t
        ! routines which including element edge data
        ! (used for FVM arrays where edge data is not shared by neighboring elements)
        ! these routines pack/unpack element data with user specified halo size
        !
        ! THESE ROUTINES SHOULD BE MERGED
        !
        ! input/output:
        ! v(1-nhc:npoints+nhc,1-nhc:npoints+nhc,vlyr,ntrac_d,timelevels)
        ! used to pack/unpack SPELT "Rp".  What's this?
        ! v(1-nhc:npoints+nhc,1-nhc:npoints+nhc,vlyr,ntrac_d)
        ! routines which do NOT include element edge data
        ! (used for SPELT arrays and GLL point arrays, where edge data is shared and does not need
        ! to be sent/received.
        ! these routines pack/unpack element data with user specifed halo size
        !
        ! THESE ROUTINES CAN ALL BE REPLACED BY ghostVpack3D (if we make extra corner data arrays
        ! an optional argument).  Or at least these should be merged to 1 routine
        ! input/output:
        ! v(1-nhc:npoints+nhc,1-nhc:npoints+nhc, vlyr, ntrac_d,timelevels)
        ! used to pack/unpack SPELT%sga.  what's this?
        ! input/output
        !   v(1-nhc:npoints+nhc,1-nhc:npoints+nhc)
        ! used to pack/unpack FV vertex data (velocity/grid)
        ! input/output
        ! v(1-nhc:npoints+nhc,1-nhc:npoints+nhc, vlyr)
        ! Wrap pointer so we can make an array of them.

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_rotation_t
            MODULE PROCEDURE kgen_read_edgedescriptor_t
        END INTERFACE kgen_read

        PUBLIC kgen_verify
        INTERFACE kgen_verify
            MODULE PROCEDURE kgen_verify_rotation_t
            MODULE PROCEDURE kgen_verify_edgedescriptor_t
        END INTERFACE kgen_verify

        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_real_kind_dim3_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=real_kind), INTENT(OUT), POINTER, DIMENSION(:,:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2,idx3
                INTEGER, DIMENSION(2,3) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    READ(UNIT = kgen_unit) kgen_bound(1, 3)
                    READ(UNIT = kgen_unit) kgen_bound(2, 3)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_real_kind_dim3_ptr

            SUBROUTINE kgen_read_integer_int_kind_dim1_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                integer(KIND=int_kind), INTENT(OUT), POINTER, DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_integer_int_kind_dim1_ptr

            SUBROUTINE kgen_read_logical_log_kind_dim1_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                logical(KIND=log_kind), INTENT(OUT), POINTER, DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_logical_log_kind_dim1_ptr

            SUBROUTINE kgen_read_cartesian3d_t_dim2_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                TYPE(cartesian3d_t), INTENT(OUT), POINTER, DIMENSION(:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2
                INTEGER, DIMENSION(2,2) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                    DO idx1=kgen_bound(1,1), kgen_bound(2, 1)
                        DO idx2=kgen_bound(1,2), kgen_bound(2, 2)
                    IF ( PRESENT(printvar) ) THEN
                                CALL kgen_read_mod6(var(idx1,idx2), kgen_unit, printvar=printvar)
                    ELSE
                                CALL kgen_read_mod6(var(idx1,idx2), kgen_unit)
                    END IF
                        END DO
                    END DO
                END IF
            END SUBROUTINE kgen_read_cartesian3d_t_dim2_ptr

            SUBROUTINE kgen_read_rotation_t_dim1_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                TYPE(rotation_t), INTENT(OUT), POINTER, DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                    DO idx1=kgen_bound(1,1), kgen_bound(2, 1)
                    IF ( PRESENT(printvar) ) THEN
                            CALL kgen_read_rotation_t(var(idx1), kgen_unit, printvar=printvar)
                    ELSE
                            CALL kgen_read_rotation_t(var(idx1), kgen_unit)
                    END IF
                    END DO
                END IF
            END SUBROUTINE kgen_read_rotation_t_dim1_ptr

        ! No module extern variables
        SUBROUTINE kgen_read_rotation_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(rotation_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%nbr
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%nbr **", var%nbr
            END IF
            READ(UNIT=kgen_unit) var%reverse
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%reverse **", var%reverse
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_real_kind_dim3_ptr(var%r, kgen_unit, printvar=printvar//"%r")
            ELSE
                CALL kgen_read_real_real_kind_dim3_ptr(var%r, kgen_unit)
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_read_edgedescriptor_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(edgedescriptor_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%use_rotation
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%use_rotation **", var%use_rotation
            END IF
            READ(UNIT=kgen_unit) var%padding
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%padding **", var%padding
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_int_kind_dim1_ptr(var%putmapp, kgen_unit, printvar=printvar//"%putmapp")
            ELSE
                CALL kgen_read_integer_int_kind_dim1_ptr(var%putmapp, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_int_kind_dim1_ptr(var%getmapp, kgen_unit, printvar=printvar//"%getmapp")
            ELSE
                CALL kgen_read_integer_int_kind_dim1_ptr(var%getmapp, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_int_kind_dim1_ptr(var%putmapp_ghost, kgen_unit, printvar=printvar//"%putmapp_ghost")
            ELSE
                CALL kgen_read_integer_int_kind_dim1_ptr(var%putmapp_ghost, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_int_kind_dim1_ptr(var%getmapp_ghost, kgen_unit, printvar=printvar//"%getmapp_ghost")
            ELSE
                CALL kgen_read_integer_int_kind_dim1_ptr(var%getmapp_ghost, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_int_kind_dim1_ptr(var%globalid, kgen_unit, printvar=printvar//"%globalid")
            ELSE
                CALL kgen_read_integer_int_kind_dim1_ptr(var%globalid, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_int_kind_dim1_ptr(var%loc2buf, kgen_unit, printvar=printvar//"%loc2buf")
            ELSE
                CALL kgen_read_integer_int_kind_dim1_ptr(var%loc2buf, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_cartesian3d_t_dim2_ptr(var%neigh_corners, kgen_unit, printvar=printvar//"%neigh_corners")
            ELSE
                CALL kgen_read_cartesian3d_t_dim2_ptr(var%neigh_corners, kgen_unit)
            END IF
            READ(UNIT=kgen_unit) var%actual_neigh_edges
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%actual_neigh_edges **", var%actual_neigh_edges
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_logical_log_kind_dim1_ptr(var%reverse, kgen_unit, printvar=printvar//"%reverse")
            ELSE
                CALL kgen_read_logical_log_kind_dim1_ptr(var%reverse, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_rotation_t_dim1_ptr(var%rot, kgen_unit, printvar=printvar//"%rot")
            ELSE
                CALL kgen_read_rotation_t_dim1_ptr(var%rot, kgen_unit)
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_rotation_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(rotation_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_integer("nbr", dtype_check_status, var%nbr, ref_var%nbr)
            CALL kgen_verify_integer("reverse", dtype_check_status, var%reverse, ref_var%reverse)
            CALL kgen_verify_real_real_kind_dim3_ptr("r", dtype_check_status, var%r, ref_var%r)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_edgedescriptor_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(edgedescriptor_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_integer_int_kind("use_rotation", dtype_check_status, var%use_rotation, ref_var%use_rotation)
            CALL kgen_verify_integer_int_kind("padding", dtype_check_status, var%padding, ref_var%padding)
            CALL kgen_verify_integer_int_kind_dim1_ptr("putmapp", dtype_check_status, var%putmapp, ref_var%putmapp)
            CALL kgen_verify_integer_int_kind_dim1_ptr("getmapp", dtype_check_status, var%getmapp, ref_var%getmapp)
            CALL kgen_verify_integer_int_kind_dim1_ptr("putmapp_ghost", dtype_check_status, var%putmapp_ghost, ref_var%putmapp_ghost)
            CALL kgen_verify_integer_int_kind_dim1_ptr("getmapp_ghost", dtype_check_status, var%getmapp_ghost, ref_var%getmapp_ghost)
            CALL kgen_verify_integer_int_kind_dim1_ptr("globalid", dtype_check_status, var%globalid, ref_var%globalid)
            CALL kgen_verify_integer_int_kind_dim1_ptr("loc2buf", dtype_check_status, var%loc2buf, ref_var%loc2buf)
            CALL kgen_verify_cartesian3d_t_dim2_ptr("neigh_corners", dtype_check_status, var%neigh_corners, ref_var%neigh_corners)
            CALL kgen_verify_integer("actual_neigh_edges", dtype_check_status, var%actual_neigh_edges, ref_var%actual_neigh_edges)
            CALL kgen_verify_logical_log_kind_dim1_ptr("reverse", dtype_check_status, var%reverse, ref_var%reverse)
            CALL kgen_verify_rotation_t_dim1_ptr("rot", dtype_check_status, var%rot, ref_var%rot)
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

            SUBROUTINE kgen_verify_real_real_kind_dim3_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in), DIMENSION(:,:,:), POINTER :: var, ref_var
                real(KIND=real_kind) :: nrmsdiff, rmsdiff
                real(KIND=real_kind), allocatable, DIMENSION(:,:,:) :: temp, temp2
                integer :: n
                IF ( ASSOCIATED(var) ) THEN
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
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                
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
                END IF
            END SUBROUTINE kgen_verify_real_real_kind_dim3_ptr

            SUBROUTINE kgen_verify_integer_int_kind( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer(KIND=int_kind), intent(in) :: var, ref_var
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
            END SUBROUTINE kgen_verify_integer_int_kind

            SUBROUTINE kgen_verify_integer_int_kind_dim1_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer(KIND=int_kind), intent(in), DIMENSION(:), POINTER :: var, ref_var
                IF ( ASSOCIATED(var) ) THEN
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
                END IF
            END SUBROUTINE kgen_verify_integer_int_kind_dim1_ptr

            SUBROUTINE kgen_verify_cartesian3d_t_dim2_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                type(check_t) :: dtype_check_status
                TYPE(cartesian3d_t), intent(in), DIMENSION(:,:), POINTER :: var, ref_var
                integer :: idx1,idx2
                IF ( ASSOCIATED(var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                CALL kgen_init_check(dtype_check_status)
                DO idx1=LBOUND(var,1), UBOUND(var,1)
                    DO idx2=LBOUND(var,2), UBOUND(var,2)
                        CALL kgen_verify_mod6(varname, dtype_check_status, var(idx1,idx2), ref_var(idx1,idx2))
                    END DO
                END DO
                IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1
                ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                    check_status%numFatal = check_status%numFatal + 1
                ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                    check_status%numWarning = check_status%numWarning + 1
                END IF
                END IF
            END SUBROUTINE kgen_verify_cartesian3d_t_dim2_ptr

            SUBROUTINE kgen_verify_logical_log_kind_dim1_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                logical(KIND=log_kind), intent(in), DIMENSION(:), POINTER :: var, ref_var
                IF ( ASSOCIATED(var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                IF ( ALL( var .EQV. ref_var ) ) THEN
                
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
                END IF
            END SUBROUTINE kgen_verify_logical_log_kind_dim1_ptr

            SUBROUTINE kgen_verify_rotation_t_dim1_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                type(check_t) :: dtype_check_status
                TYPE(rotation_t), intent(in), DIMENSION(:), POINTER :: var, ref_var
                integer :: idx1
                IF ( ASSOCIATED(var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                CALL kgen_init_check(dtype_check_status)
                DO idx1=LBOUND(var,1), UBOUND(var,1)
                    CALL kgen_verify_rotation_t("rotation_t", dtype_check_status, var(idx1), ref_var(idx1))
                END DO
                IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1
                ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                    check_status%numFatal = check_status%numFatal + 1
                ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                    check_status%numWarning = check_status%numWarning + 1
                END IF
                END IF
            END SUBROUTINE kgen_verify_rotation_t_dim1_ptr

        ! =========================================
        ! initEdgeBuffer:
        !
        ! create an Real based communication buffer
        ! =========================================

        ! =========================================
        ! initLongEdgeBuffer:
        !
        ! create an Integer based communication buffer
        ! =========================================

        ! =========================================
        ! edgeDGVpack:
        !
        ! Pack edges of v into buf for DG stencil
        ! =========================================

        ! ===========================================
        !  FreeEdgeBuffer:
        !
        !  Freed an edge communication buffer
        ! =========================================


        ! ===========================================
        !  FreeLongEdgeBuffer:
        !
        !  Freed an edge communication buffer
        ! =========================================

        ! =========================================
        !
        !> @brief Pack edges of v into an edge buffer for boundary exchange.
        !
        !> This subroutine packs for one or more vertical layers into an edge
        !! buffer. If the buffer associated with edge is not large enough to
        !! hold all vertical layers you intent to pack, the method will
        !! halt the program with a call to parallel_mod::haltmp().
        !! @param[in] edge Edge Buffer into which the data will be packed.
        !! This buffer must be previously allocated with initEdgeBuffer().
        !! @param[in] v The data to be packed.
        !! @param[in] vlyr Number of vertical level coming into the subroutine
        !! for packing for input v.
        !! @param[in] kptr Vertical pointer to the place in the edge buffer where
        !! data will be located.
        ! =========================================

        ! =========================================
        ! LongEdgeVpack:
        !
        ! Pack edges of v into buf...
        ! =========================================

        ! ========================================
        ! edgeVunpack:
        !
        ! Unpack edges from edge buffer into v...
        ! ========================================


        ! ========================================
        ! edgeDGVunpack:
        !
        ! Unpack edges from edge buffer into v...
        ! ========================================

        ! ========================================
        ! edgeVunpackMIN/MAX:
        !
        ! Finds the Min/Max edges from edge buffer into v...
        ! ========================================


        ! ========================================
        ! LongEdgeVunpackMIN:
        !
        ! Finds the Min edges from edge buffer into v...
        ! ========================================

        ! =============================
        ! edgerotate:
        !
        ! Rotate edges in buffer...
        ! =============================

        ! =============================================
        ! buffermap:
        !
        ! buffermap translates element number, inum and
        ! element edge/corner, facet, into an edge buffer
        ! memory location, loc.
        ! =============================================

        ! ===========================================
        !  FreeGhostBuffer:
        !  Author: Christoph Erath, Mark Taylor
        !  Freed an ghostpoints communication buffer
        ! =========================================

        ! =========================================
        ! =========================================
        !
        !> @brief Pack edges of v into an edge buffer for boundary exchange.
        !
        !> This subroutine packs for one or more vertical layers into an edge
        !! buffer. If the buffer associated with edge is not large enough to
        !! hold all vertical layers you intent to pack, the method will
        !! halt the program with a call to parallel_mod::haltmp().
        !! @param[in] edge Ghost Buffer into which the data will be packed.
        !! This buffer must be previously allocated with initghostbufferfull().
        !! @param[in] v The data to be packed.
        !! @param[in] vlyr Number of vertical level coming into the subroutine
        !! for packing for input v.
        !! @param[in] kptr Vertical pointer to the place in the edge buffer where
        !! data will be located.
        ! =========================================

        ! ========================================
        ! edgeVunpack:
        !
        ! Unpack edges from edge buffer into v...
        ! ========================================

        ! =========================================
        !
        !> @brief Pack edges of v into an edge buffer for boundary exchange.
        !
        !> This subroutine packs for one or more vertical layers into an edge
        !! buffer. If the buffer associated with edge is not large enough to
        !! hold all vertical layers you intent to pack, the method will
        !! halt the program with a call to parallel_mod::haltmp().
        !! @param[in] edge Ghost Buffer into which the data will be packed.
        !! This buffer must be previously allocated with initghostbuffer().
        !! @param[in] v The data to be packed.
        !! @param[in] vlyr Number of vertical level coming into the subroutine
        !! for packing for input v.
        !! @param[in] kptr Vertical pointer to the place in the edge buffer where
        !! data will be located.
        ! =========================================

        ! ========================================
        ! edgeVunpack:
        !
        ! Unpack edges from edge buffer into v...
        ! ========================================

        ! =========================================
        ! initGhostBuffer:
        ! Author: Christoph Erath
        ! create an Real based communication buffer
        ! npoints is the number of points on one side
        ! nhc is the deep of the ghost/halo zone
        ! =========================================

        ! =========================================
        ! Christoph Erath
        !> Packs the halo zone from v
        ! =========================================

        ! =========================================
        ! Christoph Erath
        !> Packs the halo zone from v
        ! =========================================
        ! NOTE: I have to give timelevels as argument, because element_mod is not compiled first
        ! and the array call has to be done in this way because of performance reasons!!!

        ! ========================================
        ! Christoph Erath
        !
        ! Unpack the halo zone into v
        ! ========================================

        ! ========================================
        ! Christoph Erath
        !
        ! Unpack the halo zone into v
        ! ========================================
        ! NOTE: I have to give timelevels as argument, because element_mod is not compiled first
        ! and the array call has to be done in this way because of performance reasons!!!

        ! =================================================================================
        ! GHOSTVPACK2D
        ! AUTHOR: Christoph Erath
        ! Pack edges of v into an ghost buffer for boundary exchange.
        !
        ! This subroutine packs for one vertical layers into an ghost
        ! buffer. It is for cartesian points (v is only two dimensional).
        ! If the buffer associated with edge is not large enough to
        ! hold all vertical layers you intent to pack, the method will
        ! halt the program with a call to parallel_mod::haltmp().
        ! INPUT:
        ! - ghost Buffer into which the data will be packed.
        !   This buffer must be previously allocated with initGhostBuffer().
        ! - v The data to be packed.
        ! - nhc deep of ghost/halo zone
        ! - npoints number of points on on side
        ! - kptr Vertical pointer to the place in the edge buffer where
        ! data will be located.
        ! =================================================================================

        ! =================================================================================
        ! GHOSTVUNPACK2D
        ! AUTHOR: Christoph Erath
        ! Unpack ghost points from ghost buffer into v...
        ! It is for cartesian points (v is only two dimensional).
        ! INPUT SAME arguments as for GHOSTVPACK2d
        ! =================================================================================

        ! =================================================================================
        ! GHOSTVPACK2D
        ! AUTHOR: Christoph Erath
        ! Pack edges of v into an ghost buffer for boundary exchange.
        !
        ! This subroutine packs for one vertical layers into an ghost
        ! buffer. It is for cartesian points (v is only two dimensional).
        ! If the buffer associated with edge is not large enough to
        ! hold all vertical layers you intent to pack, the method will
        ! halt the program with a call to parallel_mod::haltmp().
        ! INPUT:
        ! - ghost Buffer into which the data will be packed.
        !   This buffer must be previously allocated with initGhostBuffer().
        ! - v The data to be packed.
        ! - nhc deep of ghost/halo zone
        ! - npoints number of points on on side
        ! - kptr Vertical pointer to the place in the edge buffer where
        ! data will be located.
        ! =================================================================================

        ! =================================================================================
        ! GHOSTVUNPACK2D
        ! AUTHOR: Christoph Erath
        ! Unpack ghost points from ghost buffer into v...
        ! It is for cartesian points (v is only two dimensional).
        ! INPUT SAME arguments as for GHOSTVPACK2d
        ! =================================================================================

        ! =================================================================================
        ! GHOSTVPACK2D
        ! AUTHOR: Christoph Erath
        ! Pack edges of v into an ghost buffer for boundary exchange.
        !
        ! This subroutine packs for one vertical layers into an ghost
        ! buffer. It is for cartesian points (v is only two dimensional).
        ! If the buffer associated with edge is not large enough to
        ! hold all vertical layers you intent to pack, the method will
        ! halt the program with a call to parallel_mod::haltmp().
        ! INPUT:
        ! - ghost Buffer into which the data will be packed.
        !   This buffer must be previously allocated with initGhostBuffer().
        ! - v The data to be packed.
        ! - nhc deep of ghost/halo zone
        ! - npoints number of points on on side
        ! - kptr Vertical pointer to the place in the edge buffer where
        ! data will be located.
        ! =================================================================================

        ! =================================================================================
        ! GHOSTVUNPACK2D
        ! AUTHOR: Christoph Erath
        ! Unpack ghost points from ghost buffer into v...
        ! It is for cartesian points (v is only two dimensional).
        ! INPUT SAME arguments as for GHOSTVPACK2d
        ! =================================================================================

        ! =========================================
        ! initGhostBuffer3d:
        ! Author: James Overfelt
        ! create an Real based communication buffer
        ! npoints is the number of points on one side
        ! nhc is the deep of the ghost/halo zone
        ! =========================================

        ! =================================================================================
        ! GHOSTVPACK3D
        ! AUTHOR: James Overfelt (from a subroutine of Christoph Erath, ghostvpack2D)
        ! Pack edges of v into an ghost buffer for boundary exchange.
        !
        ! This subroutine packs for many vertical layers into an ghost
        ! buffer.
        ! If the buffer associated with edge is not large enough to
        ! hold all vertical layers you intent to pack, the method will
        ! halt the program with a call to parallel_mod::haltmp().
        ! INPUT:
        ! - ghost Buffer into which the data will be packed.
        !   This buffer must be previously allocated with initGhostBuffer().
        ! - v The data to be packed.
        ! - nhc deep of ghost/halo zone
        ! - npoints number of points on on side
        ! - kptr Vertical pointer to the place in the edge buffer where
        ! data will be located.
        ! =================================================================================

        ! =================================================================================
        ! GHOSTVUNPACK3D
        ! AUTHOR: James Overfelt (from a subroutine of Christoph Erath, ghostVunpack2d)
        ! Unpack ghost points from ghost buffer into v...
        ! It is for cartesian points (v is only two dimensional).
        ! INPUT SAME arguments as for GHOSTVPACK
        ! =================================================================================

    END MODULE edge_mod
