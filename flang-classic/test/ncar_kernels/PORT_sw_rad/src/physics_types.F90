
! KGEN-generated Fortran source file
!
! Filename    : physics_types.F90
! Generated at: 2015-07-07 00:48:23
! KGEN version: 0.4.13



    MODULE physics_types
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        USE ppgrid, ONLY: pcols
        USE ppgrid, ONLY: psubcols
        IMPLICIT NONE
        PRIVATE ! Make default type private to the module
        ! Public types:
        PUBLIC physics_state
        ! Public interfaces
        ! Check state object for invalid data.
        ! adjust dry mass and energy for change in water
        ! cannot be applied to eul or sld dycores
        ! copy a physics_state object
        ! copy a physics_ptend object
        ! accumulate physics_ptend objects
        ! initialize a physics_tend object
        ! calculate dry air masses in state variable
        ! allocate individual components within state
        ! allocate components set by dycore
        ! deallocate individual components within state
        ! allocate individual components within tend
        ! deallocate individual components within tend
        ! allocate individual components within tend
        ! deallocate individual components within tend
        !-------------------------------------------------------------------------------
        TYPE physics_state
            INTEGER :: lchnk, ngrdcol, nsubcol(pcols), psetcols=0, ncol=0, indcol(pcols*psubcols)
            ! chunk index
            ! -- Grid        -- number of active columns (on the grid)
            ! -- Sub-columns -- number of active sub-columns in each grid column
            ! --             -- max number of columns set - if subcols = pcols*psubcols, else = pcols
            ! --             -- sum of nsubcol for all ngrdcols - number of active columns
            ! --             -- indices for mapping from subcols to grid cols
            REAL(KIND=r8), dimension(:), allocatable :: lat, lon, ps, psdry, phis, ulat, ulon
            ! latitude (radians)
            ! longitude (radians)
            ! surface pressure
            ! dry surface pressure
            ! surface geopotential
            ! unique latitudes  (radians)
            ! unique longitudes (radians)
            REAL(KIND=r8), dimension(:,:), allocatable :: t, u, v, s, omega, pmid, pmiddry, pdel, pdeldry, rpdel, rpdeldry, &
            lnpmid, lnpmiddry, exner, zm
            ! temperature (K)
            ! zonal wind (m/s)
            ! meridional wind (m/s)
            ! dry static energy
            ! vertical pressure velocity (Pa/s)
            ! midpoint pressure (Pa)
            ! midpoint pressure dry (Pa)
            ! layer thickness (Pa)
            ! layer thickness dry (Pa)
            ! reciprocal of layer thickness (Pa)
            ! recipricol layer thickness dry (Pa)
            ! ln(pmid)
            ! log midpoint pressure dry (Pa)
            ! inverse exner function w.r.t. surface pressure (ps/p)^(R/cp)
            ! geopotential height above surface at midpoints (m)
            REAL(KIND=r8), dimension(:,:,:), allocatable :: q
            ! constituent mixing ratio (kg/kg moist or dry air depending on type)
            REAL(KIND=r8), dimension(:,:), allocatable :: pint, pintdry, lnpint, lnpintdry, zi
            ! interface pressure (Pa)
            ! interface pressure dry (Pa)
            ! ln(pint)
            ! log interface pressure dry (Pa)
            ! geopotential height above surface at interfaces (m)
            REAL(KIND=r8), dimension(:), allocatable :: te_ini, te_cur, tw_ini, tw_cur
            ! vertically integrated total (kinetic + static) energy of initial state
            ! vertically integrated total (kinetic + static) energy of current state
            ! vertically integrated total water of initial state
            ! vertically integrated total water of new state
            INTEGER :: count ! count of values with significant energy or water imbalances
            INTEGER, dimension(:), allocatable :: latmapback, lonmapback, cid
            ! map from column to unique lat for that column
            ! map from column to unique lon for that column
            ! unique column id
            INTEGER :: ulatcnt, uloncnt ! number of unique lats in chunk
            ! number of unique lons in chunk
            ! Whether allocation from dycore has happened.
            LOGICAL :: dycore_alloc = .false.
            ! WACCM variables set by dycore
            REAL(KIND=r8), dimension(:,:), allocatable :: uzm, frontgf, frontga
            ! zonal wind for qbo (m/s)
            ! frontogenesis function
            ! frontogenesis angle
        END TYPE physics_state
        !-------------------------------------------------------------------------------
        !-------------------------------------------------------------------------------
        ! This is for tendencies returned from individual parameterizations
        !===============================================================================

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_physics_state
        END INTERFACE kgen_read

        PUBLIC kgen_verify
        INTERFACE kgen_verify
            MODULE PROCEDURE kgen_verify_physics_state
        END INTERFACE kgen_verify

        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim1_alloc(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: var
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
            END SUBROUTINE kgen_read_real_r8_dim1_alloc

            SUBROUTINE kgen_read_real_r8_dim2_alloc(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: var
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
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim2_alloc

            SUBROUTINE kgen_read_integer_4_dim1_alloc(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                integer(KIND=4), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: var
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
            END SUBROUTINE kgen_read_integer_4_dim1_alloc

            SUBROUTINE kgen_read_real_r8_dim3_alloc(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:,:) :: var
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
            END SUBROUTINE kgen_read_real_r8_dim3_alloc

        ! No module extern variables
        SUBROUTINE kgen_read_physics_state(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(physics_state), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%lchnk
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%lchnk **", var%lchnk
            END IF
            READ(UNIT=kgen_unit) var%ngrdcol
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ngrdcol **", var%ngrdcol
            END IF
            READ(UNIT=kgen_unit) var%nsubcol
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%nsubcol **", var%nsubcol
            END IF
            READ(UNIT=kgen_unit) var%psetcols
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%psetcols **", var%psetcols
            END IF
            READ(UNIT=kgen_unit) var%ncol
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ncol **", var%ncol
            END IF
            READ(UNIT=kgen_unit) var%indcol
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%indcol **", var%indcol
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%lat, kgen_unit, printvar=printvar//"%lat")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%lat, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%lon, kgen_unit, printvar=printvar//"%lon")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%lon, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%ps, kgen_unit, printvar=printvar//"%ps")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%ps, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%psdry, kgen_unit, printvar=printvar//"%psdry")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%psdry, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%phis, kgen_unit, printvar=printvar//"%phis")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%phis, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%ulat, kgen_unit, printvar=printvar//"%ulat")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%ulat, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%ulon, kgen_unit, printvar=printvar//"%ulon")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%ulon, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%t, kgen_unit, printvar=printvar//"%t")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%t, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%u, kgen_unit, printvar=printvar//"%u")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%u, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%v, kgen_unit, printvar=printvar//"%v")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%v, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%s, kgen_unit, printvar=printvar//"%s")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%s, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%omega, kgen_unit, printvar=printvar//"%omega")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%omega, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%pmid, kgen_unit, printvar=printvar//"%pmid")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%pmid, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%pmiddry, kgen_unit, printvar=printvar//"%pmiddry")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%pmiddry, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%pdel, kgen_unit, printvar=printvar//"%pdel")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%pdel, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%pdeldry, kgen_unit, printvar=printvar//"%pdeldry")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%pdeldry, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%rpdel, kgen_unit, printvar=printvar//"%rpdel")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%rpdel, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%rpdeldry, kgen_unit, printvar=printvar//"%rpdeldry")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%rpdeldry, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%lnpmid, kgen_unit, printvar=printvar//"%lnpmid")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%lnpmid, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%lnpmiddry, kgen_unit, printvar=printvar//"%lnpmiddry")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%lnpmiddry, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%exner, kgen_unit, printvar=printvar//"%exner")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%exner, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%zm, kgen_unit, printvar=printvar//"%zm")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%zm, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim3_alloc(var%q, kgen_unit, printvar=printvar//"%q")
            ELSE
                CALL kgen_read_real_r8_dim3_alloc(var%q, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%pint, kgen_unit, printvar=printvar//"%pint")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%pint, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%pintdry, kgen_unit, printvar=printvar//"%pintdry")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%pintdry, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%lnpint, kgen_unit, printvar=printvar//"%lnpint")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%lnpint, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%lnpintdry, kgen_unit, printvar=printvar//"%lnpintdry")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%lnpintdry, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%zi, kgen_unit, printvar=printvar//"%zi")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%zi, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%te_ini, kgen_unit, printvar=printvar//"%te_ini")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%te_ini, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%te_cur, kgen_unit, printvar=printvar//"%te_cur")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%te_cur, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%tw_ini, kgen_unit, printvar=printvar//"%tw_ini")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%tw_ini, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim1_alloc(var%tw_cur, kgen_unit, printvar=printvar//"%tw_cur")
            ELSE
                CALL kgen_read_real_r8_dim1_alloc(var%tw_cur, kgen_unit)
            END IF
            READ(UNIT=kgen_unit) var%count
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%count **", var%count
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_4_dim1_alloc(var%latmapback, kgen_unit, printvar=printvar//"%latmapback")
            ELSE
                CALL kgen_read_integer_4_dim1_alloc(var%latmapback, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_4_dim1_alloc(var%lonmapback, kgen_unit, printvar=printvar//"%lonmapback")
            ELSE
                CALL kgen_read_integer_4_dim1_alloc(var%lonmapback, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_4_dim1_alloc(var%cid, kgen_unit, printvar=printvar//"%cid")
            ELSE
                CALL kgen_read_integer_4_dim1_alloc(var%cid, kgen_unit)
            END IF
            READ(UNIT=kgen_unit) var%ulatcnt
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ulatcnt **", var%ulatcnt
            END IF
            READ(UNIT=kgen_unit) var%uloncnt
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%uloncnt **", var%uloncnt
            END IF
            READ(UNIT=kgen_unit) var%dycore_alloc
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dycore_alloc **", var%dycore_alloc
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%uzm, kgen_unit, printvar=printvar//"%uzm")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%uzm, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%frontgf, kgen_unit, printvar=printvar//"%frontgf")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%frontgf, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%frontga, kgen_unit, printvar=printvar//"%frontga")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%frontga, kgen_unit)
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_physics_state(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(physics_state), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_integer("lchnk", dtype_check_status, var%lchnk, ref_var%lchnk)
            CALL kgen_verify_integer("ngrdcol", dtype_check_status, var%ngrdcol, ref_var%ngrdcol)
            CALL kgen_verify_integer_4_dim1("nsubcol", dtype_check_status, var%nsubcol, ref_var%nsubcol)
            CALL kgen_verify_integer("psetcols", dtype_check_status, var%psetcols, ref_var%psetcols)
            CALL kgen_verify_integer("ncol", dtype_check_status, var%ncol, ref_var%ncol)
            CALL kgen_verify_integer_4_dim1("indcol", dtype_check_status, var%indcol, ref_var%indcol)
            CALL kgen_verify_real_r8_dim1_alloc("lat", dtype_check_status, var%lat, ref_var%lat)
            CALL kgen_verify_real_r8_dim1_alloc("lon", dtype_check_status, var%lon, ref_var%lon)
            CALL kgen_verify_real_r8_dim1_alloc("ps", dtype_check_status, var%ps, ref_var%ps)
            CALL kgen_verify_real_r8_dim1_alloc("psdry", dtype_check_status, var%psdry, ref_var%psdry)
            CALL kgen_verify_real_r8_dim1_alloc("phis", dtype_check_status, var%phis, ref_var%phis)
            CALL kgen_verify_real_r8_dim1_alloc("ulat", dtype_check_status, var%ulat, ref_var%ulat)
            CALL kgen_verify_real_r8_dim1_alloc("ulon", dtype_check_status, var%ulon, ref_var%ulon)
            CALL kgen_verify_real_r8_dim2_alloc("t", dtype_check_status, var%t, ref_var%t)
            CALL kgen_verify_real_r8_dim2_alloc("u", dtype_check_status, var%u, ref_var%u)
            CALL kgen_verify_real_r8_dim2_alloc("v", dtype_check_status, var%v, ref_var%v)
            CALL kgen_verify_real_r8_dim2_alloc("s", dtype_check_status, var%s, ref_var%s)
            CALL kgen_verify_real_r8_dim2_alloc("omega", dtype_check_status, var%omega, ref_var%omega)
            CALL kgen_verify_real_r8_dim2_alloc("pmid", dtype_check_status, var%pmid, ref_var%pmid)
            CALL kgen_verify_real_r8_dim2_alloc("pmiddry", dtype_check_status, var%pmiddry, ref_var%pmiddry)
            CALL kgen_verify_real_r8_dim2_alloc("pdel", dtype_check_status, var%pdel, ref_var%pdel)
            CALL kgen_verify_real_r8_dim2_alloc("pdeldry", dtype_check_status, var%pdeldry, ref_var%pdeldry)
            CALL kgen_verify_real_r8_dim2_alloc("rpdel", dtype_check_status, var%rpdel, ref_var%rpdel)
            CALL kgen_verify_real_r8_dim2_alloc("rpdeldry", dtype_check_status, var%rpdeldry, ref_var%rpdeldry)
            CALL kgen_verify_real_r8_dim2_alloc("lnpmid", dtype_check_status, var%lnpmid, ref_var%lnpmid)
            CALL kgen_verify_real_r8_dim2_alloc("lnpmiddry", dtype_check_status, var%lnpmiddry, ref_var%lnpmiddry)
            CALL kgen_verify_real_r8_dim2_alloc("exner", dtype_check_status, var%exner, ref_var%exner)
            CALL kgen_verify_real_r8_dim2_alloc("zm", dtype_check_status, var%zm, ref_var%zm)
            CALL kgen_verify_real_r8_dim3_alloc("q", dtype_check_status, var%q, ref_var%q)
            CALL kgen_verify_real_r8_dim2_alloc("pint", dtype_check_status, var%pint, ref_var%pint)
            CALL kgen_verify_real_r8_dim2_alloc("pintdry", dtype_check_status, var%pintdry, ref_var%pintdry)
            CALL kgen_verify_real_r8_dim2_alloc("lnpint", dtype_check_status, var%lnpint, ref_var%lnpint)
            CALL kgen_verify_real_r8_dim2_alloc("lnpintdry", dtype_check_status, var%lnpintdry, ref_var%lnpintdry)
            CALL kgen_verify_real_r8_dim2_alloc("zi", dtype_check_status, var%zi, ref_var%zi)
            CALL kgen_verify_real_r8_dim1_alloc("te_ini", dtype_check_status, var%te_ini, ref_var%te_ini)
            CALL kgen_verify_real_r8_dim1_alloc("te_cur", dtype_check_status, var%te_cur, ref_var%te_cur)
            CALL kgen_verify_real_r8_dim1_alloc("tw_ini", dtype_check_status, var%tw_ini, ref_var%tw_ini)
            CALL kgen_verify_real_r8_dim1_alloc("tw_cur", dtype_check_status, var%tw_cur, ref_var%tw_cur)
            CALL kgen_verify_integer("count", dtype_check_status, var%count, ref_var%count)
            CALL kgen_verify_integer_4_dim1_alloc("latmapback", dtype_check_status, var%latmapback, ref_var%latmapback)
            CALL kgen_verify_integer_4_dim1_alloc("lonmapback", dtype_check_status, var%lonmapback, ref_var%lonmapback)
            CALL kgen_verify_integer_4_dim1_alloc("cid", dtype_check_status, var%cid, ref_var%cid)
            CALL kgen_verify_integer("ulatcnt", dtype_check_status, var%ulatcnt, ref_var%ulatcnt)
            CALL kgen_verify_integer("uloncnt", dtype_check_status, var%uloncnt, ref_var%uloncnt)
            CALL kgen_verify_logical("dycore_alloc", dtype_check_status, var%dycore_alloc, ref_var%dycore_alloc)
            CALL kgen_verify_real_r8_dim2_alloc("uzm", dtype_check_status, var%uzm, ref_var%uzm)
            CALL kgen_verify_real_r8_dim2_alloc("frontgf", dtype_check_status, var%frontgf, ref_var%frontgf)
            CALL kgen_verify_real_r8_dim2_alloc("frontga", dtype_check_status, var%frontga, ref_var%frontga)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
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

            SUBROUTINE kgen_verify_real_r8_dim1_alloc( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:), ALLOCATABLE :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:) :: temp, temp2
                integer :: n
                IF ( ALLOCATED(var) ) THEN
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
                    allocate(temp(SIZE(var,dim=1)))
                    allocate(temp2(SIZE(var,dim=1)))
                
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
            END SUBROUTINE kgen_verify_real_r8_dim1_alloc

            SUBROUTINE kgen_verify_real_r8_dim2_alloc( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:), ALLOCATABLE :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:) :: temp, temp2
                integer :: n
                IF ( ALLOCATED(var) ) THEN
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
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2)))
                
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
            END SUBROUTINE kgen_verify_real_r8_dim2_alloc

            SUBROUTINE kgen_verify_real_r8_dim3_alloc( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:,:), ALLOCATABLE :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:,:) :: temp, temp2
                integer :: n
                IF ( ALLOCATED(var) ) THEN
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
            END SUBROUTINE kgen_verify_real_r8_dim3_alloc

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

            SUBROUTINE kgen_verify_integer_4_dim1_alloc( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer, intent(in), DIMENSION(:), ALLOCATABLE :: var, ref_var
                IF ( ALLOCATED(var) ) THEN
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
            END SUBROUTINE kgen_verify_integer_4_dim1_alloc

            SUBROUTINE kgen_verify_logical( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                logical, intent(in) :: var, ref_var
                check_status%numTotal = check_status%numTotal + 1
                IF ( var .EQV. ref_var ) THEN
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
            END SUBROUTINE kgen_verify_logical

        !===============================================================================

        !===============================================================================


        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================


        !===============================================================================

        !-----------------------------------------------------------------------
        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

    END MODULE physics_types
