
! KGEN-generated Fortran source file
!
! Filename    : shr_spfn_mod.F90
! Generated at: 2015-03-31 09:44:41
! KGEN version: 0.4.5



    MODULE shr_spfn_mod
        ! Module for common mathematical functions
        ! This #ifdef is to allow the module to be compiled with no dependencies,
        ! even on shr_kind_mod.
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        USE shr_const_mod, ONLY: pi => shr_const_pi
        IMPLICIT NONE
        PRIVATE
        ! Error functions



        ! Gamma functions
        ! Note that we lack an implementation of log_gamma, but we do have an
        ! implementation of the upper incomplete gamma function, which is not in
        ! Fortran 2008.
        ! Note also that this gamma function is only for double precision. We
        ! haven't needed an r4 version yet.
        PUBLIC shr_spfn_gamma

        INTERFACE shr_spfn_gamma
            MODULE PROCEDURE shr_spfn_gamma_r8
        END INTERFACE shr_spfn_gamma
        ! Mathematical constants
        ! sqrt(pi)
        ! Define machine-specific constants needed in this module.
        ! These were used by the original gamma and calerf functions to guarantee
        ! safety against overflow, and precision, on many different machines.
        ! By defining the constants in this way, we assume that 1/xmin is
        ! representable (i.e. does not overflow the real type). This assumption was
        ! not in the original code, but is valid for IEEE single and double
        ! precision.
        ! Double precision
        !---------------------------------------------------------------------
        ! Machine epsilon
        REAL(KIND=r8), parameter :: epsr8 = epsilon(1._r8)
        ! "Huge" value is returned when actual value would be infinite.
        REAL(KIND=r8), parameter :: xinfr8 = huge(1._r8)
        ! Smallest normal value.
        REAL(KIND=r8), parameter :: xminr8 = tiny(1._r8)
        ! Largest number that, when added to 1., yields 1.
        ! Largest argument for which erfcx > 0.
        ! Single precision
        !---------------------------------------------------------------------
        ! Machine epsilon
        ! "Huge" value is returned when actual value would be infinite.
        ! Smallest normal value.
        ! Largest number that, when added to 1., yields 1.
        ! Largest argument for which erfcx > 0.
        ! For gamma/igamma
        ! Approximate value of largest acceptable argument to gamma,
        ! for IEEE double-precision.
        REAL(KIND=r8), parameter :: xbig_gamma = 171.624_r8
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        ! Wrapper functions for erf






        ! Wrapper functions for erfc






        ! Wrapper functions for erfc_scaled



        elemental FUNCTION shr_spfn_gamma_r8(x) RESULT ( res )
            REAL(KIND=r8), intent(in) :: x
            REAL(KIND=r8) :: res
            ! No intrinsic
            res = shr_spfn_gamma_nonintrinsic_r8(x)
        END FUNCTION shr_spfn_gamma_r8
        !------------------------------------------------------------------
        !
        ! 6 December 2006 -- B. Eaton
        ! The following comments are from the original version of CALERF.
        ! The only changes in implementing this module are that the function
        ! names previously used for the single precision versions have been
        ! adopted for the new generic interfaces.  To support these interfaces
        ! there is now both a single precision version (calerf_r4) and a
        ! double precision version (calerf_r8) of CALERF below.  These versions
        ! are hardcoded to use IEEE arithmetic.
        !
        !------------------------------------------------------------------
        !
        ! This packet evaluates  erf(x),  erfc(x),  and  exp(x*x)*erfc(x)
        !   for a real argument  x.  It contains three FUNCTION type
        !   subprograms: ERF, ERFC, and ERFCX (or ERF_R8, ERFC_R8, and ERFCX_R8),
        !   and one SUBROUTINE type subprogram, CALERF.  The calling
        !   statements for the primary entries are:
        !
        !                   Y=ERF(X)     (or   Y=ERF_R8(X)),
        !
        !                   Y=ERFC(X)    (or   Y=ERFC_R8(X)),
        !   and
        !                   Y=ERFCX(X)   (or   Y=ERFCX_R8(X)).
        !
        !   The routine  CALERF  is intended for internal packet use only,
        !   all computations within the packet being concentrated in this
        !   routine.  The function subprograms invoke  CALERF  with the
        !   statement
        !
        !          CALL CALERF(ARG,RESULT,JINT)
        !
        !   where the parameter usage is as follows
        !
        !      Function                     Parameters for CALERF
        !       call              ARG                  Result          JINT
        !
        !     ERF(ARG)      ANY REAL ARGUMENT         ERF(ARG)          0
        !     ERFC(ARG)     ABS(ARG) .LT. XBIG        ERFC(ARG)         1
        !     ERFCX(ARG)    XNEG .LT. ARG .LT. XMAX   ERFCX(ARG)        2
        !
        !   The main computation evaluates near-minimax approximations
        !   from "Rational Chebyshev approximations for the error function"
        !   by W. J. Cody, Math. Comp., 1969, PP. 631-638.  This
        !   transportable program uses rational functions that theoretically
        !   approximate  erf(x)  and  erfc(x)  to at least 18 significant
        !   decimal digits.  The accuracy achieved depends on the arithmetic
        !   system, the compiler, the intrinsic functions, and proper
        !   selection of the machine-dependent constants.
        !
        !*******************************************************************
        !*******************************************************************
        !
        ! Explanation of machine-dependent constants
        !
        !   XMIN   = the smallest positive floating-point number.
        !   XINF   = the largest positive finite floating-point number.
        !   XNEG   = the largest negative argument acceptable to ERFCX;
        !            the negative of the solution to the equation
        !            2*exp(x*x) = XINF.
        !   XSMALL = argument below which erf(x) may be represented by
        !            2*x/sqrt(pi)  and above which  x*x  will not underflow.
        !            A conservative value is the largest machine number X
        !            such that   1.0 + X = 1.0   to machine precision.
        !   XBIG   = largest argument acceptable to ERFC;  solution to
        !            the equation:  W(x) * (1-0.5/x**2) = XMIN,  where
        !            W(x) = exp(-x*x)/[x*sqrt(pi)].
        !   XHUGE  = argument above which  1.0 - 1/(2*x*x) = 1.0  to
        !            machine precision.  A conservative value is
        !            1/[2*sqrt(XSMALL)]
        !   XMAX   = largest acceptable argument to ERFCX; the minimum
        !            of XINF and 1/[sqrt(pi)*XMIN].
        !
        !   Approximate values for some important machines are:
        !
        !                          XMIN       XINF        XNEG     XSMALL
        !
        !  CDC 7600      (S.P.)  3.13E-294   1.26E+322   -27.220  7.11E-15
        !  CRAY-1        (S.P.)  4.58E-2467  5.45E+2465  -75.345  7.11E-15
        !  IEEE (IBM/XT,
        !    SUN, etc.)  (S.P.)  1.18E-38    3.40E+38     -9.382  5.96E-8
        !  IEEE (IBM/XT,
        !    SUN, etc.)  (D.P.)  2.23D-308   1.79D+308   -26.628  1.11D-16
        !  IBM 195       (D.P.)  5.40D-79    7.23E+75    -13.190  1.39D-17
        !  UNIVAC 1108   (D.P.)  2.78D-309   8.98D+307   -26.615  1.73D-18
        !  VAX D-Format  (D.P.)  2.94D-39    1.70D+38     -9.345  1.39D-17
        !  VAX G-Format  (D.P.)  5.56D-309   8.98D+307   -26.615  1.11D-16
        !
        !
        !                          XBIG       XHUGE       XMAX
        !
        !  CDC 7600      (S.P.)  25.922      8.39E+6     1.80X+293
        !  CRAY-1        (S.P.)  75.326      8.39E+6     5.45E+2465
        !  IEEE (IBM/XT,
        !    SUN, etc.)  (S.P.)   9.194      2.90E+3     4.79E+37
        !  IEEE (IBM/XT,
        !    SUN, etc.)  (D.P.)  26.543      6.71D+7     2.53D+307
        !  IBM 195       (D.P.)  13.306      1.90D+8     7.23E+75
        !  UNIVAC 1108   (D.P.)  26.582      5.37D+8     8.98D+307
        !  VAX D-Format  (D.P.)   9.269      1.90D+8     1.70D+38
        !  VAX G-Format  (D.P.)  26.569      6.71D+7     8.98D+307
        !
        !*******************************************************************
        !*******************************************************************
        !
        ! Error returns
        !
        !  The program returns  ERFC = 0      for  ARG .GE. XBIG;
        !
        !                       ERFCX = XINF  for  ARG .LT. XNEG;
        !      and
        !                       ERFCX = 0     for  ARG .GE. XMAX.
        !
        !
        ! Intrinsic functions required are:
        !
        !     ABS, AINT, EXP
        !
        !
        !  Author: W. J. Cody
        !          Mathematics and Computer Science Division
        !          Argonne National Laboratory
        !          Argonne, IL 60439
        !
        !  Latest modification: March 19, 1990
        !
        !------------------------------------------------------------------

        !------------------------------------------------------------------------------------------

        !------------------------------------------------------------------------------------------
        !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        pure FUNCTION shr_spfn_gamma_nonintrinsic_r8(x) RESULT ( gamma )
            !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            !
            ! 7 Feb 2013 -- S. Santos
            ! The following comments are from the original version. Changes have
            ! been made to update syntax and allow inclusion into this module.
            !
            !----------------------------------------------------------------------
            !
            ! THIS ROUTINE CALCULATES THE GAMMA FUNCTION FOR A REAL ARGUMENT X.
            !   COMPUTATION IS BASED ON AN ALGORITHM OUTLINED IN REFERENCE 1.
            !   THE PROGRAM USES RATIONAL FUNCTIONS THAT APPROXIMATE THE GAMMA
            !   FUNCTION TO AT LEAST 20 SIGNIFICANT DECIMAL DIGITS.  COEFFICIENTS
            !   FOR THE APPROXIMATION OVER THE INTERVAL (1,2) ARE UNPUBLISHED.
            !   THOSE FOR THE APPROXIMATION FOR X .GE. 12 ARE FROM REFERENCE 2.
            !   THE ACCURACY ACHIEVED DEPENDS ON THE ARITHMETIC SYSTEM, THE
            !   COMPILER, THE INTRINSIC FUNCTIONS, AND PROPER SELECTION OF THE
            !   MACHINE-DEPENDENT CONSTANTS.
            !
            !
            !*******************************************************************
            !*******************************************************************
            !
            ! EXPLANATION OF MACHINE-DEPENDENT CONSTANTS
            !
            ! BETA   - RADIX FOR THE FLOATING-POINT REPRESENTATION
            ! MAXEXP - THE SMALLEST POSITIVE POWER OF BETA THAT OVERFLOWS
            ! XBIG   - THE LARGEST ARGUMENT FOR WHICH GAMMA(X) IS REPRESENTABLE
            !          IN THE MACHINE, I.E., THE SOLUTION TO THE EQUATION
            !                  GAMMA(XBIG) = BETA**MAXEXP
            ! XINF   - THE LARGEST MACHINE REPRESENTABLE FLOATING-POINT NUMBER;
            !          APPROXIMATELY BETA**MAXEXP
            ! EPS    - THE SMALLEST POSITIVE FLOATING-POINT NUMBER SUCH THAT
            !          1.0+EPS .GT. 1.0
            ! XMININ - THE SMALLEST POSITIVE FLOATING-POINT NUMBER SUCH THAT
            !          1/XMININ IS MACHINE REPRESENTABLE
            !
            !     APPROXIMATE VALUES FOR SOME IMPORTANT MACHINES ARE:
            !
            !                            BETA       MAXEXP        XBIG
            !
            ! CRAY-1         (S.P.)        2         8191        966.961
            ! CYBER 180/855
            !   UNDER NOS    (S.P.)        2         1070        177.803
            ! IEEE (IBM/XT,
            !   SUN, ETC.)   (S.P.)        2          128        35.040
            ! IEEE (IBM/XT,
            !   SUN, ETC.)   (D.P.)        2         1024        171.624
            ! IBM 3033       (D.P.)       16           63        57.574
            ! VAX D-FORMAT   (D.P.)        2          127        34.844
            ! VAX G-FORMAT   (D.P.)        2         1023        171.489
            !
            !                            XINF         EPS        XMININ
            !
            ! CRAY-1         (S.P.)   5.45E+2465   7.11E-15    1.84E-2466
            ! CYBER 180/855
            !   UNDER NOS    (S.P.)   1.26E+322    3.55E-15    3.14E-294
            ! IEEE (IBM/XT,
            !   SUN, ETC.)   (S.P.)   3.40E+38     1.19E-7     1.18E-38
            ! IEEE (IBM/XT,
            !   SUN, ETC.)   (D.P.)   1.79D+308    2.22D-16    2.23D-308
            ! IBM 3033       (D.P.)   7.23D+75     2.22D-16    1.39D-76
            ! VAX D-FORMAT   (D.P.)   1.70D+38     1.39D-17    5.88D-39
            ! VAX G-FORMAT   (D.P.)   8.98D+307    1.11D-16    1.12D-308
            !
            !*******************************************************************
            !*******************************************************************
            !
            ! ERROR RETURNS
            !
            !  THE PROGRAM RETURNS THE VALUE XINF FOR SINGULARITIES OR
            !     WHEN OVERFLOW WOULD OCCUR.  THE COMPUTATION IS BELIEVED
            !     TO BE FREE OF UNDERFLOW AND OVERFLOW.
            !
            !
            !  INTRINSIC FUNCTIONS REQUIRED ARE:
            !
            !     INT, DBLE, EXP, LOG, REAL, SIN
            !
            !
            ! REFERENCES:  AN OVERVIEW OF SOFTWARE DEVELOPMENT FOR SPECIAL
            !              FUNCTIONS   W. J. CODY, LECTURE NOTES IN MATHEMATICS,
            !              506, NUMERICAL ANALYSIS DUNDEE, 1975, G. A. WATSON
            !              (ED.), SPRINGER VERLAG, BERLIN, 1976.
            !
            !              COMPUTER APPROXIMATIONS, HART, ET. AL., WILEY AND
            !              SONS, NEW YORK, 1968.
            !
            !  LATEST MODIFICATION: OCTOBER 12, 1989
            !
            !  AUTHORS: W. J. CODY AND L. STOLTZ
            !           APPLIED MATHEMATICS DIVISION
            !           ARGONNE NATIONAL LABORATORY
            !           ARGONNE, IL 60439
            !
            !----------------------------------------------------------------------
            REAL(KIND=r8), intent(in) :: x
            REAL(KIND=r8) :: gamma
            REAL(KIND=r8) :: fact
            REAL(KIND=r8) :: sum
            REAL(KIND=r8) :: y
            REAL(KIND=r8) :: y1
            REAL(KIND=r8) :: res
            REAL(KIND=r8) :: z
            REAL(KIND=r8) :: xnum
            REAL(KIND=r8) :: xden
            REAL(KIND=r8) :: ysq
            INTEGER :: n
            INTEGER :: i
            LOGICAL :: negative_odd
            ! log(2*pi)/2
            REAL(KIND=r8), parameter :: logsqrt2pi = 0.9189385332046727417803297e0_r8
            !----------------------------------------------------------------------
            !  NUMERATOR AND DENOMINATOR COEFFICIENTS FOR RATIONAL MINIMAX
            !     APPROXIMATION OVER (1,2).
            !----------------------------------------------------------------------
            REAL(KIND=r8), parameter :: p(8) =        (/-1.71618513886549492533811e+0_r8, 2.47656508055759199108314e+1_r8,        &
              -3.79804256470945635097577e+2_r8, 6.29331155312818442661052e+2_r8,           8.66966202790413211295064e+2_r8,&
            -3.14512729688483675254357e+4_r8,          -3.61444134186911729807069e+4_r8, 6.64561438202405440627855e+4_r8 /)
            REAL(KIND=r8), parameter :: q(8) =        (/-3.08402300119738975254353e+1_r8, 3.15350626979604161529144e+2_r8,        &
              -1.01515636749021914166146e+3_r8,-3.10777167157231109440444e+3_r8,           2.25381184209801510330112e+4_r8, &
            4.75584627752788110767815e+3_r8,          -1.34659959864969306392456e+5_r8,-1.15132259675553483497211e+5_r8 /)
            !----------------------------------------------------------------------
            !  COEFFICIENTS FOR MINIMAX APPROXIMATION OVER (12, INF).
            !----------------------------------------------------------------------
            REAL(KIND=r8), parameter :: c(7) =        (/-1.910444077728e-03_r8,          8.4171387781295e-04_r8,          &
            -5.952379913043012e-04_r8,       7.93650793500350248e-04_r8,          -2.777777777777681622553e-03_r8, &
            8.333333333333333331554247e-02_r8,           5.7083835261e-03_r8 /)
            negative_odd = .false.
            fact = 1._r8
            n = 0
            y = x
            IF (y <= 0._r8) THEN
                !----------------------------------------------------------------------
                !  ARGUMENT IS NEGATIVE
                !----------------------------------------------------------------------
                y = -x
                y1 = aint(y)
                res = y - y1
                IF (res /= 0._r8) THEN
                    negative_odd = (y1 /= aint(y1*0.5_r8)*2._r8)
                    fact = -pi/sin(pi*res)
                    y = y + 1._r8
                    ELSE
                    gamma = xinfr8
                    RETURN
                END IF 
            END IF 
            !----------------------------------------------------------------------
            !  ARGUMENT IS POSITIVE
            !----------------------------------------------------------------------
            IF (y < epsr8) THEN
                !----------------------------------------------------------------------
                !  ARGUMENT .LT. EPS
                !----------------------------------------------------------------------
                IF (y >= xminr8) THEN
                    res = 1._r8/y
                    ELSE
                    gamma = xinfr8
                    RETURN
                END IF 
                ELSE IF (y < 12._r8) THEN
                y1 = y
                IF (y < 1._r8) THEN
                    !----------------------------------------------------------------------
                    !  0.0 .LT. ARGUMENT .LT. 1.0
                    !----------------------------------------------------------------------
                    z = y
                    y = y + 1._r8
                    ELSE
                    !----------------------------------------------------------------------
                    !  1.0 .LT. ARGUMENT .LT. 12.0, REDUCE ARGUMENT IF NECESSARY
                    !----------------------------------------------------------------------
                    n = int(y) - 1
                    y = y - real(n, r8)
                    z = y - 1._r8
                END IF 
                !----------------------------------------------------------------------
                !  EVALUATE APPROXIMATION FOR 1.0 .LT. ARGUMENT .LT. 2.0
                !----------------------------------------------------------------------
                xnum = 0._r8
                xden = 1._r8
                DO i=1,8
                    xnum = (xnum+p(i))*z
                    xden = xden*z + q(i)
                END DO 
                res = xnum/xden + 1._r8
                IF (y1 < y) THEN
                    !----------------------------------------------------------------------
                    !  ADJUST RESULT FOR CASE  0.0 .LT. ARGUMENT .LT. 1.0
                    !----------------------------------------------------------------------
                    res = res/y1
                    ELSE IF (y1 > y) THEN
                    !----------------------------------------------------------------------
                    !  ADJUST RESULT FOR CASE  2.0 .LT. ARGUMENT .LT. 12.0
                    !----------------------------------------------------------------------
                    DO i = 1,n
                        res = res*y
                        y = y + 1._r8
                    END DO 
                END IF 
                ELSE
                !----------------------------------------------------------------------
                !  EVALUATE FOR ARGUMENT .GE. 12.0,
                !----------------------------------------------------------------------
                IF (y <= xbig_gamma) THEN
                    ysq = y*y
                    sum = c(7)
                    DO i=1,6
                        sum = sum/ysq + c(i)
                    END DO 
                    sum = sum/y - y + logsqrt2pi
                    sum = sum + (y-0.5_r8)*log(y)
                    res = exp(sum)
                    ELSE
                    gamma = xinfr8
                    RETURN
                END IF 
            END IF 
            !----------------------------------------------------------------------
            !  FINAL ADJUSTMENTS AND RETURN
            !----------------------------------------------------------------------
            IF (negative_odd) res = -res
            IF (fact /= 1._r8) res = fact/res
            gamma = res
            ! ---------- LAST LINE OF GAMMA ----------
        END FUNCTION shr_spfn_gamma_nonintrinsic_r8
        !! Incomplete Gamma function
        !!
        !! @author  Tianyi Fan
        !! @version August-2010

    END MODULE shr_spfn_mod
