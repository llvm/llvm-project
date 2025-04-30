! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 

#ifndef DESC_I8
#define IEEE_ARITHMETIC ieee_arithmetic
#else
#define IEEE_ARITHMETIC ieee_arithmetic_la
#endif

module IEEE_ARITHMETIC

  use ieee_exceptions
  use, intrinsic :: iso_c_binding
#ifdef PGDLL
!DEC$ ATTRIBUTES DLLEXPORT :: IEEE_ARITHMETIC
#endif

  type ieee_class_type
    private
    integer :: ct
  end type ieee_class_type

  type ieee_round_type
    private
    integer :: rt
  end type ieee_round_type

  integer, private, parameter :: round_lo = -1
  integer, private, parameter :: round_hi = 3

#ifdef TARGET_LINUX_ARM
  integer, private, parameter :: FE_TONEAREST  = 0
  integer, private, parameter :: FE_DOWNWARD   = X'00800000'
  integer, private, parameter :: FE_UPWARD     = X'00400000'
  integer, private, parameter :: FE_TOWARDZERO = X'00c00000'
#else
#ifdef TARGET_LINUX_POWER
  integer, private, parameter :: FE_TONEAREST  = 0
  integer, private, parameter :: FE_TOWARDZERO = 1
  integer, private, parameter :: FE_UPWARD     = 2
  integer, private, parameter :: FE_DOWNWARD   = 3
#else
  ! These are from fenv.h on linux64
  integer, private, parameter :: FE_TONEAREST  = 0
  integer, private, parameter :: FE_DOWNWARD   = 1024
  integer, private, parameter :: FE_UPWARD     = 2048
  integer, private, parameter :: FE_TOWARDZERO = 3072
#endif
#endif

  type(ieee_round_type), parameter :: ieee_nearest = ieee_round_type(0)
  type(ieee_round_type), parameter :: ieee_down    = ieee_round_type(1)
  type(ieee_round_type), parameter :: ieee_up      = ieee_round_type(2)
  type(ieee_round_type), parameter :: ieee_to_zero = ieee_round_type(3)
  type(ieee_round_type), parameter :: ieee_other   = ieee_round_type(-1)

  type(ieee_class_type), parameter :: ieee_positive_zero=ieee_class_type(0)
  type(ieee_class_type), parameter :: ieee_negative_zero=ieee_class_type(1)
  type(ieee_class_type), parameter :: ieee_positive_denormal=ieee_class_type(2)
  type(ieee_class_type), parameter :: ieee_negative_denormal=ieee_class_type(3)
  type(ieee_class_type), parameter :: ieee_positive_normal=ieee_class_type(4)
  type(ieee_class_type), parameter :: ieee_negative_normal=ieee_class_type(5)
  type(ieee_class_type), parameter :: ieee_positive_inf=ieee_class_type(6)
  type(ieee_class_type), parameter :: ieee_negative_inf=ieee_class_type(7)
  type(ieee_class_type), parameter :: ieee_signaling_nan=ieee_class_type(8)
  type(ieee_class_type), parameter :: ieee_quiet_nan=ieee_class_type(9)
  type(ieee_class_type), parameter :: ieee_other_value=ieee_class_type(15)

  private ieee_arithmetic_eqi
  private ieee_arithmetic_eqr
  private ieee_arithmetic_eqti
  private ieee_arithmetic_eqtr
  private ieee_arithmetic_neti
  private ieee_arithmetic_netr

  interface assignment (=)
    module procedure ieee_arithmetic_eqi
    module procedure ieee_arithmetic_eqr
    module procedure ieee_arithmetic_eqct
  end interface

  interface operator (.eq.)
    module procedure ieee_arithmetic_eqti
    module procedure ieee_arithmetic_eqtr
    module procedure ieee_arithmetic_eqtct
  end interface

  interface operator (==)
    module procedure ieee_arithmetic_eqti
    module procedure ieee_arithmetic_eqtr
    module procedure ieee_arithmetic_eqtct
  end interface

  interface operator (.ne.)
    module procedure ieee_arithmetic_neti
    module procedure ieee_arithmetic_netr
    module procedure ieee_arithmetic_netct
  end interface

  interface operator (/=)
    module procedure ieee_arithmetic_neti
    module procedure ieee_arithmetic_netr
    module procedure ieee_arithmetic_netct
  end interface

  ! Generic interfaces
  !--------------------
  interface ieee_support_datatype
    module procedure ieee_support_datatypenox, ieee_support_datatyper
  end interface

  interface ieee_support_denormal
    module procedure ieee_support_denormalnox, ieee_support_denormalr
  end interface

  interface ieee_support_divide
    module procedure ieee_support_dividenox, ieee_support_divider
  end interface

  interface ieee_support_inf
    module procedure ieee_support_infnox, ieee_support_infr
  end interface

  interface ieee_support_nan
    module procedure ieee_support_nannox, ieee_support_nanr
  end interface

  interface ieee_support_rounding
    module procedure ieee_support_roundingnox, ieee_support_roundingr
  end interface

  interface ieee_support_sqrt
    module procedure ieee_support_sqrtnox, ieee_support_sqrtr
  end interface

  interface ieee_support_standard
    module procedure ieee_support_standardnox, ieee_support_standardr
  end interface

  interface ieee_support_underflow_control
    module procedure ieee_support_uflowctrlnox, ieee_support_uflowctrlr
  end interface

  interface ieee_get_underflow_mode
    module procedure ieee_get_underflow_mode, ieee_get_underflow_model8
  end interface

  interface ieee_set_underflow_mode
    module procedure ieee_set_underflow_mode, ieee_set_underflow_model8
  end interface

  ! More Generic interfaces
  !--------------------
  interface ieee_class
    module procedure ieee_classr4, ieee_classr8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_classr16
#endif
  end interface

  interface ieee_copy_sign
    module procedure ieee_copy_signr4, ieee_copy_signr8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_copy_signr16
#endif
  end interface

  interface ieee_is_finite
    module procedure ieee_is_finiter4, ieee_is_finiter8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_is_finiter16
#endif
  end interface

  interface ieee_is_nan
    module procedure ieee_is_nanr4, ieee_is_nanr8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_is_nanr16
#endif
  end interface

  interface ieee_is_negative
    module procedure ieee_is_negativer4, ieee_is_negativer8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_is_negativer16
#endif
  end interface

  interface ieee_is_normal
    module procedure ieee_is_normalr4, ieee_is_normalr8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_is_normalr16
#endif
  end interface

  interface ieee_logb
    module procedure ieee_logbr4, ieee_logbr8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_logbr16
#endif
  end interface

  interface ieee_next_after
    module procedure ieee_nextafterr4, ieee_nextafterr8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_nextafterr16
#endif
  end interface

  interface ieee_rem
    module procedure ieee_rem4x4, ieee_rem4x8
    module procedure ieee_rem8x4, ieee_rem8x8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_rem4x16
    module procedure ieee_rem8x16
    module procedure ieee_rem16x4, ieee_rem16x8, ieee_rem16x16
#endif
  end interface

  interface ieee_rint
    module procedure ieee_rintr4, ieee_rintr8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_rintr16
#endif
  end interface

  interface ieee_scalb
    module procedure ieee_scalbr4, ieee_scalbr8
    module procedure ieee_scalbr4i8, ieee_scalbr8i8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_scalbr16, ieee_scalbr16i8
#endif
  end interface

  interface ieee_unordered
    module procedure ieee_unorderedr4, ieee_unorderedr8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_unorderedr16
#endif
  end interface

  interface ieee_value
    module procedure ieee_valuer4, ieee_valuer8
#ifdef TARGET_SUPPORTS_QUADFP
    module procedure ieee_valuer16
#endif
  end interface

  ! External interfaces
  !--------------------
  interface
    function __fenv_fegetround() bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int) :: __fenv_fegetround
    end function __fenv_fegetround
  end interface
  interface
    function __fenv_fesetround(rounds) bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: rounds
      integer(c_int) :: __fenv_fesetround
    end function __fenv_fesetround
  end interface
  interface
    function __fenv_fegetzerodenorm() bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int) :: __fenv_fegetzerodenorm
    end function __fenv_fegetzerodenorm
  end interface
  interface
    function __fenv_fesetzerodenorm(uflow) bind(c)
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: uflow
      integer(c_int) :: __fenv_fesetzerodenorm
    end function __fenv_fesetzerodenorm
  end interface

#ifdef TARGET_SUPPORTS_QUADFP
  interface
    pure function __nearbyintl(x) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_long_double), value :: x
      real(c_long_double) :: __nearbyintl
    end function __nearbyintl
  end interface
#endif
  interface
    pure function __nearbyint(x) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_double), value :: x
      real(c_double) :: __nearbyint
    end function __nearbyint
  end interface
  interface
    pure function __nearbyintf(x) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_float), value :: x
      real(c_float) :: __nearbyintf
    end function __nearbyintf
  end interface

#ifdef TARGET_SUPPORTS_QUADFP
  interface
    pure function __remainderl(x, y) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_long_double), value :: x, y
      real(c_long_double) :: __remainderl
    end function __remainderl
  end interface
#endif
  interface
    pure function __remainder(x, y) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_double), value :: x, y
      real(c_double) :: __remainder
    end function __remainder
  end interface
  interface
    pure function __remainderf(x, y) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_float), value :: x, y
      real(c_float) :: __remainderf
    end function __remainderf
  end interface

#ifdef TARGET_SUPPORTS_QUADFP
  interface
    pure function __nextafterl(x, y) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_long_double), value :: x, y
      real(c_long_double) :: __nextafterl
      end function __nextafterl
  end interface
#endif
  interface
    pure function __nextafter(x, y) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_double), value :: x, y
      real(c_double) :: __nextafter
    end function __nextafter
  end interface
  interface
    pure function __nextafterf(x, y) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_float), value :: x, y
      real(c_float) :: __nextafterf
    end function __nextafterf
  end interface

#ifdef TARGET_SUPPORTS_QUADFP
  interface
    pure function __scalbnl(x, i) bind(c)
    use, intrinsic :: iso_c_binding
    real(c_long_double), value :: x
    integer(c_int), value :: i
    real(c_long_double) :: __scalbnl
    end function __scalbnl
  end interface
#endif
  interface
    pure function __scalbn(x, i) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_double), value :: x
      integer(c_int), value :: i
      real(c_double) :: __scalbn
    end function __scalbn
  end interface
  interface
    pure function __scalbnf(x, i) bind(c)
      use, intrinsic :: iso_c_binding
      real(c_float), value :: x
      integer(c_int), value :: i
      real(c_float) :: __scalbnf
    end function __scalbnf
  end interface

contains
  subroutine ieee_arithmetic_eqi(ra, ib)
    type(ieee_round_type), intent(out) :: ra
    integer, intent(in) :: ib
    if ((ib .ge. round_lo) .and. (ib .le. round_hi)) then
      ra%rt = ib
    endif
    return
  end subroutine

  subroutine ieee_arithmetic_eqr(ra, rb)
    type(ieee_round_type), intent(out) :: ra
    type(ieee_round_type), intent(in)  :: rb
    ra%rt = rb%rt
    return
  end subroutine

  elemental logical function ieee_arithmetic_eqti(ra, ib)
    type(ieee_round_type), intent(in) :: ra
    integer, intent(in) :: ib
    if (ra%rt .eq. ib) then
      ieee_arithmetic_eqti = .true.
    else
      ieee_arithmetic_eqti = .false.
    endif
    return
  end function

  elemental logical function ieee_arithmetic_eqtr(ra, rb)
    type(ieee_round_type), intent(in) :: ra
    type(ieee_round_type), intent(in)  :: rb
    if (ra%rt .eq. rb%rt) then
      ieee_arithmetic_eqtr = .true.
    else
      ieee_arithmetic_eqtr = .false.
    endif
    return
  end function

  elemental logical function ieee_arithmetic_neti(ra, ib)
    type(ieee_round_type), intent(in) :: ra
    integer, intent(in) :: ib
    if (ra%rt .ne. ib) then
      ieee_arithmetic_neti = .true.
    else
      ieee_arithmetic_neti = .false.
    endif
    return
  end function

  elemental logical function ieee_arithmetic_netr(ra, rb)
    type(ieee_round_type), intent(in) :: ra
    type(ieee_round_type), intent(in) :: rb
    if (ra%rt .ne. rb%rt) then
      ieee_arithmetic_netr = .true.
    else
      ieee_arithmetic_netr = .false.
    endif
    return
  end function

  pure subroutine ieee_arithmetic_eqct(ca, cb)
    type(ieee_class_type), intent(out) :: ca
    type(ieee_class_type), intent(in)  :: cb
    ca%ct = cb%ct
    return
  end subroutine

  elemental logical function ieee_arithmetic_eqtct(ca, cb)
    type(ieee_class_type), intent(in) :: ca
    type(ieee_class_type), intent(in) :: cb
    if (ca%ct .eq. cb%ct) then
      ieee_arithmetic_eqtct = .true.
    else
      ieee_arithmetic_eqtct = .false.
    endif
    return
  end function

  elemental logical function ieee_arithmetic_netct(ca, cb)
    type(ieee_class_type), intent(in) :: ca
    type(ieee_class_type), intent(in) :: cb
    if (ca%ct .ne. cb%ct) then
      ieee_arithmetic_netct = .true.
    else
      ieee_arithmetic_netct = .false.
    endif
    return
  end function

!--------------------------------------------------------------------------
! Inquiry functions for IEEE arithmetic

  pure logical function ieee_support_datatypenox()
!pgi$ defaultkind
!! OR !pgi$ defaultkind (ieee_support_datatypenox)
    ieee_support_datatypenox = .true.
    return
  end function ieee_support_datatypenox
  pure logical function ieee_support_datatyper(x)
!pgi$ defaultkind
!dir$ ignore_tkr (kr) x
    real :: x
    ieee_support_datatyper = .true.
    return 
  end function ieee_support_datatyper

  pure logical function ieee_support_denormalnox()
!pgi$ defaultkind
#if defined TARGET_LINUX_ARM || defined TARGET_LINUX_POWER || defined PGFLANG
    ieee_support_denormalnox = .false.
#else
    ieee_support_denormalnox = .true.
#endif
    return
  end function ieee_support_denormalnox
  pure logical function ieee_support_denormalr(x)
!pgi$ defaultkind
!dir$ ignore_tkr (kr) x
    real :: x
#if defined TARGET_LINUX_ARM || defined TARGET_LINUX_POWER || defined PGFLANG
    ieee_support_denormalr = .false.
#else
    ieee_support_denormalr = .true.
#endif
    return 
  end function ieee_support_denormalr

  pure logical function ieee_support_dividenox()
!pgi$ defaultkind
    ieee_support_dividenox = .true.
    return
  end function ieee_support_dividenox
  pure logical function ieee_support_divider(x)
!pgi$ defaultkind
!dir$ ignore_tkr (kr) x
    real :: x
    ieee_support_divider = .true.
    return 
  end function ieee_support_divider

  pure logical function ieee_support_infnox()
!pgi$ defaultkind
    ieee_support_infnox = .true.
    return
  end function ieee_support_infnox
  pure logical function ieee_support_infr(x)
!pgi$ defaultkind
!dir$ ignore_tkr (kr) x
    real :: x
    ieee_support_infr = .true.
    return 
  end function ieee_support_infr

  pure logical function ieee_support_nannox()
!pgi$ defaultkind
    ieee_support_nannox = .true.
    return
  end function ieee_support_nannox
  pure logical function ieee_support_nanr(x)
!pgi$ defaultkind
!dir$ ignore_tkr (kr) x
    real :: x
    ieee_support_nanr = .true.
    return 
  end function ieee_support_nanr

  pure logical function ieee_support_roundingnox(rv)
!pgi$ defaultkind
    type(ieee_round_type) :: rv
    i = rv%rt
    ieee_support_roundingnox = ((i.ge.0).and.(i.le.3))
    return
  end function ieee_support_roundingnox
  pure logical function ieee_support_roundingr(rv,x)
!pgi$ defaultkind
    type(ieee_round_type) :: rv
!dir$ ignore_tkr (kr) x
    real :: x
    i = rv%rt
    ieee_support_roundingr = ((i.ge.0).and.(i.le.3))
    return 
  end function ieee_support_roundingr

  pure logical function ieee_support_sqrtnox()
!pgi$ defaultkind
    ieee_support_sqrtnox = .true.
    return
  end function ieee_support_sqrtnox
  pure logical function ieee_support_sqrtr(x)
!pgi$ defaultkind
!dir$ ignore_tkr (kr) x
    real :: x
    ieee_support_sqrtr = .true.
    return 
  end function ieee_support_sqrtr

  pure logical function ieee_support_standardnox()
!pgi$ defaultkind
    ieee_support_standardnox = .true.
    return
  end function ieee_support_standardnox
  pure logical function ieee_support_standardr(x)
!pgi$ defaultkind
!dir$ ignore_tkr (kr) x
    real :: x
    ieee_support_standardr = .true.
    return 
  end function ieee_support_standardr

  pure logical function ieee_support_uflowctrlnox()
!pgi$ defaultkind
#if defined TARGET_LINUX_ARM || defined TARGET_LINUX_POWER || defined PGFLANG
    ieee_support_uflowctrlnox = .false.
#else
    ieee_support_uflowctrlnox = .true.
#endif
    return
  end function ieee_support_uflowctrlnox
  pure logical function ieee_support_uflowctrlr(x)
!pgi$ defaultkind
!dir$ ignore_tkr (kr) x
    real :: x
#if defined TARGET_LINUX_ARM || defined TARGET_LINUX_POWER || defined PGFLANG
    ieee_support_uflowctrlr = .false.
#else
    ieee_support_uflowctrlr = .true.
#endif
    return 
  end function ieee_support_uflowctrlr

  !-----------------------------------------------------------------
  subroutine ieee_get_rounding_mode(ra)
    type(ieee_round_type), intent(out) :: ra
    integer i
    i = __fenv_fegetround()
    if (i .eq. FE_TOWARDZERO) then
      ra = 3
    else if (i .eq. FE_UPWARD) then
      ra = 2
    else if (i .eq. FE_DOWNWARD) then
      ra = 1
    else if (i .eq. FE_TONEAREST) then
      ra = 0
    else
      ra = -1
    endif
  end subroutine
   
  subroutine ieee_set_rounding_mode(ra)
    type(ieee_round_type), intent(in) :: ra
    integer i,j
    i = ra%rt
    if (i .eq. 3) then
      j = __fenv_fesetround(FE_TOWARDZERO)
    else if (i .eq. 2) then
      j = __fenv_fesetround(FE_UPWARD)
    else if (i .eq. 1) then
      j = __fenv_fesetround(FE_DOWNWARD)
    else if (i .eq. 0) then
      j = __fenv_fesetround(FE_TONEAREST)
    endif
  end subroutine

  !-----------------------------------------------------------------
  subroutine ieee_get_underflow_mode(uflow)
    logical*4, intent(out) :: uflow
    integer i
#if defined TARGET_LINUX_POWER
      uflow = .true.
#else
    i = __fenv_fegetzerodenorm()
    if (i .eq. 0) then
      uflow = .true.
    else 
      uflow = .false.
    endif
#endif
  end subroutine
   
  subroutine ieee_get_underflow_model8(uflow)
    logical*8, intent(out) :: uflow
    integer i
#if defined TARGET_LINUX_POWER
      uflow = .true.
#else
    i = __fenv_fegetzerodenorm()
    if (i .eq. 0) then
      uflow = .true.
    else 
      uflow = .false.
    endif
#endif
  end subroutine
   
  subroutine ieee_set_underflow_mode(uflow)
    logical*4, intent(in) :: uflow
    integer i, j
    if (uflow) then
      i = 0
    else 
      i = 1
    endif
    j = __fenv_fesetzerodenorm(i)
  end subroutine
   
  subroutine ieee_set_underflow_model8(uflow)
    logical*8, intent(in) :: uflow
    integer i, j
    if (uflow) then
      i = 0
    else 
      i = 1
    endif
    j = __fenv_fesetzerodenorm(i)
  end subroutine
   
  !-----------------------------------------------------------------
  elemental type(ieee_class_type) function ieee_classr4(x)
    real*4 x, ex
    integer*4 ix, iexp, imant
#if 0
   ix = transfer(x,ix)
#else
    equivalence(ex,ix)
#endif
    ex = x
    iexp = ishft(ix,-23)
    iexp = iand(iexp,z'ff')
    if (iexp .eq. 0) then
      if (ix .eq. 0) then
        ieee_classr4 = ieee_positive_zero
      else if (iand(ix,z'7fffffff').eq.0) then
        ieee_classr4 = ieee_negative_zero
      else if (iand(ix,z'80000000').eq.0) then
        ieee_classr4 = ieee_positive_denormal
      else
        ieee_classr4 = ieee_negative_denormal
      endif
    else  if (iexp .lt. 255) then
      if (iand(ix,z'80000000').eq.0) then
        ieee_classr4 = ieee_positive_normal
      else
        ieee_classr4 = ieee_negative_normal
      endif
    else
      imant = iand(ix,z'7fffff')
      if (imant .eq. 0) then
        if (iand(ix,z'80000000').eq.0) then
          ieee_classr4 = ieee_positive_inf
        else
          ieee_classr4 = ieee_negative_inf
        endif
      else if (iand(ix,z'400000') .ne. 0) then
        ieee_classr4 = ieee_quiet_nan
      else
        ieee_classr4 = ieee_signaling_nan
      end if
    end if
    return
  end function

  elemental type(ieee_class_type) function ieee_classr8(x)
    real*8 x, ex
    integer*4 iz(2), ix, iy, iexp, imant
#if 0
    iz = transfer(x,iz)
#else
    equivalence(ex,iz)
#endif
    ex = x
    ix = iz(2)
    iy = iz(1)
    iexp = ishft(ix,-20)
    iexp = iand(iexp,z'7ff')

    if (iexp .eq. 0) then
      if ((ix .eq. 0) .and. (iy .eq. 0)) then
        ieee_classr8 = ieee_positive_zero
      else if ((iand(ix,z'7fffffff').eq.0) .and. (iy .eq. 0)) then
        ieee_classr8 = ieee_negative_zero
      else if (iand(ix,z'80000000').eq.0) then
        ieee_classr8 = ieee_positive_denormal
      else
        ieee_classr8 = ieee_negative_denormal
      endif
    else  if (iexp .lt. 2047) then
      if (iand(ix,z'80000000').eq.0) then
        ieee_classr8 = ieee_positive_normal
      else
        ieee_classr8 = ieee_negative_normal
      endif
    else
      imant = ior(iand(ix,z'fffff'),iy)
      if (imant .eq. 0) then
        if (iand(ix,z'80000000').eq.0) then
          ieee_classr8 = ieee_positive_inf
        else
          ieee_classr8 = ieee_negative_inf
        endif
      else if (iand(ix,z'80000') .ne. 0) then
        ieee_classr8 = ieee_quiet_nan
      else
        ieee_classr8 = ieee_signaling_nan
      end if
    end if
    return
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental type(ieee_class_type) function ieee_classr16(x)
    real*16 x, ex
    integer*4 ir(4), iw, ix, iy, iz, iexp, imant
    equivalence(ex, ir)
    ex = x
    iw = ir(4)
    ix = ir(3)
    iy = ir(2)
    iz = ir(1)
    iexp = ishft(iw, -16)
    iexp = iand(iexp, z'7fff')

    if (iexp .eq. 0) then
      if((iw .eq. 0 ) .and. (ix .eq. 0) .and. (iy .eq. 0) .and. (iz .eq. 0)) then
        ieee_classr16 = ieee_positive_zero
      else if ((iand(iw, z'7fffffff') .eq. 0) .and. (ix .eq. 0) .and. (iy .eq. 0) .and. (iz .eq. 0)) then
        ieee_classr16 = ieee_negative_zero
      else if (iand(iw, z'80000000') .eq. 0) then
        ieee_classr16 = ieee_positive_denormal
      else
        ieee_classr16 = ieee_negative_denormal
      endif
    else  if (iexp .lt. 32767) then
      if (iand(iw, z'80000000') .eq. 0) then
        ieee_classr16 = ieee_positive_normal
      else
        ieee_classr16 = ieee_negative_normal
      endif
    else
      imant = ior(ior(ior(iand(iw, z'ffff'), ix), iy), iz)
      if (imant .eq. 0) then
        if (iand(iw, z'80000000') .eq. 0) then
          ieee_classr16 = ieee_positive_inf
        else
          ieee_classr16 = ieee_negative_inf
        endif
      else if (iand(iw, z'8000') .ne. 0) then
        ieee_classr16 = ieee_quiet_nan
      else
        ieee_classr16 = ieee_signaling_nan
      end if
    end if
    return
  end function
#endif

  elemental real*4 function ieee_valuer4(x, cl)
    real*4 x, ex
    type(ieee_class_type) :: cl
    integer*4 ix
#if 0
    ix = transfer(x,ix)
#else
    equivalence(ex,ix)
#endif
    if (cl%ct .eq. 0) ix = z'00000000'
    if (cl%ct .eq. 1) ix = z'80000000'
    if (cl%ct .eq. 2) ix = z'00400000'
    if (cl%ct .eq. 3) ix = z'80400000'
    if (cl%ct .eq. 4) ix = z'3f800000'
    if (cl%ct .eq. 5) ix = z'bf800000'
    if (cl%ct .eq. 6) ix = z'7f800000'
    if (cl%ct .eq. 7) ix = z'ff800000'
    if (cl%ct .eq. 8) ix = z'7fa00000'
    if (cl%ct .eq. 9) ix = z'7fc00000'
#if 0
    ieee_valuer4 = transfer(ix,x)
#else
    ieee_valuer4 = ex
#endif
  end function

  elemental real*8 function ieee_valuer8(x, cl)
    real*8 x, ex
    type(ieee_class_type) :: cl
    integer*4 ix, iz(2)
#if 0
    iz = transfer(x,iz)
#else
    equivalence(ex,iz)
#endif
    if (cl%ct .eq. 0) ix = z'00000000'
    if (cl%ct .eq. 1) ix = z'80000000'
    if (cl%ct .eq. 2) ix = z'00080000'
    if (cl%ct .eq. 3) ix = z'80080000'
    if (cl%ct .eq. 4) ix = z'3ff00000'
    if (cl%ct .eq. 5) ix = z'bff00000'
    if (cl%ct .eq. 6) ix = z'7ff00000'
    if (cl%ct .eq. 7) ix = z'fff00000'
    if (cl%ct .eq. 8) ix = z'7ff40000'
    if (cl%ct .eq. 9) ix = z'7ff80000'
    iz(1) = 0
    iz(2) = ix
#if 0
    ieee_valuer8 = transfer(iz,x)
#else
    ieee_valuer8 = ex
#endif
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental real*16 function ieee_valuer16(x, cl)
    real*16 x, ex
    type(ieee_class_type) :: cl
    integer*4 ix, iz(4)
    equivalence(ex, iz)
    if (cl%ct .eq. 0) ix = z'00000000'
    if (cl%ct .eq. 1) ix = z'80000000'
    if (cl%ct .eq. 2) ix = z'00008000'
    if (cl%ct .eq. 3) ix = z'80008000'
    if (cl%ct .eq. 4) ix = z'3fff0000'
    if (cl%ct .eq. 5) ix = z'bfff0000'
    if (cl%ct .eq. 6) ix = z'7fff0000'
    if (cl%ct .eq. 7) ix = z'ffff0000'
    if (cl%ct .eq. 8) ix = z'7fff4000'
    if (cl%ct .eq. 9) ix = z'7fff8000'
    iz(1) = 0
    iz(2) = 0
    iz(3) = 0
    iz(4) = ix
    ieee_valuer16 = ex
  end function
#endif

  elemental real*4 function ieee_copy_signr4(x, y)
    real*4 x, y, ex, ey
    integer*4 ix, iy
#if 0
    ix = transfer(x,ix)
    iy = transfer(y,iy)
#else
    equivalence(ex,ix)
    equivalence(ey,iy)
#endif
    ex = x
    ey = y
    ix = iand(ix,z'7fffffff')
    iy = iand(iy,z'80000000')
    ix = ior(ix,iy)
#if 0
    ieee_copy_signr4 = transfer(ix,x)
#else
    ieee_copy_signr4 = ex
#endif
  end function

  elemental real*8 function ieee_copy_signr8(x, y)
    real*8 x, y, ex
    integer*4 ix, iy, iz(2)
#if 0
    iz = transfer(ex,iz)
#else
    equivalence(ex,iz)
#endif
#if 0
    iz = transfer(y,iz)
#else
    ex = y
#endif
    iy = iz(2)
    iy = iand(iy,z'80000000')
#if 0
    iz = transfer(x,iz)
#else
    ex = x
#endif
    ix = iz(2)
    ix = iand(ix,z'7fffffff')
    ix = ior(ix,iy)
    iz(2) = ix
#if 0
    ieee_copy_signr8 = transfer(iz,x)
#else
    ieee_copy_signr8 = ex
#endif
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental real*16 function ieee_copy_signr16(x, y)
    real*16 x, y, ex
    integer*4 ix, iy, iz(4)
    equivalence(ex, iz)
    ex = y
    iy = iz(4)
    iy = iand(iy, z'80000000')
    ex = x
    ix = iz(4)
    ix = iand(ix, z'7fffffff')
    ix = ior(ix, iy)
    iz(4) = ix
    ieee_copy_signr16 = ex
  end function
#endif

  elemental logical function ieee_is_finiter4(x)
!pgi$ defaultkind
    real*4 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if (cl%ct .lt. 6) then
      ieee_is_finiter4 = .true.
    else
      ieee_is_finiter4 = .false.
    end if
  end function

  elemental logical function ieee_is_finiter8(x)
!pgi$ defaultkind
    real*8 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if (cl%ct .lt. 6) then
      ieee_is_finiter8 = .true.
    else
      ieee_is_finiter8 = .false.
    end if
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental logical function ieee_is_finiter16(x)
!pgi$ defaultkind
    real*16 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if (cl%ct .lt. 6) then
      ieee_is_finiter16 = .true.
    else
      ieee_is_finiter16 = .false.
    end if
  end function
#endif

  elemental logical function ieee_is_nanr4(x)
!pgi$ defaultkind
    real*4 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if ((cl%ct .eq. 8) .or. (cl%ct .eq. 9)) then
      ieee_is_nanr4 = .true.
    else
      ieee_is_nanr4 = .false.
    end if
  end function

  elemental logical function ieee_is_nanr8(x)
!pgi$ defaultkind
    real*8 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if ((cl%ct .eq. 8) .or. (cl%ct .eq. 9)) then
      ieee_is_nanr8 = .true.
    else
      ieee_is_nanr8 = .false.
    end if
  end function

#ifdef TARGET_SUPPORTS_QUADFP
elemental logical function ieee_is_nanr16(x)
!pgi$ defaultkind
    real*16 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if ((cl%ct .eq. 8) .or. (cl%ct .eq. 9)) then
      ieee_is_nanr16 = .true.
    else
      ieee_is_nanr16 = .false.
    end if
  end function
#endif

  elemental logical function ieee_unorderedr4(x, y)
!pgi$ defaultkind
    real*4 x, y
    ieee_unorderedr4 = (ieee_is_nanr4(x) .or. ieee_is_nanr4(y))
  end function

  elemental logical function ieee_unorderedr8(x, y)
!pgi$ defaultkind
    real*8 x, y
    ieee_unorderedr8 = (ieee_is_nanr8(x) .or. ieee_is_nanr8(y))
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental logical function ieee_unorderedr16(x, y)
!pgi$ defaultkind
    real*16 x, y
    ieee_unorderedr16 = (ieee_is_nanr16(x) .or. ieee_is_nanr16(y))
  end function
#endif

  elemental logical function ieee_is_negativer4(x)
!pgi$ defaultkind
    real*4 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if ((cl%ct .lt. 8) .and. (iand(cl%ct,1) .eq. 1)) then
      ieee_is_negativer4 = .true.
    else
      ieee_is_negativer4 = .false.
    end if
  end function

  elemental logical function ieee_is_negativer8(x)
!pgi$ defaultkind
    real*8 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if ((cl%ct .lt. 8) .and. (iand(cl%ct,1) .eq. 1)) then
      ieee_is_negativer8 = .true.
    else
      ieee_is_negativer8 = .false.
    end if
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental logical function ieee_is_negativer16(x)
!pgi$ defaultkind
    real*16 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if ((cl%ct .lt. 8) .and. (iand(cl%ct, 1) .eq. 1)) then
      ieee_is_negativer16 = .true.
    else
      ieee_is_negativer16 = .false.
    end if
  end function
#endif

  elemental logical function ieee_is_normalr4(x)
!pgi$ defaultkind
    real*4 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if ((cl%ct .lt. 6) .and. (iand(cl%ct,2) .eq. 0)) then
      ieee_is_normalr4 = .true.
    else
      ieee_is_normalr4 = .false.
    end if
  end function

  elemental logical function ieee_is_normalr8(x)
!pgi$ defaultkind
    real*8 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if ((cl%ct .lt. 6) .and. (iand(cl%ct,2) .eq. 0)) then
      ieee_is_normalr8 = .true.
    else
      ieee_is_normalr8 = .false.
    end if
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental logical function ieee_is_normalr16(x)
!pgi$ defaultkind
    real*16 x
    type(ieee_class_type) :: cl
    cl = ieee_class(x)
    if ((cl%ct .lt. 6) .and. (iand(cl%ct, 2) .eq. 0)) then
      ieee_is_normalr16 = .true.
    else
      ieee_is_normalr16 = .false.
    end if
  end function
#endif

  elemental real*4 function ieee_logbr4(x)
    real*4 x, ex
    integer*4 ix, iexp, imant, ibitp
#if 0
   ix = transfer(x,ix)
#else
    equivalence(ex,ix)
#endif
    ex = x
    iexp = ishft(ix,-23)
    iexp = iand(iexp,z'ff')
    if (iand(ix,z'7fffffff').eq.0) then
      ieee_logbr4 = ieee_value(x,ieee_negative_inf)
    else if (iexp .eq. 0) then
      imant = iand(ix,z'007fffff')
      ibitp = z'00800000'
      do while (imant .lt. ibitp)
        ibitp = ibitp / 2
        iexp = iexp - 1
      end do
      ieee_logbr4 = float(iexp - 126)
    else if (iexp .eq. 255) then
      ieee_logbr4 = x
    else
      ieee_logbr4 = float(iexp - 127)
    end if
  end function

  elemental real*8 function ieee_logbr8(x)
    real*8 x, ex
    integer*4 iz(2), ix, iy, iexp, imant
#if 0
   iz = transfer(x,iz)
#else
    equivalence(ex,iz)
#endif
    ex = x
    ix = iz(2)
    iy = iz(1)
    iexp = ishft(ix,-20)
    iexp = iand(iexp,z'7ff')
    if ((iand(ix,z'7fffffff').eq.0) .and. (iy .eq. 0)) then
      ieee_logbr8 = ieee_value(x,ieee_negative_inf)
    else if (iexp .eq. 0) then
      imant = iand(ix,z'000fffff')
      if (imant .ne. 0) then
        ibitp = z'00100000'
        do while (imant .lt. ibitp)
          ibitp = ibitp / 2
          iexp = iexp - 1
        end do
      else
        iexp = iexp - 20
        imant = iand(ishft(iy,-12),z'000fffff')
        if (imant .ne. 0) then
          ibitp = z'00100000'
          do while (imant .lt. ibitp)
            ibitp = ibitp / 2
            iexp = iexp - 1
          end do
        else
          iexp = iexp - 20
          imant = iand(iy,z'00000fff')
          ibitp = z'00001000'
          do while (imant .lt. ibitp)
            ibitp = ibitp / 2
            iexp = iexp - 1
          end do
        end if
      end if
      ieee_logbr8 = dble(iexp - 1022)
    else if (iexp .eq. 2047) then
      ieee_logbr8 = x
    else
      ieee_logbr8 = dble(iexp - 1023)
    end if
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental real*16 function ieee_logbr16(x)
    real*16 x, ex
    integer*4 iz(4), ix, iy, ix2, iy2, iexp, imant
    equivalence(ex, iz)
    ex = x
    ix = iz(4)
    ix2 = iz(3)
    iy = iz(2)
    iy2 = iz(1)
    iexp = ishft(ix, -16)
    iexp = iand(iexp, z'7fff')
    if ((iand(ix, z'7fffffff') .eq. 0) .and. (ix2 .eq. 0) &
        .and. (iy .eq. 0) .and. (iy2 .eq. 0)) then
      ieee_logbr16 = ieee_value(x, ieee_negative_inf)
    else if (iexp .eq. 0) then
      imant = iand(ix, z'0000ffff')
      if (imant .ne. 0) then
        ibitp = z'00010000'
        do while (imant .lt. ibitp)
          ibitp = ibitp / 2
          iexp = iexp - 1
        end do
      else
        iexp = iexp - 16
        imant = iand(ishft(ix2, -16), z'0000ffff')
        if (imant .ne. 0) then
          ibitp = z'00010000'
          do while (imant .lt. ibitp)
            ibitp = ibitp / 2
            iexp = iexp - 1
          end do
        else
          iexp = iexp - 16
          imant = iand(ix2, z'0000ffff')
          if (imant .ne. 0) then
            ibitp = z'00010000'
            do while (imant .lt. ibitp)
              ibitp = ibitp / 2
              iexp = iexp - 1
            end do
          else
            iexp = iexp - 16
            imant = iand(ishft(iy, -16), z'0000ffff')
            if (imant .ne. 0) then
              ibitp = z'00010000'
              do while (imant .lt. ibitp)
                ibitp = ibitp / 2
                iexp = iexp - 1
              end do
            else
              iexp = iexp - 16
              imant = iand(iy, z'0000ffff')
              if (imant .ne. 0) then
                ibitp = z'00010000'
                do while (imant .lt. ibitp)
                  ibitp = ibitp / 2
                  iexp = iexp - 1
                end do
              else
                iexp = iexp - 16
                imant = iand(ishft(iy2, -16), z'0000ffff')
                if (imant .ne. 0) then
                  ibitp = z'00010000'
                  do while (imant .lt. ibitp)
                    ibitp = ibitp / 2
                    iexp = iexp - 1
                  end do
                else
                  iexp = iexp - 16
                  imant = iand(iy2, z'0000ffff')
                  if (imant .ne. 0) then
                    ibitp = z'00010000'
                    do while (imant .lt. ibitp)
                      ibitp = ibitp / 2
                      iexp = iexp - 1
                    end do
                  endif
                endif
              endif
            endif
          endif
        endif
      endif
      ieee_logbr16 = real(iexp - 16382, kind = 16)
    else if (iexp .eq. 32767 ) then
      ieee_logbr16 = x
    else
      ieee_logbr16 = real(iexp - 16383, kind = 16)
    end if
  end function
#endif

  elemental real*4 function ieee_nextafterr4(x,y)
    real*4 x, y
    ieee_nextafterr4 = __nextafterf(x, y)
  end function

  elemental real*8 function ieee_nextafterr8(x,y)
    real*8 x, y
    ieee_nextafterr8 = __nextafter(x, y)
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental real*16 function ieee_nextafterr16(x,y)
    real*16 x, y
    ieee_nextafterr16 = __nextafterl(x, y)
  end function
#endif

  elemental real*4 function ieee_rem4x4(x,y)
    real*4 x, y
    ieee_rem4x4 = __remainderf(x, y)
  end function

  elemental real*8 function ieee_rem4x8(x,y)
    real*4 x
    real*8 y
    ieee_rem4x8 = __remainder(dble(x), y)
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental real*16 function ieee_rem4x16(x,y)
    real*4 x
    real*16 y
    ieee_rem4x16 = __remainderl(real(x, kind = 16), y)
  end function
#endif

  elemental real*8 function ieee_rem8x4(x,y)
    real*8 x
    real*4 y
    ieee_rem8x4 = __remainder(x, dble(y))
  end function

  elemental real*8 function ieee_rem8x8(x,y)
    real*8 x, y
    ieee_rem8x8 = __remainder(x, y)
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental real*16 function ieee_rem8x16(x,y)
    real*8 x
    real*16 y
    ieee_rem8x16 = __remainderl(real(x, kind = 16), y)
  end function

  elemental real*16 function ieee_rem16x4(x,y)
    real*16 x
    real*4 y
    ieee_rem16x4 = __remainderl(x, real(y, kind = 16))
  end function

  elemental real*16 function ieee_rem16x8(x,y)
    real*16 x
    real*8 y
    ieee_rem16x8 = __remainderl(x, real(y, kind = 16))
  end function

  elemental real*16 function ieee_rem16x16(x,y)
    real*16 x, y
    ieee_rem16x16 = __remainderl(x, y)
  end function
#endif

  elemental real*4 function ieee_rintr4(x)
    real*4 x
    ieee_rintr4 = __nearbyintf(x)
  end function

  elemental real*8 function ieee_rintr8(x)
    real*8 x
    ieee_rintr8 = __nearbyint(x)
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental real*16 function ieee_rintr16(x)
    real*16 x
    ieee_rintr16 = __nearbyintl(x)
  end function
#endif

  elemental real*4 function ieee_scalbr4(x, i)
    real*4 x
    integer*4 i
    ieee_scalbr4 = __scalbnf(x, i)
  end function

  elemental real*8 function ieee_scalbr8(x, i)
    real*8 x
    integer*4 i
    ieee_scalbr8 = __scalbn(x, i)
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental real*16 function ieee_scalbr16(x, i)
    real*16 x
    integer*4 i
    ieee_scalbr16 = __scalbnl(x, i)
  end function
#endif

  elemental real*4 function ieee_scalbr4i8(x, i)
    real*4 x
    integer*8 i
    integer*4 j
    if (i .gt. 2048) then
      j = 2048
    else if (i .lt. -2048) then
      j = -2048
    else
      j = i
    end if
    ieee_scalbr4i8 = __scalbnf(x, j)
  end function

  elemental real*8 function ieee_scalbr8i8(x, i)
    real*8 x
    integer*8 i
    integer*4 j
    if (i .gt. 2048) then
      j = 2048
    else if (i .lt. -2048) then
      j = -2048
    else
      j = i
    end if
    ieee_scalbr8i8 = __scalbn(x, j)
  end function

#ifdef TARGET_SUPPORTS_QUADFP
  elemental real*16 function ieee_scalbr16i8(x, i)
    real*16 x
    integer*8 i
    integer*4 j
    if (i .gt. 32768) then
      j = 32768
    else if (i .lt. -32768) then
      j = -32768
    else
      j = i
    end if
    ieee_scalbr16i8 = __scalbnl(x, j)
  end function
#endif

end module IEEE_ARITHMETIC
