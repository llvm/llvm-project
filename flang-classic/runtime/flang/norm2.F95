! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 

#ifdef PGFLANG
#ifdef DESC_I8
#define F90(x) f90_norm2_##x##_i8
#else
#define F90(x) f90_norm2_##x
#endif
#else
#ifdef DESC_I8
#define F90(x) pg_norm2_##x##_i8
#else
#define F90(x) pg_norm2_##x
#endif
#endif

#ifdef DESC_I8
#define I8(x) x##_i8 
#else
#define I8(x) x 
#endif

#define INIT_SCALARS() sum = zero; s = zero; range = lowr; sc = scalelow; scinv = 1.0d0 / sc; y = big/dsqrt(dble(array_size)); naninput = .false.; return_inf = .false.;

! 
!  The algorithm used is based on the following paper:
! 
!  ACM Transactions on Mathematical Software, Vol 44, No. 3, Article 24.
!  Publication date: December 2017, Richard J. Hanson, Tim Hopkins 2017.  
!  Remark on Algorithm 539:  A Modern Fortran Reference Implementation
!  for Carefully Computing the Euclidean Norm.  Retrieved from
!  https://dl.acm.org/citation.cfm?id=3134441

! norm2 routines for each source array rank value (1-7) have two variants :
! A. real(4) versions : for each rank have only one case for both fast, relaxed and precise 
!    calculations which is to do the them in real(8) precision. 
! B. real(8) versions : have two separate cases for fast(relaxed also falls here till a new case
!    is introduced). 'pfr' passed in by the compiler indicates which of two cases is chosen. 
!    In case of fast (pfr = 1), the code is same as the real(4) version of routine.
!    In case of precise (pfr = 2), algorithm described in the paper is broken down as below :
!     1. Save the IEEE underflow flag state
!     2. Set IEEE halting mode
!     3. Calculate compensated sum of squares over all rows. Here, those values which carry over
!        from one loop iteration to another are passed in as arguments to the 
!        compensated_sum_of_squares() routine, since the sum needed to be calculated over multiple rows
!     4. get the square root of sum calculated in the last step
!     5. Check for overflow or underflow
!     6. Based on occurance of underflow, nans, infinity chose to calculate a precise sum of squares 
!        over all the rows instead
!     7. Whilst calculating the precise sum, if the value is ever inf jump out of the loop else finish it
!     8. get square root of precise sum
!     9. Calculate the final result from based on existance of nans in source array or infinity generated 
!        during the calculations
!     10. Restore the IEEE underflow flag state

! module I8(__norm2) has utility functions and compile time constants that are used later on by routines
! that calculate norm of arrays of each rank value (1-7) and data type (real(4) & real(8)) for the dim
! and nodim versions

module I8(__norm2)
!     .. Parameters ..
  real(8) :: zero
  parameter (zero=0.0D+0)
  real(8) :: one
  parameter (one=1.0D+0)
!     ..
  integer(8) :: lowr, mediumr, highr
  parameter (lowr=0, mediumr=1, highr=2)

  real(8) :: low, big, scalehigh, scalelow
  parameter (low=6.7178761075670888D-139)
  parameter (big=1.3407807929942597D+154)
  parameter (scalehigh=1.1113793747425387D-162)
  parameter (scalelow=3.0191699398572331D+169)
  contains

! incoming arrays need not be stride-1 assumed shape arrays
!pgi$g -Hx,54,2
!pgi$g -Mx,54,2

! calculate sum of squares of elements of a single dimensional real(4) array
! at real(8) precision, it is called by routines that calculate norm2 of
! arrays with rank 1 to 7
  function sum_of_squares_real4(array) result(res)
    implicit none

    real(4) :: array(:)
    real(8) :: val, res 
    integer(8) :: i, array_lbnd(1), array_ubnd(1)

    array_lbnd = lbound(array)
    array_ubnd = ubound(array)
    res = zero
    do i = array_lbnd(1), array_ubnd(1)
      val = dble(array(i))
      res = res + (val * val)
    end do
  end function sum_of_squares_real4

! calculate sum of squares of elements of a single dimensional real(8) array
! at REAL(8) precision, it is called by routines that calculate norm2 of
! arrays with rank 1 to 7
  function sum_of_squares_real8(array) result(res)
    implicit none

    real(8) :: array(:), res
    integer(8) :: i, array_lbnd(1), array_ubnd(1)

    array_lbnd = lbound(array)
    array_ubnd = ubound(array)
    res = zero
    do i = array_lbnd(1), array_ubnd(1)
      res = res + (array(i) * array(i))
    end do
  end function sum_of_squares_real8

! calculate norm2() of a single real(4) dimensional array. It is called by 
! routines of the form norm2(array, dim) where arrays rank >= 1 and <= 7 
! and dim <= rank.
  function norm_real4(array) result(res)
    implicit none

    real(4) :: array(:), res
    real(8) :: sum, dres
    integer(8) :: start, array_size

    if (is_contiguous(array)) then
      start = loc(array(:))
      array_size = size(array)
      call I8(stride_1_norm2_real4) (start, array_size, res)
    else
      sum = sum_of_squares_real4 (array)
      dres = dsqrt(sum)
      res = real(dres)
    endif
  end function norm_real4

! calculate norm2() of a single real(8) dimensional array. It is called by 
! routines of the form norm2(array, dim) where arrays rank >= 1 and <= 7 
! and dim <= rank.
! The handling of fast, relaxed and precise cases based on compiler supplied 
! 'pfr' is described later
  function norm_real8(array, pfr) result(res)
    implicit none

    real(8) :: array(:), res, sum, sc, s, scinv, y
    integer(8) :: start, i, array_size, array_lbnd(1), array_ubnd(1), range
    integer(4) :: pfr
    logical :: unf_save, unf, ovf_halting, unf_halting, ovf, return_inf, naninput

    select case (pfr)
      case (1)
        if (is_contiguous(array)) then
          start = loc(array)
          array_size = size(array)
          call I8(stride_1_norm2_real8) (start, array_size, res)
        else
          sum = sum_of_squares_real8 (array)
          res = dsqrt(sum)
        endif
      case (2)
        !add more constraints here
        !placeholder for precise function
        sum = zero
        array_lbnd = lbound(array)
        array_ubnd = ubound(array)
        array_size = size(array)

        call save_unf_state(unf_save, unf)
        call set_halting_mode(ovf_halting, unf_halting)
        sum = zero; s = zero;
        call compensated_sum_of_squares(array, s, sum)
        res = dsqrt(sum)
        call check_ovf_unf(ovf, ovf_halting, unf, unf_halting)

        if (use_precise_method(unf, res)) then
          INIT_SCALARS()
          call precise_sum_of_squares (array, sum, sc, s, scinv, range, y, return_inf, naninput)
          if (return_inf) then
            goto 100
          endif

100 continue
          res = res_under_exceptions (naninput, return_inf, sum, scinv)
        endif

        call restore_unf_state(unf_save)
    end select
  end function norm_real8

  subroutine save_unf_state (unf_save, unf)
    use ieee_arithmetic
    use ieee_exceptions
    implicit none

    logical :: unf_save, unf
    unf_save = .false.
    call ieee_get_flag(ieee_underflow, unf)
    if (unf .eqv. .true.) then
      unf_save = unf
      call ieee_set_flag(ieee_underflow, .false.)
    endif
  end subroutine save_unf_state

  subroutine restore_unf_state (unf_save)
    use ieee_arithmetic
    use ieee_exceptions
    implicit none

    logical :: unf_save

    if (unf_save .eqv. .true.) then
      call ieee_set_flag(ieee_underflow,.true.)
    endif
  end subroutine restore_unf_state

!  Mask overflow and underflow exceptions if they are enabled. This is
!  due to the fact that the "fast" algorithm could create spurious overflows
!  or underflows based on the input data.
  subroutine set_halting_mode (ovf_halting, unf_halting)
    use ieee_arithmetic
    use ieee_exceptions
    implicit none

    logical :: ovf_halting, unf_halting

    call ieee_get_halting_mode(ieee_overflow, ovf_halting)
    if (ovf_halting) then
       call ieee_set_halting_mode(ieee_overflow, .false.)
    end if
    call ieee_get_halting_mode(ieee_underflow, unf_halting)
    if (unf_halting) then
       call ieee_set_halting_mode(ieee_underflow, .false.)
    end if            
  end subroutine set_halting_mode

  subroutine check_ovf_unf (ovf, ovf_halting, unf, unf_halting)
    use ieee_arithmetic
    use ieee_exceptions
    implicit none

    logical :: ovf, unf, ovf_halting, unf_halting

    ! Check for overflows
    call ieee_get_flag(ieee_overflow, ovf)
    if (ovf) then
       call ieee_set_flag(ieee_overflow, .false.)
    end if
    if (ovf_halting) then
       call ieee_set_halting_mode(ieee_overflow, .true.)
    end if

    ! Check for underflows
    call ieee_get_flag(ieee_underflow,unf)
    if (unf) then
       call ieee_set_flag(ieee_underflow,.false.)
    end if
    if (unf_halting) then
       call ieee_set_halting_mode(ieee_underflow, .true.)
    end if
  end subroutine check_ovf_unf

! Calculate "compensated sum of squares" as described in the paper
  subroutine compensated_sum_of_squares(array, s, sum)
    implicit none

    real(8) :: array(:), s, sum, t
    integer(8) :: i, array_lbnd(1), array_ubnd(1)

    array_lbnd = lbound(array)
    array_ubnd = ubound(array)

    do i = array_lbnd(1), array_ubnd(1)
      s = s + (array(i) * array(i))
      t = sum
      sum = t + s
      s = s + (t - sum) 
    enddo    
  end subroutine compensated_sum_of_squares

! The paper describes a precise version of norm2 for a single dimensional array
! However, we have to support multidimensional arrays (even discontiguous).  
! So break down the algorithm to calculate sum of squares precisely and use this to 
! aggregate the sum over all rows & columns
  subroutine precise_sum_of_squares (array, sum, sc, s, scinv, range, y, return_inf, naninput)
    use ieee_arithmetic
    use ieee_exceptions
    implicit none

    ! .. Arguments ..
    real(8) :: array(:), sum, s, sc, scinv, y
    logical :: naninput, return_inf
    integer(8) :: range

    ! .. Local Scalars ..
    real(8) :: absxi, t
    integer(8) ix, array_size, array_lbnd(1), array_lbdn(1)

    ! .. Intrinsic Functions ..
    intrinsic abs, dsqrt, lbound, ubound

    array_lbnd = lbound(array)
    array_lbdn = ubound(array)

    do ix = array_lbnd(1), array_lbdn(1)
        if (ieee_is_finite(array(ix))) then
            if (.not. naninput) then

                absxi = abs(array(ix))
                if (range .eq. lowr) then
                    if (absxi .le. low) then
                        s = s + ((array(ix) * sc) * (array(ix) * sc))
                    elseif (absxi .ge. big) then
                        sc = scalehigh
                        scinv = 1.0d0 / sc
                        sum = zero
                        s = ((array(ix) * sc) * (array(ix) * sc))
                        range = highr
                    else
                        s = (s * scinv) * scinv + (array(ix) * array(ix))
                        sum = (sum * scinv) * scinv
                        sc = one
                        scinv = one
                        range = mediumr
                    endif
                elseif (range .eq. mediumr) then
                    if (absxi .LT. y) then
                        s = s + (array(ix) * array(ix))
                    else
                       sc = scalehigh
                       scinv = 1.0d0 / sc
                       s = (s * sc) * sc + ((array(ix) * sc) * (array(ix) * sc))
                       sum = (sum * sc) * sc
                       range = highr
                    endif
                else
                    s = s + ((array(ix) * sc) * (array(ix) * sc))
                endif
                t = sum
                sum = t + s
                s = (t - sum) + s
            endif
        else
            if (ieee_is_nan(array(ix))) then
                naninput = .true.
                s = array(ix)
            else
                sum = array(ix) * array(ix)
                return_inf = .true.
                return
            endif
        endif
    enddo
  end subroutine precise_sum_of_squares

! This precise is different from the compiler flag value specified in 'pfr' variable.
! This routine is called only when pfr = 2 (precise), to determine whether to calcluate
! precise_sum_of_squares() and disregard compensated_sum_of_squares() or not
  function use_precise_method (unf, norm) result(res)
    use ieee_arithmetic
    use ieee_exceptions
    implicit none

    logical :: res, unf
    real(8) :: norm
    
    res = (unf .eqv. .true.) .or. (ieee_is_nan(norm)) .or. (norm .eq. ieee_value(norm, ieee_positive_inf))
  end function use_precise_method 

! Pick the correct result based on generated exceptions
  function res_under_exceptions(naninput, return_inf, sum, scinv) result(res)
    implicit none

    real(8) ::  sum, scinv, res
    logical :: naninput, return_inf
    if ((naninput) .OR. (return_inf)) then
      res = sum
    else
      res = dsqrt(sum) * scinv
    endif      
  end function res_under_exceptions
end module I8(__norm2)

! The next few routines actually handle nodim version of norm2() calls
! for each rank of real(4) and real(8) arrays. Refer the algorithm 
! described at the top of the file to understand the handling of precise
! case in nodim variant of norm2()

subroutine F90(nodim_1_real4) (res, array)
  use I8(__norm2)
  implicit none

  real(4) :: array(:), res, sum
  integer(8) :: start, array_size

  res = norm_real4(array)
end subroutine F90(nodim_1_real4)

subroutine F90(nodim_1_real8) (res, array, pfr)
  use I8(__norm2)
  implicit none

  real(8) :: array(:), res, sum
  integer(8) :: start, array_size
  integer(4) :: pfr

  res = norm_real8(array, pfr)
end subroutine F90(nodim_1_real8)

subroutine F90(nodim_2_real4) (res, array)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:), res
  real(8) :: sum, dres
  integer(8) :: start, i, array_size, array_lbnd(2), array_ubnd(2)

  if (is_contiguous(array)) then
    start = loc(array(:,:))
    array_size = size(array)
    call I8(stride_1_norm2_real4) (start, array_size, res)
  else
    sum = zero
    array_lbnd = lbound(array)
    array_ubnd = ubound(array)

    do i = array_lbnd(1), array_ubnd(1)
      sum = sum + sum_of_squares_real4 (array(i,:))
    end do
    dres = dsqrt(sum)
    res = real(dres)
  endif
end subroutine F90(nodim_2_real4)

subroutine F90(nodim_2_real8) (res, array, pfr)
  use I8(__norm2)
  use ieee_arithmetic
  use ieee_exceptions
  implicit none

  real(8) :: array(:,:), res, sum, sc, s, scinv, y
  integer(8) :: start, i, array_size, array_lbnd(2), array_ubnd(2), range
  integer(4) :: pfr
  logical :: unf_save, unf, ovf_halting, unf_halting, ovf, return_inf, naninput

  if (size(array) .eq. 1) then
    array_lbnd = lbound(array)
    res = array(array_lbnd(1), array_lbnd(2))
  else
    select case (pfr)
      case (1)
        if (is_contiguous(array)) then
          start = loc(array(:,:))
          array_size = size(array)
          call I8(stride_1_norm2_real8) (start, array_size, res)
        else
          sum = zero
          array_lbnd = lbound(array)
          array_ubnd = ubound(array)

          do i = array_lbnd(1), array_ubnd(1)
            sum = sum + sum_of_squares_real8 (array(i,:))
          end do
          res = dsqrt(sum)
        endif
      case (2)
        !add more constraints here
        !placeholder for precise function
        sum = zero
        array_lbnd = lbound(array)
        array_ubnd = ubound(array)
        array_size = size(array)

        call save_unf_state(unf_save, unf)
        call set_halting_mode(ovf_halting, unf_halting)
        sum = zero; s = zero;
        do i = array_lbnd(1), array_ubnd(1)
          call compensated_sum_of_squares(array(i,:), s, sum)
        end do
        res = dsqrt(sum)
        call check_ovf_unf(ovf, ovf_halting, unf, unf_halting)

        if (use_precise_method(unf, res)) then
          INIT_SCALARS()
          do i = array_lbnd(1), array_ubnd(1)
            call precise_sum_of_squares (array(i,:), sum, sc, s, scinv, range, y, return_inf, naninput)
            if (return_inf) then
              goto 100
            endif
          end do

100 continue
          res = res_under_exceptions (naninput, return_inf, sum, scinv)
        endif

        call restore_unf_state(unf_save)
    end select
  endif
end subroutine F90(nodim_2_real8)

subroutine F90(nodim_3_real4) (res, array)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:), res
  real(8) :: sum, dres
  integer(8) :: start, i, j, array_size, array_lbnd(3), array_ubnd(3)

  if (is_contiguous(array)) then
    start = loc(array(:,:,:))
    array_size = size(array)
    call I8(stride_1_norm2_real4) (start, array_size, res)
  else
    sum = zero
    array_lbnd = lbound(array)
    array_ubnd = ubound(array)

    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        sum = sum + sum_of_squares_real4 (array(i,j,:))
      end do
    end do
    dres = dsqrt(sum)
    res = real(dres)
  endif
end subroutine F90(nodim_3_real4)

subroutine F90(nodim_3_real8) (res, array, pfr)
  use I8(__norm2)
  use ieee_arithmetic
  use ieee_exceptions
  implicit none

  real(8) :: array(:,:,:), res, sum, sc, s, scinv, y
  integer(8) :: start, i, j, array_size, array_lbnd(3), array_ubnd(3), range
  integer(4) :: pfr
  logical :: unf_save, unf, ovf_halting, unf_halting, ovf, return_inf, naninput

  select case (pfr)
    case (1)
      if (is_contiguous(array)) then
        start = loc(array(:,:,:))
        array_size = size(array)
        call I8(stride_1_norm2_real8) (start, array_size, res)
      else
        sum = zero
        array_lbnd = lbound(array)
        array_ubnd = ubound(array)

        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            sum = sum + sum_of_squares_real8 (array(i,j,:))
          end do
        end do
        res = dsqrt(sum)
      endif
    case (2)
      !add more constraints here
      !placeholder for precise function
      sum = zero
      array_lbnd = lbound(array)
      array_ubnd = ubound(array)
      array_size = size(array)

      call save_unf_state(unf_save, unf)
      call set_halting_mode(ovf_halting, unf_halting)
      sum = zero; s = zero;
      do i = array_lbnd(1), array_ubnd(1)
        do j = array_lbnd(2), array_ubnd(2)
          call compensated_sum_of_squares(array(i,j,:), s, sum)
        end do
      end do
      res = dsqrt(sum)
      call check_ovf_unf(ovf, ovf_halting, unf, unf_halting)

      if (use_precise_method(unf, res)) then
        INIT_SCALARS()
        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            call precise_sum_of_squares (array(i,j,:), sum, sc, s, scinv, range, y, return_inf, naninput)
            if (return_inf) then
              goto 100
            endif
          end do
        end do

100 continue
        res = res_under_exceptions (naninput, return_inf, sum, scinv)
      endif

      call restore_unf_state(unf_save)
  end select
end subroutine F90(nodim_3_real8)

subroutine F90(nodim_4_real4) (res, array)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:,:), res
  real(8) :: sum, dres
  integer(8) :: start, i, j, k, array_size, array_lbnd(4), array_ubnd(4)

  if (is_contiguous(array)) then
    start = loc(array(:,:,:,:))
    array_size = size(array)
    call I8(stride_1_norm2_real4) (start, array_size, res)
  else
    sum = zero
    array_lbnd = lbound(array)
    array_ubnd = ubound(array)

    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          sum = sum + sum_of_squares_real4 (array(i,j,k,:))
        end do
      end do
    end do
    dres = dsqrt(sum)
    res = real(dres)
  endif
end subroutine F90(nodim_4_real4)

subroutine F90(nodim_4_real8) (res, array, pfr)
  use I8(__norm2)
  use ieee_arithmetic
  use ieee_exceptions
  implicit none

  real(8) :: array(:,:,:,:), res, sum, sc, s, scinv, y
  integer(8) :: start, i, j, k, array_size, array_lbnd(4), array_ubnd(4), range
  integer(4) :: pfr
  logical :: unf_save, unf, ovf_halting, unf_halting, ovf, return_inf, naninput

  select case (pfr)
    case (1)
      if (is_contiguous(array)) then
        start = loc(array(:,:,:,:))
        array_size = size(array)
        call I8(stride_1_norm2_real8) (start, array_size, res)
      else
        sum = zero
        array_lbnd = lbound(array)
        array_ubnd = ubound(array)

        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            do k = array_lbnd(3), array_ubnd(3)
              sum = sum + sum_of_squares_real8 (array(i,j,k,:))
            end do
          end do
        end do
        res = dsqrt(sum)
      endif
    case (2)
      !add more constraints here
      !placeholder for precise function
      sum = zero
      array_lbnd = lbound(array)
      array_ubnd = ubound(array)
      array_size = size(array)

      call save_unf_state(unf_save, unf)
      call set_halting_mode(ovf_halting, unf_halting)
      sum = zero; s = zero;
      do i = array_lbnd(1), array_ubnd(1)
        do j = array_lbnd(2), array_ubnd(2)
          do k = array_lbnd(3), array_ubnd(3)
            call compensated_sum_of_squares(array(i,j,k,:), s, sum)
          end do
        end do
      end do
      res = dsqrt(sum)
      call check_ovf_unf(ovf, ovf_halting, unf, unf_halting)

      if (use_precise_method(unf, res)) then
        INIT_SCALARS()
        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            do k = array_lbnd(3), array_ubnd(3)
              call precise_sum_of_squares (array(i,j,k,:), sum, sc, s, scinv, range, y, return_inf, naninput)
              if (return_inf) then
                goto 100
              endif
            end do
          end do
        end do

100 continue
        res = res_under_exceptions (naninput, return_inf, sum, scinv)
      endif

      call restore_unf_state(unf_save)
  end select
end subroutine F90(nodim_4_real8)

subroutine F90(nodim_5_real4) (res, array)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:,:,:), res
  real(8) :: sum, dres
  integer(8) :: start, i, j, k, l, array_size, array_lbnd(5), array_ubnd(5)

  if (is_contiguous(array)) then
    start = loc(array(:,:,:,:,:))
    array_size = size(array)
    call I8(stride_1_norm2_real4) (start, array_size, res)
  else
    sum = zero
    array_lbnd = lbound(array)
    array_ubnd = ubound(array)

    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            sum = sum + sum_of_squares_real4 (array(i,j,k,l,:))
          end do
        end do
      end do
    end do
    dres = dsqrt(sum)
    res = real(dres)
  endif
end subroutine F90(nodim_5_real4)

subroutine F90(nodim_5_real8) (res, array, pfr)
  use I8(__norm2)
  use ieee_arithmetic
  use ieee_exceptions
  implicit none

  real(8) :: array(:,:,:,:,:), res, sum, sc, s, scinv, y
  integer(8) :: start, i, j, k, l, array_size, array_lbnd(5), array_ubnd(5), range
  integer(4) :: pfr
  logical :: unf_save, unf, ovf_halting, unf_halting, ovf, return_inf, naninput

  select case (pfr)
    case (1)
      if (is_contiguous(array)) then
        start = loc(array(:,:,:,:,:))
        array_size = size(array)
        call I8(stride_1_norm2_real8) (start, array_size, res)
      else
        sum = zero
        array_lbnd = lbound(array)
        array_ubnd = ubound(array)

        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            do k = array_lbnd(3), array_ubnd(3)
              do l = array_lbnd(4), array_ubnd(4)
                sum = sum + sum_of_squares_real8 (array(i,j,k,l,:))
              end do
            end do
          end do
        end do
        res = dsqrt(sum)
      endif
    case (2)
      !add more constraints here
      !placeholder for precise function
      sum = zero
      array_lbnd = lbound(array)
      array_ubnd = ubound(array)
      array_size = size(array)

      call save_unf_state(unf_save, unf)
      call set_halting_mode(ovf_halting, unf_halting)
      sum = zero; s = zero;
      do i = array_lbnd(1), array_ubnd(1)
        do j = array_lbnd(2), array_ubnd(2)
          do k = array_lbnd(3), array_ubnd(3)
            do l = array_lbnd(4), array_ubnd(4)
              call compensated_sum_of_squares(array(i,j,k,l,:), s, sum)
            end do
          end do
        end do
      end do
      res = dsqrt(sum)
      call check_ovf_unf(ovf, ovf_halting, unf, unf_halting)

      if (use_precise_method(unf, res)) then
        INIT_SCALARS()
        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            do k = array_lbnd(3), array_ubnd(3)
              do l = array_lbnd(4), array_ubnd(4)
                call precise_sum_of_squares (array(i,j,k,l,:), sum, sc, s, scinv, range, y, return_inf, naninput)
                if (return_inf) then
                  goto 100
                endif
              end do
            end do
          end do
        end do

100 continue
        res = res_under_exceptions (naninput, return_inf, sum, scinv)
      endif

      call restore_unf_state(unf_save)
  end select
end subroutine F90(nodim_5_real8)

subroutine F90(nodim_6_real4) (res, array)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:,:,:,:), res
  real(8) :: sum, dres
  integer(8) :: start, i, j, k, l, m, array_size, array_lbnd(6), array_ubnd(6)

  if (is_contiguous(array)) then
    start = loc(array(:,:,:,:,:,:))
    array_size = size(array)
    call I8(stride_1_norm2_real4) (start, array_size, res)
  else
    sum = zero
    array_lbnd = lbound(array)
    array_ubnd = ubound(array)

    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(5), array_ubnd(5)
              sum = sum + sum_of_squares_real4 (array(i,j,k,l,m,:))
            end do
          end do
        end do
      end do
    end do
    dres = dsqrt(sum)
    res = real(dres)
  endif
end subroutine F90(nodim_6_real4)

subroutine F90(nodim_6_real8) (res, array, pfr)
  use I8(__norm2)
  use ieee_arithmetic
  use ieee_exceptions
  implicit none

  real(8) :: array(:,:,:,:,:,:), res, sum, sc, s, scinv, y
  integer(8) :: start, i, j, k, l, m, array_size, array_lbnd(6), array_ubnd(6), range
  integer(4) :: pfr
  logical :: unf_save, unf, ovf_halting, unf_halting, ovf, return_inf, naninput

  select case (pfr)
    case (1)
      if (is_contiguous(array)) then
        start = loc(array(:,:,:,:,:,:))
        array_size = size(array)
        call I8(stride_1_norm2_real8) (start, array_size, res)
      else
        sum = zero
        array_lbnd = lbound(array)
        array_ubnd = ubound(array)

        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            do k = array_lbnd(3), array_ubnd(3)
              do l = array_lbnd(4), array_ubnd(4)
                do m = array_lbnd(5), array_ubnd(5)
                  sum = sum + sum_of_squares_real8 (array(i,j,k,l,m,:))
                end do
              end do
            end do
          end do
        end do
        res = dsqrt(sum)
      endif
    case (2)
      !add more constraints here
      !placeholder for precise function
      sum = zero
      array_lbnd = lbound(array)
      array_ubnd = ubound(array)
      array_size = size(array)

      call save_unf_state(unf_save, unf)
      call set_halting_mode(ovf_halting, unf_halting)
      sum = zero; s = zero;
      do i = array_lbnd(1), array_ubnd(1)
        do j = array_lbnd(2), array_ubnd(2)
          do k = array_lbnd(3), array_ubnd(3)
            do l = array_lbnd(4), array_ubnd(4)
              do m = array_lbnd(5), array_ubnd(5)
                call compensated_sum_of_squares(array(i,j,k,l,m,:), s, sum)
              end do
            end do
          end do
        end do
      end do
      res = dsqrt(sum)
      call check_ovf_unf(ovf, ovf_halting, unf, unf_halting)

      if (use_precise_method(unf, res)) then
        INIT_SCALARS()
        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            do k = array_lbnd(3), array_ubnd(3)
              do l = array_lbnd(4), array_ubnd(4)
                do m = array_lbnd(5), array_ubnd(5)
                  call precise_sum_of_squares (array(i,j,k,l,m,:), sum, sc, s, scinv, range, y, return_inf, naninput)
                  if (return_inf) then
                    goto 100
                  endif
                end do
              end do
            end do
          end do
        end do        

100 continue
        res = res_under_exceptions (naninput, return_inf, sum, scinv)
      endif

      call restore_unf_state(unf_save)
  end select
end subroutine F90(nodim_6_real8)

subroutine F90(nodim_7_real4) (res, array)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:,:,:,:,:), res
  real(8) :: sum, dres
  integer(8) :: start, i, j, k, l, m, n, array_size, array_lbnd(7), array_ubnd(7)

  if (is_contiguous(array)) then
    start = loc(array(:,:,:,:,:,:,:))
    array_size = size(array)
    call I8(stride_1_norm2_real4) (start, array_size, res)
  else
    sum = zero
    array_lbnd = lbound(array)
    array_ubnd = ubound(array)

    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(5), array_ubnd(5)
              do n = array_lbnd(6), array_ubnd(6)
                sum = sum + sum_of_squares_real4 (array(i,j,k,l,m,n,:))
              end do
            end do
          end do
        end do
      end do
    end do
    dres = dsqrt(sum)
    res = real(dres)
  endif
end subroutine F90(nodim_7_real4)

subroutine F90(nodim_7_real8) (res, array, pfr)
  use I8(__norm2)
  use ieee_arithmetic
  use ieee_exceptions
  implicit none

  real(8) :: array(:,:,:,:,:,:,:), res, sum, sc, s, scinv, y
  integer(8) :: start, i, j, k, l, m, n, array_size, array_lbnd(7), array_ubnd(7), range
  integer(4) :: pfr
  logical :: unf_save, unf, ovf_halting, unf_halting, ovf, return_inf, naninput

  select case (pfr)
    case (1)
      if (is_contiguous(array)) then
        start = loc(array(:,:,:,:,:,:,:))
        array_size = size(array)
        call I8(stride_1_norm2_real8) (start, array_size, res)
      else
        sum = zero
        array_lbnd = lbound(array)
        array_ubnd = ubound(array)

        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            do k = array_lbnd(3), array_ubnd(3)
              do l = array_lbnd(4), array_ubnd(4)
                do m = array_lbnd(5), array_ubnd(5)
                  do n = array_lbnd(6), array_ubnd(6)
                    sum = sum + sum_of_squares_real8 (array(i,j,k,l,m,n,:))
                  end do
                end do
              end do
            end do
          end do
        end do
        res = dsqrt(sum)
      endif
    case (2)
      !add more constraints here
      !placeholder for precise function
      sum = zero
      array_lbnd = lbound(array)
      array_ubnd = ubound(array)
      array_size = size(array)

      call save_unf_state(unf_save, unf)
      call set_halting_mode(ovf_halting, unf_halting)
      sum = zero; s = zero;
      do i = array_lbnd(1), array_ubnd(1)
        do j = array_lbnd(2), array_ubnd(2)
          do k = array_lbnd(3), array_ubnd(3)
            do l = array_lbnd(4), array_ubnd(4)
              do m = array_lbnd(5), array_ubnd(5)
                do n = array_lbnd(6), array_ubnd(6)
                  call compensated_sum_of_squares(array(i,j,k,l,m,n,:), s, sum)
                end do
              end do
            end do
          end do
        end do
      end do
      res = dsqrt(sum)
      call check_ovf_unf(ovf, ovf_halting, unf, unf_halting)

      if (use_precise_method(unf, res)) then
        INIT_SCALARS()
        do i = array_lbnd(1), array_ubnd(1)
          do j = array_lbnd(2), array_ubnd(2)
            do k = array_lbnd(3), array_ubnd(3)
              do l = array_lbnd(4), array_ubnd(4)
                do m = array_lbnd(5), array_ubnd(5)
                  do n = array_lbnd(6), array_ubnd(6)
                    call precise_sum_of_squares (array(i,j,k,l,m,n,:), sum, sc, s, scinv, range, y, return_inf, naninput)
                    if (return_inf) then
                      goto 100
                    endif
                  end do
                end do
              end do
            end do
          end do
        end do        

100 continue
        res = res_under_exceptions (naninput, return_inf, sum, scinv)
      endif

      call restore_unf_state(unf_save)
  end select 
end subroutine F90(nodim_7_real8)

! The next few routines actually handle dim version of norm2() calls
! for each rank of real(4) and real(8) arrays

subroutine F90(dim_2_real4) (res, array, dim)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:), res(:)
  integer(4) :: dim
  integer(8) :: i
  integer(8) :: array_lbnd(2), array_ubnd(2)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      res(i) = norm_real4(array(:,i))
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      res(i) = norm_real4(array(i,:))
    end do
  endif
end subroutine F90(dim_2_real4)

subroutine F90(dim_2_real8) (res, array, pfr, dim)
  use I8(__norm2)
  implicit none

  real(8) :: array(:,:), res(:)
  integer(4) :: dim, pfr
  integer(8) :: i
  integer(8) :: array_lbnd(2), array_ubnd(2)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      res(i) = norm_real8(array(:,i), pfr)
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      res(i) = norm_real8(array(i,:), pfr)
    end do
  endif
end subroutine F90(dim_2_real8)

subroutine F90(dim_3_real4) (res, array, dim)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:), res(:,:)
  integer(4) :: dim
  integer(8) :: i, j
  integer(8) :: array_lbnd(3), array_ubnd(3)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        res(i,j) = norm_real4(array(:,i,j))
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        res(i,j) = norm_real4(array(i,:,j))
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        res(i,j) = norm_real4(array(i,j,:))
      end do
    end do
  endif
end subroutine F90(dim_3_real4)

subroutine F90(dim_3_real8) (res, array, pfr, dim)
  use I8(__norm2)
  implicit none

  real(8) :: array(:,:,:), res(:,:)
  integer(4) :: dim, pfr
  integer(8) :: i, j
  integer(8) :: array_lbnd(3), array_ubnd(3)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        res(i,j) = norm_real8(array(:,i,j), pfr)
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        res(i,j) = norm_real8(array(i,:,j), pfr)
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        res(i,j) = norm_real8(array(i,j,:), pfr)
      end do
    end do
  endif
end subroutine F90(dim_3_real8)

subroutine F90(dim_4_real4) (res, array, dim)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:,:), res(:,:,:)
  integer(4) :: dim
  integer(8) :: i, j, k
  integer(8) :: array_lbnd(4), array_ubnd(4)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          res(i,j,k) = norm_real4(array(:,i,j,k))
        end do
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          res(i,j,k) = norm_real4(array(i,:,j,k))
        end do
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(4), array_ubnd(4)
          res(i,j,k) = norm_real4(array(i,j,:,k))
        end do
      end do
    end do
  else if (dim .eq. 4) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          res(i,j,k) = norm_real4(array(i,j,k,:))
        end do
      end do
    end do
  endif
end subroutine F90(dim_4_real4)

subroutine F90(dim_4_real8) (res, array, pfr, dim)
  use I8(__norm2)
  implicit none

  real(8) :: array(:,:,:,:), res(:,:,:)
  integer(4) :: dim, pfr
  integer(8) :: i, j, k
  integer(8) :: array_lbnd(4), array_ubnd(4)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          res(i,j,k) = norm_real8(array(:,i,j,k), pfr)
        end do
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          res(i,j,k) = norm_real8(array(i,:,j,k), pfr)
        end do
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(4), array_ubnd(4)
          res(i,j,k) = norm_real8(array(i,j,:,k), pfr)
        end do
      end do
    end do
  else if (dim .eq. 4) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          res(i,j,k) = norm_real8(array(i,j,k,:), pfr)
        end do
      end do
    end do
  endif
end subroutine F90(dim_4_real8)

subroutine F90(dim_5_real4) (res, array, dim)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:,:,:), res(:,:,:,:)
  integer(4) :: dim
  integer(8) :: i, j, k, l
  integer(8) :: array_lbnd(5), array_ubnd(5)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            res(i,j,k,l) = norm_real4(array(:,i,j,k,l))
          end do
        end do
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            res(i,j,k,l) = norm_real4(array(i,:,j,k,l))
          end do
        end do
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            res(i,j,k,l) = norm_real4(array(i,j,:,k,l))
          end do
        end do
      end do
    end do
  else if (dim .eq. 4) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(5), array_ubnd(5)
            res(i,j,k,l) = norm_real4(array(i,j,k,:,l))
          end do
        end do
      end do
    end do
  else if (dim .eq. 5) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            res(i,j,k,l) = norm_real4(array(i,j,k,l,:))
          end do
        end do
      end do
    end do
  endif
end subroutine F90(dim_5_real4)

subroutine F90(dim_5_real8) (res, array, pfr, dim)
  use I8(__norm2)
  implicit none

  real(8) :: array(:,:,:,:,:), res(:,:,:,:)
  integer(4) :: dim, pfr
  integer(8) :: i, j, k, l
  integer(8) :: array_lbnd(5), array_ubnd(5)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            res(i,j,k,l) = norm_real8(array(:,i,j,k,l), pfr)
          end do
        end do
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            res(i,j,k,l) = norm_real8(array(i,:,j,k,l), pfr)
          end do
        end do
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            res(i,j,k,l) = norm_real8(array(i,j,:,k,l), pfr)
          end do
        end do
      end do
    end do
  else if (dim .eq. 4) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(5), array_ubnd(5)
            res(i,j,k,l) = norm_real8(array(i,j,k,:,l), pfr)
          end do
        end do
      end do
    end do
  else if (dim .eq. 5) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            res(i,j,k,l) = norm_real8(array(i,j,k,l,:), pfr)
          end do
        end do
      end do
    end do
  endif
end subroutine F90(dim_5_real8)

subroutine F90(dim_6_real4) (res, array, dim)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:,:,:,:), res(:,:,:,:,:)
  integer(4) :: dim
  integer(8) :: i, j, k, l, m
  integer(8) :: array_lbnd(6), array_ubnd(6)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real4(array(:,i,j,k,l,m))
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real4(array(i,:,j,k,l,m))
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real4(array(i,j,:,k,l,m))
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 4) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real4(array(i,j,k,:,l,m))
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 5) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real4(array(i,j,k,l,:,m))
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 6) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(5), array_ubnd(5)
              res(i,j,k,l,m) = norm_real4(array(i,j,k,l,m,:))
            end do
          end do
        end do
      end do
    end do
  endif
end subroutine F90(dim_6_real4)

subroutine F90(dim_6_real8) (res, array, pfr, dim)
  use I8(__norm2)
  implicit none

  real(8) :: array(:,:,:,:,:,:), res(:,:,:,:,:)
  integer(4) :: dim, pfr
  integer(8) :: i, j, k, l, m
  integer(8) :: array_lbnd(6), array_ubnd(6)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real8(array(:,i,j,k,l,m), pfr)
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real8(array(i,:,j,k,l,m), pfr)
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real8(array(i,j,:,k,l,m), pfr)
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 4) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real8(array(i,j,k,:,l,m), pfr)
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 5) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(6), array_ubnd(6)
              res(i,j,k,l,m) = norm_real8(array(i,j,k,l,:,m), pfr)
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 6) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(5), array_ubnd(5)
              res(i,j,k,l,m) = norm_real8(array(i,j,k,l,m,:), pfr)
            end do
          end do
        end do
      end do
    end do
  endif
end subroutine F90(dim_6_real8)

subroutine F90(dim_7_real4) (res, array, dim)
  use I8(__norm2)
  implicit none

  real(4) :: array(:,:,:,:,:,:,:), res(:,:,:,:,:,:)
  integer(4) :: dim
  integer(8) :: i, j, k, l, m, n
  integer(8) :: array_lbnd(7), array_ubnd(7)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real4(array(:,i,j,k,l,m,n))
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real4(array(i,:,j,k,l,m,n))
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real4(array(i,j,:,k,l,m,n))
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 4) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real4(array(i,j,k,:,l,m,n))
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 5) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real4(array(i,j,k,l,:,m,n))
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 6) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(5), array_ubnd(5)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real4(array(i,j,k,l,m,:,n))
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 7) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(5), array_ubnd(5)
              do n = array_lbnd(6), array_ubnd(6)
                res(i,j,k,l,m,n) = norm_real4(array(i,j,k,l,m,n,:))
              end do
            end do
          end do
        end do
      end do
    end do
  endif
end subroutine F90(dim_7_real4)

subroutine F90(dim_7_real8) (res, array, pfr, dim)
  use I8(__norm2)
  implicit none

  real(8) :: array(:,:,:,:,:,:,:), res(:,:,:,:,:,:)
  integer(4) :: dim, pfr
  integer(8) :: i, j, k, l, m, n
  integer(8) :: array_lbnd(7), array_ubnd(7)

  array_lbnd = lbound(array)
  array_ubnd = ubound(array)

  if (dim .eq. 1) then
    do i = array_lbnd(2), array_ubnd(2)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real8(array(:,i,j,k,l,m,n), pfr)
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 2) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(3), array_ubnd(3)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real8(array(i,:,j,k,l,m,n), pfr)
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 3) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(4), array_ubnd(4)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real8(array(i,j,:,k,l,m,n), pfr)
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 4) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(5), array_ubnd(5)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real8(array(i,j,k,:,l,m,n), pfr)
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 5) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(6), array_ubnd(6)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real8(array(i,j,k,l,:,m,n), pfr)
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 6) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(5), array_ubnd(5)
              do n = array_lbnd(7), array_ubnd(7)
                res(i,j,k,l,m,n) = norm_real8(array(i,j,k,l,m,:,n), pfr)
              end do
            end do
          end do
        end do
      end do
    end do
  else if (dim .eq. 7) then
    do i = array_lbnd(1), array_ubnd(1)
      do j = array_lbnd(2), array_ubnd(2)
        do k = array_lbnd(3), array_ubnd(3)
          do l = array_lbnd(4), array_ubnd(4)
            do m = array_lbnd(5), array_ubnd(5)
              do n = array_lbnd(6), array_ubnd(6)
                res(i,j,k,l,m,n) = norm_real8(array(i,j,k,l,m,n,:), pfr)
              end do
            end do
          end do
        end do
      end do
    end do
  endif
end subroutine F90(dim_7_real8)
