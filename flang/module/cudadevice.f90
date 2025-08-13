!===-- module/cudedevice.f90 -----------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! CUDA Fortran procedures available in device subprogram

module cudadevice
  use __cuda_device
  use, intrinsic :: __fortran_builtins, only: dim3 => __builtin_dim3
  use, intrinsic :: __fortran_builtins, only: c_devptr => __builtin_c_devptr
  use, intrinsic :: __fortran_builtins, only: c_devloc => __builtin_c_devloc
implicit none

  ! Synchronization Functions

  interface syncthreads
    procedure :: syncthreads
  end interface

  interface
    attributes(device) integer function syncthreads_and(value)
      integer, value :: value
    end function
  end interface

  interface
    attributes(device) integer function syncthreads_count(value)
      integer, value :: value
    end function
  end interface

  interface
    attributes(device) integer function syncthreads_or(value)
      integer, value :: value
    end function
  end interface

  interface
    attributes(device) subroutine syncwarp(mask)
      integer, value :: mask
    end subroutine
  end interface

  ! Memory Fences

  interface
    attributes(device) subroutine threadfence()
    end subroutine
  end interface

  interface
    attributes(device) subroutine threadfence_block()
    end subroutine
  end interface

  interface
    attributes(device) subroutine threadfence_system()
    end subroutine
  end interface

  ! Math API

  interface __fadd_rn
   attributes(device) real function __fadd_rn(a,b) bind(c, name='__nv_fadd_rn')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface
  
  interface __fadd_rz
   attributes(device) real function __fadd_rz(a,b) bind(c, name='__nv_fadd_rz')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface

  interface
    attributes(device) function __fadd_rd(x, y) bind(c, name='__nv_fadd_rd')
      real, intent(in), value :: x, y
      real :: __fadd_rd
    end function
  end interface

  interface
    attributes(device) function __fadd_ru(x, y) bind(c, name='__nv_fadd_ru')
      real, intent(in), value :: x, y
      real :: __fadd_ru
    end function
  end interface

  interface __fmul_rn
   attributes(device) real function __fmul_rn(a,b) bind(c, name='__nv_fmul_rn')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface

  interface __fmul_rz
   attributes(device) real function __fmul_rz(a,b) bind(c, name='__nv_fmul_rz')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface

  interface __fmul_ru
   attributes(device) real function __fmul_ru(a,b) bind(c, name='__nv_fmul_ru')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface

  interface __fmul_rd
   attributes(device) real function __fmul_rd(a,b) bind(c, name='__nv_fmul_rd')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface

  interface __fmaf_rn
   attributes(device) real function __fmaf_rn(a,b,c) bind(c, name='__nv_fmaf_rn')
  !dir$ ignore_tkr (d) a, (d) b, (d) c
    real, value :: a, b, c
   end function
  end interface

  interface __fmaf_rz
   attributes(device) real function __fmaf_rz(a,b,c) bind(c, name='__nv_fmaf_rz')
  !dir$ ignore_tkr (d) a, (d) b, (d) c
    real, value :: a, b, c
   end function
  end interface
  
  interface __fmaf_ru
   attributes(device) real function __fmaf_ru(a,b,c) bind(c, name='__nv_fmaf_ru')
  !dir$ ignore_tkr (d) a, (d) b, (d) c
    real, value :: a, b, c
   end function
  end interface
  
  interface __fmaf_rd
   attributes(device) real function __fmaf_rd(a,b,c) bind(c, name='__nv_fmaf_rd')
  !dir$ ignore_tkr (d) a, (d) b, (d) c
    real, value :: a, b, c
   end function
  end interface

  interface __frcp_rn
   attributes(device) real function __frcp_rn(a) bind(c, name='__nv_frcp_rn')
  !dir$ ignore_tkr (d) a
    real, value :: a
   end function
  end interface

  interface __frcp_rz
   attributes(device) real function __frcp_rz(a) bind(c, name='__nv_frcp_rz')
  !dir$ ignore_tkr (d) a
    real, value :: a
   end function
  end interface

  interface __frcp_ru
   attributes(device) real function __frcp_ru(a) bind(c, name='__nv_frcp_ru')
  !dir$ ignore_tkr (d) a
    real, value :: a
   end function
  end interface

  interface __frcp_rd
   attributes(device) real function __frcp_rd(a) bind(c, name='__nv_frcp_rd')
  !dir$ ignore_tkr (d) a
    real, value :: a
   end function
  end interface

  interface __fsqrt_rn
   attributes(device) real function __fsqrt_rn(a) bind(c, name='__nv_fsqrt_rn')
  !dir$ ignore_tkr (d) a
    real, value :: a
   end function
  end interface

  interface __fsqrt_rz
   attributes(device) real function __fsqrt_rz(a) bind(c, name='__nv_fsqrt_rz')
  !dir$ ignore_tkr (d) a
    real, value :: a
   end function
  end interface

  interface __fsqrt_ru
   attributes(device) real function __fsqrt_ru(a) bind(c, name='__nv_fsqrt_ru')
  !dir$ ignore_tkr (d) a
    real, value :: a
   end function
  end interface

  interface __fsqrt_rd
   attributes(device) real function __fsqrt_rd(a) bind(c, name='__nv_fsqrt_rd')
  !dir$ ignore_tkr (d) a
    real, value :: a
   end function
  end interface

  interface __fdiv_rn
   attributes(device) real function __fdiv_rn(a,b) bind(c, name='__nv_fdiv_rn')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface

  interface __fdiv_rz
   attributes(device) real function __fdiv_rz(a,b) bind(c, name='__nv_fdiv_rz')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface

  interface __fdiv_ru
   attributes(device) real function __fdiv_ru(a,b) bind(c, name='__nv_fdiv_ru')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface

  interface __fdiv_rd
   attributes(device) real function __fdiv_rd(a,b) bind(c, name='__nv_fdiv_rd')
  !dir$ ignore_tkr (d) a, (d) b
    real, value :: a, b
   end function
  end interface

  interface __dadd_rn
   attributes(device) real(8) function __dadd_rn(a,b) bind(c, name='__nv_dadd_rn')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b
   end function
  end interface

  interface __dadd_rz
   attributes(device) real(8) function __dadd_rz(a,b) bind(c, name='__nv_dadd_rz')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b
   end function
  end interface

  interface __dadd_ru
   attributes(device) real(8) function __dadd_ru(a,b) bind(c, name='__nv_dadd_ru')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b
   end function
  end interface

  interface __dadd_rd
   attributes(device) real(8) function __dadd_rd(a,b) bind(c, name='__nv_dadd_rd')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b
   end function
  end interface

  interface __dmul_rn
   attributes(device) real(8) function __dmul_rn(a,b) bind(c, name='__nv_dmul_rn')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b
   end function
  end interface

  interface __dmul_rz
   attributes(device) real(8) function __dmul_rz(a,b) bind(c, name='__nv_dmul_rz')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b
   end function
  end interface

  interface __dmul_ru
   attributes(device) real(8) function __dmul_ru(a,b) bind(c, name='__nv_dmul_ru')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b
   end function
  end interface

  interface __dmul_rd
   attributes(device) real(8) function __dmul_rd(a,b) bind(c, name='__nv_dmul_rd')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b
   end function
  end interface

  interface __fma_rn
   attributes(device) real(8) function __fma_rn(a,b,c) bind(c, name='__nv_fma_rn')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b, c
   end function
  end interface

  interface __fma_rz
   attributes(device) real(8) function __fma_rz(a,b,c) bind(c, name='__nv_fma_rz')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b, c
   end function
  end interface

  interface __fma_ru
   attributes(device) real(8) function __fma_ru(a,b,c) bind(c, name='__nv_fma_ru')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b, c
   end function
  end interface

  interface __fma_rd
   attributes(device) real(8) function __fma_rd(a,b,c) bind(c, name='__nv_fma_rd')
  !dir$ ignore_tkr (d) a, (d) b
    real(8), value :: a, b, c
   end function
  end interface

  interface rsqrt
    attributes(device) real(4) function rsqrtf(x) bind(c,name='__nv_rsqrtf')
      real(4), value :: x
    end function
    attributes(device) real(8) function rsqrt(x) bind(c,name='__nv_rsqrt')
      real(8), value :: x
    end function
  end interface

  interface __sad
    attributes(device) integer function __sad(i,j,k) bind(c, name='__nv_sad')
      !dir$ ignore_tkr (d) i, (d) j, (d) k
      integer, value :: i,j,k
    end function
  end interface

  interface __usad
    attributes(device) integer function __usad(i,j,k) bind(c, name='__nv_usad')
      !dir$ ignore_tkr (d) i, (d) j, (d) k
      integer, value :: i,j,k
    end function
  end interface
  
  interface signbit
    attributes(device) integer(4) function signbitf(x) bind(c,name='__nv_signbitf')
      real(4), value :: x
    end function
    attributes(device) integer(4) function signbit(x) bind(c,name='__nv_signbitd')
      real(8), value :: x
    end function
  end interface

  interface
    attributes(device) subroutine sincosf(x, y, z) bind(c,name='__nv_sincosf')
      real(4), value :: x
      real(4), device :: y
      real(4), device :: z
    end subroutine
  end interface
  interface
    attributes(device) subroutine sincos(x, y, z) bind(c,name='__nv_sincos')
      real(8), value :: x
      real(8), device :: y
      real(8), device :: z
    end subroutine
  end interface
  interface sincos
    procedure :: sincosf
    procedure :: sincos
  end interface

  interface
    attributes(device) subroutine sincospif(x, y, z) bind(c,name='__nv_sincospif')
      real(4), value :: x
      real(4), device :: y
      real(4), device :: z
    end subroutine
  end interface
  interface
    attributes(device) subroutine sincospi(x, y, z) bind(c,name='__nv_sincospi')
      real(8), value :: x
      real(8), device :: y
      real(8), device :: z
    end subroutine
  end interface
  interface sincospi
    procedure :: sincospif
    procedure :: sincospi
  end interface

  interface
    attributes(device) real(4) function __cosf(x) bind(c, name='__nv_cosf')
      real(4), value :: x
    end function
  end interface

  interface
    attributes(device) real(4) function cospif(x) bind(c,name='__nv_cospif')
      real(4), value :: x
    end function
  end interface
  interface
    attributes(device) real(8) function cospi(x) bind(c,name='__nv_cospi')
      real(8), value :: x
    end function
  end interface
  interface cospi
    procedure :: cospif
    procedure :: cospi
  end interface

  interface  
    attributes(device) real(4) function sinpif(x) bind(c,name='__nv_sinpif')
      real(4), value :: x
    end function
  end interface
  interface
    attributes(device) real(8) function sinpi(x) bind(c,name='__nv_sinpi')
      real(8), value :: x
    end function
  end interface
  interface sinpi
    procedure :: sinpif
    procedure :: sinpi
  end interface
  
  interface mulhi
   attributes(device) integer function __mulhi(i,j) bind(c,name='__nv_mulhi')
  !dir$ ignore_tkr (d) i, (d) j
    integer, value :: i,j
   end function
  end interface

  interface umulhi
   attributes(device) integer function __umulhi(i,j) bind(c,name='__nv_umulhi')
  !dir$ ignore_tkr (d) i, (d) j
    integer, value :: i,j
   end function
  end interface
  
  interface mul64hi
   attributes(device) integer(8) function __mul64hi(i,j) bind(c,name='__nv_mul64hi')
  !dir$ ignore_tkr (d) i, (d) j
    integer(8), value :: i,j
   end function
  end interface
  
  interface umul64hi
   attributes(device) integer(8) function __umul64hi(i,j) bind(c,name='__nv_umul64hi')
  !dir$ ignore_tkr (d) i, (d) j
    integer(8), value :: i,j
   end function
  end interface

  interface __float2half_rn
    attributes(device) real(2) function __float2half_rn(r) bind(c, name='__nv_float2half_rn')
      !dir$ ignore_tkr (d) r
      real, value :: r
    end function
  end interface

  interface __half2float
    attributes(device) real function __half2float(i) bind(c, name='__nv_half2float')
      !dir$ ignore_tkr (d) i
      real(2), value :: i
    end function
  end interface

  interface __double2int_rd
    attributes(device) integer function __double2int_rd(r) bind(c, name='__nv_double2int_rd')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2int_rn
    attributes(device) integer function __double2int_rn(r) bind(c, name='__nv_double2int_rn')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2int_ru
    attributes(device) integer function __double2int_ru(r) bind(c, name='__nv_double2int_ru')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2int_rz
    attributes(device) integer function __double2int_rz(r) bind(c, name='__nv_double2int_rz')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2uint_rd
    attributes(device) integer function __double2uint_rd(r) bind(c, name='__nv_double2uint_rd')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2uint_rn
    attributes(device) integer function __double2uint_rn(r) bind(c, name='__nv_double2uint_rn')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2uint_ru
    attributes(device) integer function __double2uint_ru(r) bind(c, name='__nv_double2uint_ru')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2uint_rz
    attributes(device) integer function __double2uint_rz(r) bind(c, name='__nv_double2uint_rz')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2float_rn
    attributes(device) real function __double2float_rn(r) bind(c, name='__nv_double2float_rn')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2float_rz
    attributes(device) real function __double2float_rz(r) bind(c, name='__nv_double2float_rz')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2float_ru
    attributes(device) real function __double2float_ru(r) bind(c, name='__nv_double2float_ru')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2float_rd
    attributes(device) real function __double2float_rd(r) bind(c, name='__nv_double2float_rd')
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2loint
    attributes(device) integer function __double2loint(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2hiint
    attributes(device) integer function __double2hiint(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __hiloint2double
    attributes(device) double precision function __hiloint2double(i, j) bind(c)
      !dir$ ignore_tkr (d) i, (d) j
      integer, value :: i, j
    end function
  end interface

  interface __int2double_rn
    attributes(device) double precision function __int2double_rn(i) bind(c)
      !dir$ ignore_tkr (d) i
      integer, value :: i
    end function
  end interface

  interface __uint2double_rn
    attributes(device) double precision function __uint2double_rn(i) bind(c)
      !dir$ ignore_tkr (d) i
      integer, value :: i
    end function
  end interface

  interface __double2ll_rn
    attributes(device) integer(8) function __double2ll_rn(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2ll_rz
    attributes(device) integer(8) function __double2ll_rz(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2ll_ru
    attributes(device) integer(8) function __double2ll_ru(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2ll_rd
    attributes(device) integer(8) function __double2ll_rd(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2ull_rn
    attributes(device) integer(8) function __double2ull_rn(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2ull_rz
    attributes(device) integer(8) function __double2ull_rz(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2ull_ru
    attributes(device) integer(8) function __double2ull_ru(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __double2ull_rd
    attributes(device) integer(8) function __double2ull_rd(r) bind(c)
      !dir$ ignore_tkr (d) r
      double precision, value :: r
    end function
  end interface

  interface __ll2double_rn
    attributes(device) double precision function __ll2double_rn(i) bind(c)
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __ll2double_rz
    attributes(device) double precision function __ll2double_rz(i) bind(c)
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __ll2double_ru
    attributes(device) double precision function __ll2double_ru(i) bind(c)
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __ll2double_rd
    attributes(device) double precision function __ll2double_rd(i) bind(c)
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __ull2double_rd
    attributes(device) double precision function __ull2double_rd(i) bind(c, name='__nv_ull2double_rd')
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __ull2double_rn
    attributes(device) double precision function __ull2double_rn(i) bind(c, name='__nv_ull2double_rn')
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __ull2double_ru
    attributes(device) double precision function __ull2double_ru(i) bind(c, name='__nv_ull2double_ru')
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __ull2double_rz
    attributes(device) double precision function __ull2double_rz(i) bind(c, name='__nv_ull2double_rz')
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __mul24
    attributes(device) integer function __mul24(i,j) bind(c, name='__nv_mul24')
      !dir$ ignore_tkr (d) i, (d) j
      integer, value :: i,j
    end function
  end interface

  interface __umul24
    attributes(device) integer function __umul24(i,j) bind(c, name='__nv_umul24')
      !dir$ ignore_tkr (d) i, (d) j
      integer, value :: i,j
    end function
  end interface

  interface __dsqrt_rd
    attributes(device) double precision function __dsqrt_rd(x) bind(c, name='__nv_dsqrt_rd')
      !dir$ ignore_tkr (d) x
      double precision, value :: x
    end function
  end interface

  interface __dsqrt_ru
    attributes(device) double precision function __dsqrt_ru(x) bind(c, name='__nv_dsqrt_ru')
      !dir$ ignore_tkr (d) x
      double precision, value :: x
    end function
  end interface

  interface __ddiv_rn
    attributes(device) double precision function __ddiv_rn(x,y) bind(c, name='__nv_ddiv_rn')
      !dir$ ignore_tkr (d) x, (d) y
      double precision, value :: x, y
    end function
  end interface

  interface __ddiv_rz
    attributes(device) double precision function __ddiv_rz(x,y) bind(c, name='__nv_ddiv_rz')
      !dir$ ignore_tkr (d) x, (d) y
      double precision, value :: x, y
    end function
  end interface

  interface __ddiv_ru
    attributes(device) double precision function __ddiv_ru(x,y) bind(c, name='__nv_ddiv_ru')
      !dir$ ignore_tkr (d) x, (d) y
      double precision, value :: x, y
    end function
  end interface

  interface __ddiv_rd
    attributes(device) double precision function __ddiv_rd(x,y) bind(c, name='__nv_ddiv_rd')
      !dir$ ignore_tkr (d) x, (d) y
      double precision, value :: x, y
    end function
  end interface

  interface __clz
    attributes(device) integer function __clz(i) bind(c, name='__nv_clz')
      !dir$ ignore_tkr (d) i
      integer, value :: i
    end function
    attributes(device) integer function __clzll(i) bind(c, name='__nv_clzll')
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __ffs
    attributes(device) integer function __ffs(i) bind(c, name='__nv_ffs')
      !dir$ ignore_tkr (d) i
      integer, value :: i
    end function
    attributes(device) integer function __ffsll(i) bind(c, name='__nv_ffsll')
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface __popc
    attributes(device) integer function __popc(i) bind(c, name='__nv_popc')
      !dir$ ignore_tkr (d) i
      integer, value :: i
    end function
    attributes(device) integer function __popcll(i) bind(c, name='__nv_popcll')
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  interface
    attributes(device) real(4) function __powf(x,y) bind(c, name='__nv_powf')
      !dir$ ignore_tkr (d) x, y
      real(4), value :: x, y
    end function
  end interface

  interface __brev
    attributes(device) integer function __brev(i) bind(c, name='__nv_brev')
      !dir$ ignore_tkr (d) i
      integer, value :: i
    end function
    attributes(device) integer(8) function __brevll(i) bind(c, name ='__nv_brevll')
      !dir$ ignore_tkr (d) i
      integer(8), value :: i
    end function
  end interface

  ! Atomic Operations

  interface atomicadd
    attributes(device) pure integer function atomicaddi(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
    attributes(device) pure real function atomicaddf(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real, intent(inout) :: address
    real, value :: val
    end function
    attributes(device) pure real(8) function atomicaddd(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real(8), intent(inout) :: address
    real(8), value :: val
    end function
    attributes(device) pure integer(8) function atomicaddl(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer(8), intent(inout) :: address
    integer(8), value :: val
    end function
  end interface

  interface atomicsub
    attributes(device) pure integer function atomicsubi(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
    attributes(device) pure real function atomicsubf(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real, intent(inout) :: address
    real, value :: val
    end function
    attributes(device) pure real(8) function atomicsubd(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real(8), intent(inout) :: address
    real(8), value :: val
    end function
    attributes(device) pure integer(8) function atomicsubl(address, val)
  !dir$ ignore_tkr (d) address, (dk) val
    integer(8), intent(inout) :: address
    integer(8), value :: val
    end function
  end interface
  
  interface atomicmax
    attributes(device) pure integer function atomicmaxi(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
    attributes(device) pure real function atomicmaxf(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real, intent(inout) :: address
    real, value :: val
    end function
    attributes(device) pure real(8) function atomicmaxd(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real(8), intent(inout) :: address
    real(8), value :: val
    end function
    attributes(device) pure integer(8) function atomicmaxl(address, val)
  !dir$ ignore_tkr (d) address, (dk) val
    integer(8), intent(inout) :: address
    integer(8), value :: val
    end function
  end interface
  
  interface atomicmin
    attributes(device) pure integer function atomicmini(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
    attributes(device) pure real function atomicminf(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real, intent(inout) :: address
    real, value :: val
    end function
    attributes(device) pure real(8) function atomicmind(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real(8), intent(inout) :: address
    real(8), value :: val
    end function
    attributes(device) pure integer(8) function atomicminl(address, val)
  !dir$ ignore_tkr (d) address, (dk) val
    integer(8), intent(inout) :: address
    integer(8), value :: val
    end function
  end interface
  
  interface atomicand
    attributes(device) pure integer function atomicandi(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
  end interface
  
  interface atomicor
    attributes(device) pure integer function atomicori(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
  end interface

  interface atomicinc
    attributes(device) pure integer function atomicinci(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
  end interface
  
  interface atomicdec
    attributes(device) pure integer function atomicdeci(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
  end interface

  interface atomiccas
    attributes(device) pure integer function atomiccasi(address, val, val2)
  !dir$ ignore_tkr (rd) address, (d) val, (d) val2
    integer, intent(inout) :: address
    integer, value :: val, val2
    end function
    attributes(device) pure integer(8) function atomiccasul(address, val, val2)
  !dir$ ignore_tkr (rd) address, (dk) val, (dk) val2
    integer(8), intent(inout) :: address
    integer(8), value :: val, val2
    end function
    attributes(device) pure real function atomiccasf(address, val, val2)
  !dir$ ignore_tkr (rd) address, (d) val, (d) val2
    real, intent(inout) :: address
    real, value :: val, val2
    end function
    attributes(device) pure double precision function atomiccasd(address, val, val2)
  !dir$ ignore_tkr (rd) address, (d) val, (d) val2
    double precision, intent(inout) :: address
    double precision, value :: val, val2
    end function
  end interface

  interface atomicexch
    attributes(device) pure integer function atomicexchi(address, val)
  !dir$ ignore_tkr (rd) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
    attributes(device) pure integer(8) function atomicexchul(address, val)
  !dir$ ignore_tkr (rd) address, (dk) val
    integer(8), intent(inout) :: address
    integer(8), value :: val
    end function
    attributes(device) pure real function atomicexchf(address, val)
  !dir$ ignore_tkr (rd) address, (d) val
    real, intent(inout) :: address
    real, value :: val
    end function
    attributes(device) pure double precision function atomicexchd(address, val)
  !dir$ ignore_tkr (rd) address, (d) val
    double precision, intent(inout) :: address
    double precision, value :: val
    end function
  end interface

  interface atomicxor
    attributes(device) pure integer function atomicxori(address, val)
  !dir$ ignore_tkr (rd) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
  end interface

  ! Time function

  interface
    attributes(device) integer function clock()
    end function
  end interface

  interface
    attributes(device) integer(8) function clock64()
    end function
  end interface

  interface
    attributes(device) integer(8) function globalTimer()
    end function
  end interface

  ! Warp Match Functions

  interface match_all_sync
    attributes(device) integer function match_all_syncjj(mask, val, pred)
  !dir$ ignore_tkr(d) mask, (d) val, (d) pred
    integer(4), value :: mask
    integer(4), value :: val
    integer(4)        :: pred
    end function
    attributes(device) integer function match_all_syncjx(mask, val, pred)
  !dir$ ignore_tkr(d) mask, (d) val, (d) pred
    integer(4), value :: mask
    integer(8), value :: val
    integer(4)        :: pred
    end function
    attributes(device) integer function match_all_syncjf(mask, val, pred)
  !dir$ ignore_tkr(d) mask, (d) val, (d) pred
    integer(4), value :: mask
    real(4), value    :: val
    integer(4)        :: pred
    end function
    attributes(device) integer function match_all_syncjd(mask, val, pred)
  !dir$ ignore_tkr(d) mask, (d) val, (d) pred
    integer(4), value :: mask
    real(8), value    :: val
    integer(4)        :: pred
    end function
  end interface

  interface match_any_sync
    attributes(device) integer function match_any_syncjj(mask, val)
  !dir$ ignore_tkr(d) mask, (d) val
    integer(4), value :: mask
    integer(4), value :: val
    end function
    attributes(device) integer function match_any_syncjx(mask, val)
  !dir$ ignore_tkr(d) mask, (d) val
    integer(4), value :: mask
    integer(8), value :: val
    end function
    attributes(device) integer function match_any_syncjf(mask, val)
  !dir$ ignore_tkr(d) mask, (d) val
    integer(4), value :: mask
    real(4), value    :: val
    end function
    attributes(device) integer function match_any_syncjd(mask, val)
  !dir$ ignore_tkr(d) mask, (d) val
    integer(4), value :: mask
    real(8), value    :: val
    end function
  end interface

  interface all_sync
    attributes(device) integer function all_sync(mask, pred)
      !dir$ ignore_tkr(d) mask, (td) pred
      integer, value :: mask, pred
    end function
  end interface

  interface any_sync
    attributes(device) integer function any_sync(mask, pred)
      !dir$ ignore_tkr(d) mask, (td) pred
      integer, value :: mask, pred
    end function
  end interface

  interface ballot_sync
    attributes(device) integer function ballot_sync(mask, pred)
      !dir$ ignore_tkr(d) mask, (td) pred
      integer, value :: mask, pred
    end function
  end interface

  ! LDCG
  interface __ldcg
    attributes(device) pure integer(4) function __ldcg_i4(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(4), intent(in) :: x
    end function
    attributes(device) pure integer(8) function __ldcg_i8(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(8), intent(in) :: x
    end function
    attributes(device) pure function __ldcg_cd(x) bind(c) result(y)
      import c_devptr
      !dir$ ignore_tkr (d) x
      type(c_devptr), intent(in) :: x
      type(c_devptr) :: y
    end function
    attributes(device) pure real(2) function __ldcg_r2(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(2), intent(in) :: x
    end function
    attributes(device) pure real(4) function __ldcg_r4(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(4), intent(in) :: x
    end function
    attributes(device) pure real(8) function __ldcg_r8(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(8), intent(in) :: x
    end function
    attributes(device) pure complex(4) function __ldcg_c4(x) &
        bind(c,name='__ldcg_c4x')
      !dir$ ignore_tkr (d) x
      complex(4), intent(in) :: x
    end function
    attributes(device) pure complex(8) function __ldcg_c8(x) &
        bind(c,name='__ldcg_c8x')
      !dir$ ignore_tkr (d) x
      complex(8), intent(in) :: x
    end function
    attributes(device) pure function __ldcg_i4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(4), dimension(4), intent(in) :: x
      integer(4), dimension(4) :: y
    end function
    attributes(device) pure function __ldcg_i8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(8), dimension(2), intent(in) :: x
      integer(8), dimension(2) :: y
    end function
    attributes(device) pure function __ldcg_r2x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(2), dimension(2), intent(in) :: x
      real(2), dimension(2) :: y
    end function
    attributes(device) pure function __ldcg_r4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      real(4), dimension(4), intent(in) :: x
      real(4), dimension(4) :: y
    end function
    attributes(device) pure function __ldcg_r8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(8), dimension(2), intent(in) :: x
      real(8), dimension(2) :: y
    end function
  end interface

  ! LDCA
  interface __ldca
    attributes(device) pure integer(4) function __ldca_i4(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(4), intent(in) :: x
    end function
    attributes(device) pure integer(8) function __ldca_i8(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(8), intent(in) :: x
    end function
    attributes(device) pure function __ldca_cd(x) bind(c) result(y)
      !dir$ ignore_tkr (d) x
      import c_devptr
      type(c_devptr), intent(in) :: x
      type(c_devptr) :: y
    end function
    attributes(device) pure real(2) function __ldca_r2(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(2), intent(in) :: x
    end function
    attributes(device) pure real(4) function __ldca_r4(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(4), intent(in) :: x
      end function
    attributes(device) pure real(8) function __ldca_r8(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(8), intent(in) :: x
    end function
    attributes(device) pure complex(4) function __ldca_c4(x) &
        bind(c,name='__ldca_c4x')
      !dir$ ignore_tkr (d) x
      complex(4), intent(in) :: x
    end function
    attributes(device) pure complex(8) function __ldca_c8(x) &
        bind(c,name='__ldca_c8x')
      !dir$ ignore_tkr (d) x
      complex(8), intent(in) :: x
    end function
    attributes(device) pure function __ldca_i4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(4), dimension(4), intent(in) :: x
      integer(4), dimension(4) :: y
    end function
    attributes(device) pure function __ldca_i8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(8), dimension(2), intent(in) :: x
      integer(8), dimension(2) :: y
    end function
    attributes(device) pure function __ldca_r2x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(2), dimension(2), intent(in) :: x
      real(2), dimension(2) :: y
    end function
    attributes(device) pure function __ldca_r4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      real(4), dimension(4), intent(in) :: x
      real(4), dimension(4) :: y
    end function
    attributes(device) pure function __ldca_r8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(8), dimension(2), intent(in) :: x
      real(8), dimension(2) :: y
    end function
  end interface

  ! LDCS
  interface __ldcs
    attributes(device) pure integer(4) function __ldcs_i4(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(4), intent(in) :: x
    end function
    attributes(device) pure integer(8) function __ldcs_i8(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(8), intent(in) :: x
    end function
    attributes(device) pure function __ldcs_cd(x) bind(c) result(y)
      import c_devptr
      !dir$ ignore_tkr (d) x
      type(c_devptr), intent(in) :: x
      type(c_devptr) :: y
    end function
    attributes(device) pure real(2) function __ldcs_r2(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(2), intent(in) :: x
    end function
    attributes(device) pure real(4) function __ldcs_r4(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(4), intent(in) :: x
    end function
    attributes(device) pure real(8) function __ldcs_r8(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(8), intent(in) :: x
    end function
    attributes(device) pure complex(4) function __ldcs_c4(x) &
        bind(c,name='__ldcs_c4x')
      !dir$ ignore_tkr (d) x
      complex(4), intent(in) :: x
    end function
    attributes(device) pure complex(8) function __ldcs_c8(x) &
        bind(c,name='__ldcs_c8x')
      !dir$ ignore_tkr (d) x
      complex(8), intent(in) :: x
    end function
    attributes(device) pure function __ldcs_i4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(4), dimension(4), intent(in) :: x
      integer(4), dimension(4) :: y
    end function
    attributes(device) pure function __ldcs_i8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(8), dimension(2), intent(in) :: x
      integer(8), dimension(2) :: y
    end function
    attributes(device) pure function __ldcs_r2x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(2), dimension(2), intent(in) :: x
      real(2), dimension(2) :: y
    end function
    attributes(device) pure function __ldcs_r4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      real(4), dimension(4), intent(in) :: x
      real(4), dimension(4) :: y
    end function
    attributes(device) pure function __ldcs_r8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(8), dimension(2), intent(in) :: x
      real(8), dimension(2) :: y
    end function
  end interface

  ! LDLU
  interface __ldlu
    attributes(device) pure integer(4) function __ldlu_i4(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(4), intent(in) :: x
    end function
    attributes(device) pure integer(8) function __ldlu_i8(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(8), intent(in) :: x
    end function
    attributes(device) pure function __ldlu_cd(x) bind(c) result(y)
      import c_devptr
      !dir$ ignore_tkr (d) x
      type(c_devptr), intent(in) :: x
      type(c_devptr) :: y
    end function
    attributes(device) pure real(2) function __ldlu_r2(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(2), intent(in) :: x
    end function
    attributes(device) pure real(4) function __ldlu_r4(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(4), intent(in) :: x
    end function
    attributes(device) pure real(8) function __ldlu_r8(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(8), intent(in) :: x
    end function
    attributes(device) pure complex(4) function __ldlu_c4(x) &
        bind(c,name='__ldlu_c4x')
      !dir$ ignore_tkr (d) x
      complex(4), intent(in) :: x
    end function
    attributes(device) pure complex(8) function __ldlu_c8(x) &
        bind(c,name='__ldlu_c8x')
      !dir$ ignore_tkr (d) x
      complex(8), intent(in) :: x
    end function
    attributes(device) pure function __ldlu_i4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(4), dimension(4), intent(in) :: x
      integer(4), dimension(4) :: y
    end function
    attributes(device) pure function __ldlu_i8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(8), dimension(2), intent(in) :: x
      integer(8), dimension(2) :: y
    end function
    attributes(device) pure function __ldlu_r2x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(2), dimension(2), intent(in) :: x
      real(2), dimension(2) :: y
    end function
    attributes(device) pure function __ldlu_r4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      real(4), dimension(4), intent(in) :: x
      real(4), dimension(4) :: y
    end function
    attributes(device) pure function __ldlu_r8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(8), dimension(2), intent(in) :: x
      real(8), dimension(2) :: y
    end function
  end interface

  ! LDCV
  interface __ldcv
    attributes(device) pure integer(4) function __ldcv_i4(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(4), intent(in) :: x
    end function
    attributes(device) pure integer(8) function __ldcv_i8(x) bind(c)
      !dir$ ignore_tkr (d) x
      integer(8), intent(in) :: x
    end function
    attributes(device) pure function __ldcv_cd(x) bind(c) result(y)
      import c_devptr
      !dir$ ignore_tkr (d) x
      type(c_devptr), intent(in) :: x
      type(c_devptr) :: y
    end function
    attributes(device) pure real(2) function __ldcv_r2(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(2), intent(in) :: x
    end function
    attributes(device) pure real(4) function __ldcv_r4(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(4), intent(in) :: x
    end function
    attributes(device) pure real(8) function __ldcv_r8(x) bind(c)
      !dir$ ignore_tkr (d) x
      real(8), intent(in) :: x
    end function
    attributes(device) pure complex(4) function __ldcv_c4(x) &
        bind(c,name='__ldcv_c4x')
      !dir$ ignore_tkr (d) x
      complex(4), intent(in) :: x
    end function
    attributes(device) pure complex(8) function __ldcv_c8(x) &
        bind(c,name='__ldcv_c8x')
      !dir$ ignore_tkr (d) x
      complex(8), intent(in) :: x
    end function
    attributes(device) pure function __ldcv_i4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(4), dimension(4), intent(in) :: x
      integer(4), dimension(4) :: y
      end function
    attributes(device) pure function __ldcv_i8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      integer(8), dimension(2), intent(in) :: x
      integer(8), dimension(2) :: y
    end function
    attributes(device) pure function __ldcv_r2x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(2), dimension(2), intent(in) :: x
      real(2), dimension(2) :: y
    end function
    attributes(device) pure function __ldcv_r4x4(x) result(y)
      !dir$ ignore_tkr (d) x
      real(4), dimension(4), intent(in) :: x
      real(4), dimension(4) :: y
    end function
    attributes(device) pure function __ldcv_r8x2(x) result(y)
      !dir$ ignore_tkr (d) x
      real(8), dimension(2), intent(in) :: x
      real(8), dimension(2) :: y
    end function
  end interface

  ! STWB
  interface __stwb
    attributes(device) pure subroutine __stwb_i4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(4), value  :: x
      integer(4), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwb_i8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(8), value  :: x
      integer(8), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwb_cd(y, x) bind(c)
      import c_devptr
      !dir$ ignore_tkr (d) y, (d) x
      type(c_devptr), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwb_r2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(2), value :: x
      real(2), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwb_r4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(4), value :: x
      real(4), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwb_r8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(8), value :: x
      real(8), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwb_c4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (rd) x
      complex(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwb_c8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      complex(8), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwb_i4x4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(4), dimension(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwb_i8x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(8), dimension(2), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwb_r2x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(2), dimension(2), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwb_r4x4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(4), dimension(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwb_r8x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(8), dimension(2), device, intent(in) :: y, x
    end subroutine
  end interface

  ! STCG
  interface __stcg
    attributes(device) pure subroutine __stcg_i4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(4), value  :: x
      integer(4), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcg_i8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(8), value  :: x
      integer(8), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcg_cd(y, x) bind(c)
      import c_devptr
      !dir$ ignore_tkr (d) y, (d) x
      type(c_devptr), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcg_r2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(2), value :: x
      real(2), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcg_r4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(4), value :: x
      real(4), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcg_r8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(8), value :: x
      real(8), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcg_c4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (rd) x
      complex(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcg_c8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      complex(8), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcg_i4x4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(4), dimension(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcg_i8x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(8), dimension(2), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcg_r2x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(2), dimension(2), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcg_r4x4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(4), dimension(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcg_r8x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(8), dimension(2), device, intent(in) :: y, x
    end subroutine
  end interface

  ! STCS
  interface __stcs
    attributes(device) pure subroutine __stcs_i4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(4), value  :: x
      integer(4), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcs_i8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(8), value  :: x
      integer(8), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcs_cd(y, x) bind(c)
      import c_devptr
      !dir$ ignore_tkr (d) y, (d) x
      type(c_devptr), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcs_r2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(2), value :: x
      real(2), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcs_r4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(4), value :: x
      real(4), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcs_r8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(8), value :: x
      real(8), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stcs_c4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (rd) x
      complex(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcs_c8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      complex(8), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcs_i4x4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(4), dimension(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcs_i8x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(8), dimension(2), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcs_r2x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(2), dimension(2), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcs_r4x4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(4), dimension(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stcs_r8x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(8), dimension(2), device, intent(in) :: y, x
    end subroutine
  end interface

  ! STWT
  interface __stwt
    attributes(device) pure subroutine __stwt_i4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(4), value  :: x
      integer(4), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwt_i8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(8), value  :: x
      integer(8), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwt_cd(y, x) bind(c)
      import c_devptr
      !dir$ ignore_tkr (d) y, (d) x
      type(c_devptr), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwt_r2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(2), value :: x
      real(2), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwt_r4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(4), value :: x
      real(4), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwt_r8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(8), value :: x
      real(8), device, intent(in) :: y
    end subroutine
    attributes(device) pure subroutine __stwt_c4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (rd) x
      complex(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwt_c8(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      complex(8), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwt_i4x4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(4), dimension(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwt_i8x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      integer(8), dimension(2), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwt_r2x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(2), dimension(2), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwt_r4x4(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(4), dimension(4), device, intent(in) :: y, x
    end subroutine
    attributes(device) pure subroutine __stwt_r8x2(y, x) bind(c)
      !dir$ ignore_tkr (d) y, (d) x
      real(8), dimension(2), device, intent(in) :: y, x
    end subroutine
  end interface

  interface
    attributes(device,host) logical function on_device() bind(c)
    end function
  end interface

contains

  attributes(device) subroutine syncthreads()
  end subroutine

end module
