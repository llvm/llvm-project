! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#if !defined(PGDLL)
#define EXPORT(z)
#else
#define EXPORT(z) cdec$ attributes dllexport :: z
#endif


#if defined(NOMIXEDSTRLEN)
#define FTN_OLDN(A)   A
#define FTN_NEWN(S)   FTN_OLDN(S)_nm
#else
#if defined(MIXEDSTRLEN)
#define FTN_OLDN(A)   A
#define FTN_NEWN(S)   FTN_OLDN(S)_m_
#else
#define FTN_NEWN(S) S 
#endif
#endif

	real function FTN_NEWN(ftn_cabs)(c)
EXPORT(FTN_NEWN(ftn_cabs))
	complex c
	FTN_NEWN(ftn_cabs) = abs(c)
	return
	end
	complex function FTN_NEWN(ftn_ccos)(c)
EXPORT(FTN_NEWN(ftn_ccos))
	complex c
	FTN_NEWN(ftn_ccos) = cos(c)
	return
	end
	complex function FTN_NEWN(ftn_cexp)(c)
EXPORT(FTN_NEWN(ftn_cexp))
	complex c
	FTN_NEWN(ftn_cexp) = exp(c)
	return
	end
	complex function FTN_NEWN(ftn_clog)(c)
EXPORT(FTN_NEWN(ftn_clog))
	complex c
	FTN_NEWN(ftn_clog) = log(c)
	return
	end
	complex function FTN_NEWN(ftn_conjg)(c)
EXPORT(FTN_NEWN(ftn_conjg))
	complex c
	FTN_NEWN(ftn_conjg) = conjg(c)
	return
	end
	complex function FTN_NEWN(ftn_csin)(c)
EXPORT(FTN_NEWN(ftn_csin))
	complex c
	FTN_NEWN(ftn_csin) = sin(c)
	return
	end
	complex function FTN_NEWN(ftn_csqrt)(c)
EXPORT(FTN_NEWN(ftn_csqrt))
	complex c
	FTN_NEWN(ftn_csqrt) = sqrt(c)
	return
	end

	double precision function FTN_NEWN(ftn_cdabs)(c)
EXPORT(FTN_NEWN(ftn_cdabs))
	double complex c
	FTN_NEWN(ftn_cdabs) = abs(c)
	return
	end
	double complex function FTN_NEWN(ftn_cdcos)(c)
EXPORT(FTN_NEWN(ftn_cdcos))
	double complex c
	FTN_NEWN(ftn_cdcos) = cos(c)
	return
	end
	double complex function FTN_NEWN(ftn_cdexp)(c)
EXPORT(FTN_NEWN(ftn_cdexp))
	double complex c
	FTN_NEWN(ftn_cdexp) = exp(c)
	return
	end
	double complex function FTN_NEWN(ftn_cdlog)(c)
EXPORT(FTN_NEWN(ftn_cdlog))
	double complex c
	FTN_NEWN(ftn_cdlog) = log(c)
	return
	end
	double complex function FTN_NEWN(ftn_dconjg)(c)
EXPORT(FTN_NEWN(ftn_dconjg))
	double complex c
	FTN_NEWN(ftn_dconjg) = conjg(c)
	return
	end
	double complex function FTN_NEWN(ftn_cdsin)(c)
EXPORT(FTN_NEWN(ftn_cdsin))
	double complex c
	FTN_NEWN(ftn_cdsin) = sin(c)
	return
	end
	double complex function FTN_NEWN(ftn_cdsqrt)(c)
EXPORT(FTN_NEWN(ftn_cdsqrt))
	double complex c
	FTN_NEWN(ftn_cdsqrt) = sqrt(c)
	return
	end

	real*8 function FTN_NEWN(ftn_dacos)(r)
EXPORT(FTN_NEWN(ftn_dacos))
	real*8 r
	FTN_NEWN(ftn_dacos) = dacos(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dasin)(r)
EXPORT(FTN_NEWN(ftn_dasin))
	real*8 r
	FTN_NEWN(ftn_dasin) = dasin(r)
	return
	end
	real*8 function FTN_NEWN(ftn_datan)(r)
EXPORT(FTN_NEWN(ftn_datan))
	real*8 r
	FTN_NEWN(ftn_datan) = datan(r)
	return
	end
	real*8 function FTN_NEWN(ftn_datan2)(x, y)
EXPORT(FTN_NEWN(ftn_datan2))
	real*8 x, y
	FTN_NEWN(ftn_datan2) = datan2(x, y)
	return
	end
	real*8 function FTN_NEWN(ftn_dcos)(r)
EXPORT(FTN_NEWN(ftn_dcos))
	real*8 r
	FTN_NEWN(ftn_dcos) = dcos(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dcosh)(r)
EXPORT(FTN_NEWN(ftn_dcosh))
	real*8 r
	FTN_NEWN(ftn_dcosh) = dcosh(r)
	return
	end
	real*8 function FTN_NEWN(ftn_ddim)(x, y)
EXPORT(FTN_NEWN(ftn_ddim))
	real*8 x, y
	FTN_NEWN(ftn_ddim) = ddim(x, y)
	return
	end
	real*8 function FTN_NEWN(ftn_dexp)(r)
EXPORT(FTN_NEWN(ftn_dexp))
	real*8 r
	FTN_NEWN(ftn_dexp) = dexp(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dimag)(c)
EXPORT(FTN_NEWN(ftn_dimag))
	double complex c
	FTN_NEWN(ftn_dimag) = dimag(c)
	return
	end
	real*8 function FTN_NEWN(ftn_dint)(r)
EXPORT(FTN_NEWN(ftn_dint))
	real*8 r
	FTN_NEWN(ftn_dint) = dint(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dlog)(r)
EXPORT(FTN_NEWN(ftn_dlog))
	real*8 r
	FTN_NEWN(ftn_dlog) = dlog(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dlog10)(r)
EXPORT(FTN_NEWN(ftn_dlog10))
	real*8 r
	FTN_NEWN(ftn_dlog10) = dlog10(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dmod)(r, s)
EXPORT(FTN_NEWN(ftn_dmod))
	real*8 r, s
	FTN_NEWN(ftn_dmod) = dmod(r, s)
	return
	end
	real*8 function FTN_NEWN(ftn_dnint)(r)
EXPORT(FTN_NEWN(ftn_dnint))
	real*8 r
	FTN_NEWN(ftn_dnint) = dnint(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dprod)(x, y)
EXPORT(FTN_NEWN(ftn_dprod))
	real x, y
	FTN_NEWN(ftn_dprod) = dprod(x, y)
	return
	end
	real*8 function FTN_NEWN(ftn_dsign)(x, y)
EXPORT(FTN_NEWN(ftn_dsign))
	real*8 x, y
	FTN_NEWN(ftn_dsign) = dsign(x, y)
	return
	end
	real*8 function FTN_NEWN(ftn_dsin)(r)
EXPORT(FTN_NEWN(ftn_dsin))
	real*8 r
	FTN_NEWN(ftn_dsin) = dsin(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dsinh)(r)
EXPORT(FTN_NEWN(ftn_dsinh))
	real*8 r
	FTN_NEWN(ftn_dsinh) = dsinh(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dsqrt)(r)
EXPORT(FTN_NEWN(ftn_dsqrt))
	real*8 r
	FTN_NEWN(ftn_dsqrt) = dsqrt(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dtan)(r)
EXPORT(FTN_NEWN(ftn_dtan))
	real*8 r
	FTN_NEWN(ftn_dtan) = dtan(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dtanh)(r)
EXPORT(FTN_NEWN(ftn_dtanh))
	real*8 r
	FTN_NEWN(ftn_dtanh) = dtanh(r)
	return
	end
	
	integer function FTN_NEWN(ftn_idim)(i, j)
EXPORT(FTN_NEWN(ftn_idim))
	integer i, j
	FTN_NEWN(ftn_idim) = idim(i, j)
	return
	end
	integer function FTN_NEWN(ftn_idnint)(r)
EXPORT(FTN_NEWN(ftn_idnint))
	real*8 r
	FTN_NEWN(ftn_idnint) = idnint(r)
	return
	end
	integer function FTN_NEWN(ftn_isign)(i, j)
EXPORT(FTN_NEWN(ftn_isign))
	integer i, j
	FTN_NEWN(ftn_isign) = isign(i, j)
	return
	end
	integer function FTN_NEWN(ftn_mod)(i, j)
EXPORT(FTN_NEWN(ftn_mod))
	integer i, j
	FTN_NEWN(ftn_mod) = mod(i, j)
	return
	end
	integer function FTN_NEWN(ftn_nint)(r)
EXPORT(FTN_NEWN(ftn_nint))
	real r
	FTN_NEWN(ftn_nint) = nint(r)
	return
	end

	real function FTN_NEWN(ftn_abs)(r)
EXPORT(FTN_NEWN(ftn_abs))
	real r
	FTN_NEWN(ftn_abs) = abs(r)
	return
	end
	real function FTN_NEWN(ftn_acos)(r)
EXPORT(FTN_NEWN(ftn_acos))
	real r
	FTN_NEWN(ftn_acos) = acos(r)
	return
	end
	real function FTN_NEWN(ftn_aimag)(c)
EXPORT(FTN_NEWN(ftn_aimag))
	complex c
	FTN_NEWN(ftn_aimag) = aimag(c)
	return
	end
	real function FTN_NEWN(ftn_aint)(r)
EXPORT(FTN_NEWN(ftn_aint))
	real r
	FTN_NEWN(ftn_aint) = aint(r)
	return
	end
	real function FTN_NEWN(ftn_alog)(r)
EXPORT(FTN_NEWN(ftn_alog))
	real r
	FTN_NEWN(ftn_alog) = alog(r)
	return
	end
	real function FTN_NEWN(ftn_alog10)(r)
EXPORT(FTN_NEWN(ftn_alog10))
	real r
	FTN_NEWN(ftn_alog10) = alog10(r)
	return
	end
	real function FTN_NEWN(ftn_amod)(r, s)
EXPORT(FTN_NEWN(ftn_amod))
	real r, s
	FTN_NEWN(ftn_amod) = amod(r, s)
	return
	end
	real function FTN_NEWN(ftn_anint)(r)
EXPORT(FTN_NEWN(ftn_anint))
	real r
	FTN_NEWN(ftn_anint) = anint(r)
	return
	end
	real function FTN_NEWN(ftn_asin)(r)
EXPORT(FTN_NEWN(ftn_asin))
	real r
	FTN_NEWN(ftn_asin) = asin(r)
	return
	end
	real function FTN_NEWN(ftn_atan)(r)
EXPORT(FTN_NEWN(ftn_atan))
	real r
	FTN_NEWN(ftn_atan) = atan(r)
	return
	end
	real function FTN_NEWN(ftn_atan2)(x, y)
EXPORT(FTN_NEWN(ftn_atan2))
	real x, y
	FTN_NEWN(ftn_atan2) = atan2(x, y)
	return
	end
	real function FTN_NEWN(ftn_cos)(r)
EXPORT(FTN_NEWN(ftn_cos))
	real r
	FTN_NEWN(ftn_cos) = cos(r)
	return
	end
	real function FTN_NEWN(ftn_cosh)(r)
EXPORT(FTN_NEWN(ftn_cosh))
	real r
	FTN_NEWN(ftn_cosh) = cosh(r)
	return
	end
	real function FTN_NEWN(ftn_exp)(r)
EXPORT(FTN_NEWN(ftn_exp))
	real r
	FTN_NEWN(ftn_exp) = exp(r)
	return
	end
	real function FTN_NEWN(ftn_dim)(x, y)
EXPORT(FTN_NEWN(ftn_dim))
	real x, y
	FTN_NEWN(ftn_dim) = dim(x, y)
	return
	end
	real function FTN_NEWN(ftn_sign)(x, y)
EXPORT(FTN_NEWN(ftn_sign))
	real x, y
	FTN_NEWN(ftn_sign) = sign(x, y)
	return
	end
	real function FTN_NEWN(ftn_sin)(r)
EXPORT(FTN_NEWN(ftn_sin))
	real r
	FTN_NEWN(ftn_sin) = sin(r)
	return
	end
	real function FTN_NEWN(ftn_sinh)(r)
EXPORT(FTN_NEWN(ftn_sinh))
	real r
	FTN_NEWN(ftn_sinh) = sinh(r)
	return
	end
	real function FTN_NEWN(ftn_sqrt)(r)
EXPORT(FTN_NEWN(ftn_sqrt))
	real r
	FTN_NEWN(ftn_sqrt) = sqrt(r)
	return
	end
	real function FTN_NEWN(ftn_tan)(r)
EXPORT(FTN_NEWN(ftn_tan))
	real r
	FTN_NEWN(ftn_tan) = tan(r)
	return
	end
	real function FTN_NEWN(ftn_tanh)(r)
EXPORT(FTN_NEWN(ftn_tanh))
	real r
	FTN_NEWN(ftn_tanh) = tanh(r)
	return
	end
	real*8 function FTN_NEWN(ftn_dabs)(r)
EXPORT(FTN_NEWN(ftn_dabs))
	real*8 r
	FTN_NEWN(ftn_dabs) = dabs(r)
	return
	end
	integer function FTN_NEWN(ftn_iabs)(i)
EXPORT(FTN_NEWN(ftn_iabs))
	integer i
	FTN_NEWN(ftn_iabs) = iabs(i)
	return
	end
	integer function FTN_NEWN(ftn_str_len)(s)
EXPORT(FTN_NEWN(ftn_str_len))
	character s*(*)
	FTN_NEWN(ftn_str_len) = len(s)
	return
	end
	integer function FTN_NEWN(ftn_index)(s1, s2)
EXPORT(FTN_NEWN(ftn_index))
	character s1*(*), s2*(*)
	FTN_NEWN(ftn_index) = index(s1, s2)
	return
	end
	integer*2 function FTN_NEWN(ftn_iishft)(i, j)
EXPORT(FTN_NEWN(ftn_iishft))
	integer*2 i, j
	FTN_NEWN(ftn_iishft) = ishft(i, j)
	end
	integer*4 function FTN_NEWN(ftn_jishft)(i, j)
EXPORT(FTN_NEWN(ftn_jishft))
	integer*4 i, j
	FTN_NEWN(ftn_jishft) = ishft(i, j)
	end
	integer*2 function FTN_NEWN(ftn_iibits)(i, j, k)
EXPORT(FTN_NEWN(ftn_iibits))
	integer*2 i, j, k
	FTN_NEWN(ftn_iibits) = ibits(i, j, k)
	end
	integer*4 function FTN_NEWN(ftn_jibits)(i, j, k)
EXPORT(FTN_NEWN(ftn_jibits))
	integer*4 i, j, k
	FTN_NEWN(ftn_jibits) = ibits(i, j, k)
	end
	integer*2 function FTN_NEWN(ftn_iibset)(i, j)
EXPORT(FTN_NEWN(ftn_iibset))
	integer*2 i, j
	FTN_NEWN(ftn_iibset) = ibset(i, j)
	end
	integer*4 function FTN_NEWN(ftn_jibset)(i, j)
EXPORT(FTN_NEWN(ftn_jibset))
	integer*4 i, j
	FTN_NEWN(ftn_jibset) = ibset(i, j)
	end
	logical*2 function FTN_NEWN(ftn_bitest)(i, j)
EXPORT(FTN_NEWN(ftn_bitest))
	integer*2 i, j
	FTN_NEWN(ftn_bitest) = btest(i, j)
	end
	logical*4 function FTN_NEWN(ftn_bjtest)(i, j)
EXPORT(FTN_NEWN(ftn_bjtest))
	integer*4 i, j
	FTN_NEWN(ftn_bjtest) = btest(i, j)
	end
	integer*2 function FTN_NEWN(ftn_iibclr)(i, j)
EXPORT(FTN_NEWN(ftn_iibclr))
	integer*2 i, j
	FTN_NEWN(ftn_iibclr) = ibclr(i, j)
	end
	integer*4 function FTN_NEWN(ftn_jibclr)(i, j)
EXPORT(FTN_NEWN(ftn_jibclr))
	integer*4 i, j
	FTN_NEWN(ftn_jibclr) = ibclr(i, j)
	end
	integer*2 function FTN_NEWN(ftn_iishftc)(i, j, k)
EXPORT(FTN_NEWN(ftn_iishftc))
	integer*2 i, j, k
	FTN_NEWN(ftn_iishftc) = ishftc(i, j, k)
	end
	integer*4 function FTN_NEWN(ftn_jishftc)(i, j, k)
EXPORT(FTN_NEWN(ftn_jishftc))
	integer*4 i, j, k
	FTN_NEWN(ftn_jishftc) = ishftc(i, j, k)
	end
