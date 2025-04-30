/* Include gmp-mparam.h first, such that definitions of _SHORT_LIMB
   and _LONG_LONG_LIMB in it can take effect into gmp.h.  */
#include <gmp-mparam.h>

#ifndef __GMP_H__

#include <stdlib/gmp.h>

#include <bits/floatn.h>

#ifndef _ISOMAC
/* Now define the internal interfaces.  */
extern mp_size_t __mpn_extract_double (mp_ptr res_ptr, mp_size_t size,
				       int *expt, int *is_neg,
				       double value) attribute_hidden;

extern mp_size_t __mpn_extract_long_double (mp_ptr res_ptr, mp_size_t size,
					    int *expt, int *is_neg,
					    long double value)
     attribute_hidden;

#if __HAVE_DISTINCT_FLOAT128
extern mp_size_t __mpn_extract_float128 (mp_ptr res_ptr, mp_size_t size,
					 int *expt, int *is_neg,
					 _Float128 value)
     attribute_hidden;
#endif

extern float __mpn_construct_float (mp_srcptr frac_ptr, int expt, int sign)
     attribute_hidden;

extern double __mpn_construct_double (mp_srcptr frac_ptr, int expt,
				      int negative) attribute_hidden;

extern long double __mpn_construct_long_double (mp_srcptr frac_ptr, int expt,
						int sign)
     attribute_hidden;

#if __HAVE_DISTINCT_FLOAT128
extern _Float128 __mpn_construct_float128 (mp_srcptr frac_ptr, int expt,
					   int sign) attribute_hidden;
#endif

extern __typeof (mpn_add_1) mpn_add_1 attribute_hidden;
extern __typeof (mpn_addmul_1) mpn_addmul_1 attribute_hidden;
extern __typeof (mpn_add_n) mpn_add_n attribute_hidden;
extern __typeof (mpn_cmp) mpn_cmp attribute_hidden;
extern __typeof (mpn_divrem) mpn_divrem attribute_hidden;
extern __typeof (mpn_lshift) mpn_lshift attribute_hidden;
extern __typeof (mpn_mul) mpn_mul attribute_hidden;
extern __typeof (mpn_mul_1) mpn_mul_1 attribute_hidden;
extern __typeof (mpn_rshift) mpn_rshift attribute_hidden;
extern __typeof (mpn_sub_1) mpn_sub_1 attribute_hidden;
extern __typeof (mpn_submul_1) mpn_submul_1 attribute_hidden;
extern __typeof (mpn_sub_n) mpn_sub_n attribute_hidden;
#endif

#endif
