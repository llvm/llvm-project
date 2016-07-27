/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

extern __attribute__((const)) float __gcn_divrelaxed_fc_f32(bool, float, float);
extern __attribute__((const)) float __gcn_divrelaxednarrow_f32(float, float);
extern __attribute__((const)) float  __gcn_fldexp_fc_f32(bool, float, int);
extern __attribute__((const)) double __gcn_fldexp_f64(double, int);
extern __attribute__((const)) half __gcn_fldexp_f16(half, int);
extern __attribute__((const)) int __gcn_frexp_exp_fc_f32(bool, float);
extern __attribute__((const)) int __gcn_frexp_exp_f64(double);
extern __attribute__((const)) int __gcn_frexp_exp_f16(half);
extern __attribute__((const)) float  __gcn_frexp_mant_fc_f32(bool, float);
extern __attribute__((const)) double __gcn_frexp_mant_f64(double);
extern __attribute__((const)) half __gcn_frexp_mant_f16(half);
extern __attribute__((const)) float __gcn_max_f32(float,float);
extern __attribute__((const)) double __gcn_max_f64(double,double);
extern __attribute__((const)) half __gcn_max_f16(half,half);
extern __attribute__((const)) float __gcn_min_f32(float,float);
extern __attribute__((const)) double __gcn_min_f64(double,double);
extern __attribute__((const)) half __gcn_min_f16(half,half);
//extern __attribute__((pure)) ulong __gcn_mqsad_b64(ulong, uint, ulong);
//extern __attribute__((pure)) uint __gcn_msad_b32(uint, uint, uint);
//extern __attribute__((pure)) ulong __gcn_qsad_b64(ulong, uint, ulong);
extern __attribute__((const)) double __gcn_trig_preop_f64(double, int);

extern __attribute__((const)) float __hsail_f32_max3(float,float,float);
extern __attribute__((const)) float __hsail_f32_median3(float,float,float);
extern __attribute__((const)) float __hsail_f32_min3(float,float,float);
extern __attribute__((const)) int __hsail_imax3(int,int,int);
extern __attribute__((const)) int __hsail_imedian3(int,int,int);
extern __attribute__((const)) int __hsail_imin3(int,int,int);
extern __attribute__((const)) uint __hsail_umax3(uint,uint,uint);
extern __attribute__((const)) uint __hsail_umedian3(uint,uint,uint);
extern __attribute__((const)) uint __hsail_umin3(uint,uint,uint);

