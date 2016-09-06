// REQUIRES: nvptx-registered-target

// Make sure we don't allow dynamic initialization for device
// variables, but accept empty constructors allowed by CUDA.

// RUN: %clang_cc1 -verify %s -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++11 %s

#ifdef __clang__
#include "Inputs/cuda.h"
#endif

// Use the types we share with CodeGen tests.
#include "Inputs/cuda-initializers.h"

__shared__ int s_v_i = 1;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}

__device__ int d_v_f = f();
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ int s_v_f = f();
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ int c_v_f = f();
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__shared__ T s_t_i = {2};
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}

__device__ EC d_ec_i(3);
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ EC s_ec_i(3);
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ EC c_ec_i(3);
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ EC d_ec_i2 = {3};
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ EC s_ec_i2 = {3};
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ EC c_ec_i2 = {3};
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ ETC d_etc_i(3);
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ ETC s_etc_i(3);
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ ETC c_etc_i(3);
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ ETC d_etc_i2 = {3};
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ ETC s_etc_i2 = {3};
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ ETC c_etc_i2 = {3};
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ UC d_uc;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ UC s_uc;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ UC c_uc;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ UD d_ud;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ UD s_ud;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ UD c_ud;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ ECI d_eci;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ ECI s_eci;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ ECI c_eci;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ NEC d_nec;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ NEC s_nec;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ NEC c_nec;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ NED d_ned;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ NED s_ned;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ NED c_ned;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ NCV d_ncv;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ NCV s_ncv;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ NCV c_ncv;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ VD d_vd;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ VD s_vd;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ VD c_vd;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ NCF d_ncf;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ NCF s_ncf;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ NCF c_ncf;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__shared__ NCFS s_ncfs;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}

__device__ UTC d_utc;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ UTC s_utc;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ UTC c_utc;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ UTC d_utc_i(3);
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ UTC s_utc_i(3);
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ UTC c_utc_i(3);
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ NETC d_netc;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ NETC s_netc;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ NETC c_netc;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ NETC d_netc_i(3);
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ NETC s_netc_i(3);
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ NETC c_netc_i(3);
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ EC_I_EC1 d_ec_i_ec1;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ EC_I_EC1 s_ec_i_ec1;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ EC_I_EC1 c_ec_i_ec1;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ T_V_T d_t_v_t;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ T_V_T s_t_v_t;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ T_V_T c_t_v_t;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ T_B_NEC d_t_b_nec;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ T_B_NEC s_t_b_nec;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ T_B_NEC c_t_b_nec;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ T_F_NEC d_t_f_nec;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ T_F_NEC s_t_f_nec;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ T_F_NEC c_t_f_nec;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ T_FA_NEC d_t_fa_nec;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ T_FA_NEC s_t_fa_nec;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ T_FA_NEC c_t_fa_nec;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ T_B_NED d_t_b_ned;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ T_B_NED s_t_b_ned;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ T_B_NED c_t_b_ned;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ T_F_NED d_t_f_ned;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ T_F_NED s_t_f_ned;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ T_F_NED c_t_f_ned;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ T_FA_NED d_t_fa_ned;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ T_FA_NED s_t_fa_ned;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ T_FA_NED c_t_fa_ned;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

// Verify that only __shared__ local variables may be static on device
// side and that they are not allowed to be initialized.
__device__ void df_sema() {
  static __shared__ NCFS s_ncfs;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __shared__ UC s_uc;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __shared__ NED s_ned;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}

  static __device__ int ds;
  // expected-error@-1 {{Within a __device__/__global__ function, only __shared__ variables may be marked "static"}}
  static __constant__ int dc;
  // expected-error@-1 {{Within a __device__/__global__ function, only __shared__ variables may be marked "static"}}
  static int v;
  // expected-error@-1 {{Within a __device__/__global__ function, only __shared__ variables may be marked "static"}}
}
