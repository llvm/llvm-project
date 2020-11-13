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
__device__ T d_t_i = {2};
__constant__ T c_t_i = {2};

__device__ ECD d_ecd_i{};
__shared__ ECD s_ecd_i{};
__constant__ ECD c_ecd_i{};

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

// Verify that local variables may be static on device
// side and that they conform to the initialization constraints.
// __shared__ can't be initialized at all and others don't support dynamic initialization.
__device__ void df_sema() {
  static __device__ int ds;
  static __constant__ int dc;
  static int v;
  static const int cv = 1;
  static const __device__ int cds = 1;
  static const __constant__ int cdc = 1;


  // __shared__ does not need to be explicitly static.
  __shared__ int lsi;
  // __constant__ and __device__ can not be non-static local
  __constant__ int lci;
  // expected-error@-1 {{__constant__ and __device__ are not allowed on non-static local variables}}
  __device__ int ldi;
  // expected-error@-1 {{__constant__ and __device__ are not allowed on non-static local variables}}

  // Same test cases as for the globals above.

  static __device__ int d_v_f = f();
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ int s_v_f = f();
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ int c_v_f = f();
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __shared__ T s_t_i = {2};
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __device__ T d_t_i = {2};
  static __constant__ T c_t_i = {2};

  static __device__ ECD d_ecd_i;
  static __shared__ ECD s_ecd_i;
  static __constant__ ECD c_ecd_i;

  static __device__ EC d_ec_i(3);
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ EC s_ec_i(3);
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ EC c_ec_i(3);
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ EC d_ec_i2 = {3};
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ EC s_ec_i2 = {3};
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ EC c_ec_i2 = {3};
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ ETC d_etc_i(3);
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ ETC s_etc_i(3);
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ ETC c_etc_i(3);
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ ETC d_etc_i2 = {3};
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ ETC s_etc_i2 = {3};
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ ETC c_etc_i2 = {3};
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ UC d_uc;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ UC s_uc;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ UC c_uc;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ UD d_ud;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ UD s_ud;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ UD c_ud;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ ECI d_eci;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ ECI s_eci;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ ECI c_eci;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ NEC d_nec;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ NEC s_nec;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ NEC c_nec;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ NED d_ned;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ NED s_ned;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ NED c_ned;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ NCV d_ncv;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ NCV s_ncv;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ NCV c_ncv;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ VD d_vd;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ VD s_vd;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ VD c_vd;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ NCF d_ncf;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ NCF s_ncf;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ NCF c_ncf;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __shared__ NCFS s_ncfs;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}

  static __device__ UTC d_utc;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ UTC s_utc;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ UTC c_utc;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ UTC d_utc_i(3);
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ UTC s_utc_i(3);
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ UTC c_utc_i(3);
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ NETC d_netc;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ NETC s_netc;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ NETC c_netc;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ NETC d_netc_i(3);
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ NETC s_netc_i(3);
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ NETC c_netc_i(3);
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ EC_I_EC1 d_ec_i_ec1;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ EC_I_EC1 s_ec_i_ec1;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ EC_I_EC1 c_ec_i_ec1;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ T_V_T d_t_v_t;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ T_V_T s_t_v_t;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ T_V_T c_t_v_t;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ T_B_NEC d_t_b_nec;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ T_B_NEC s_t_b_nec;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ T_B_NEC c_t_b_nec;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ T_F_NEC d_t_f_nec;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ T_F_NEC s_t_f_nec;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ T_F_NEC c_t_f_nec;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ T_FA_NEC d_t_fa_nec;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ T_FA_NEC s_t_fa_nec;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ T_FA_NEC c_t_fa_nec;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ T_B_NED d_t_b_ned;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ T_B_NED s_t_b_ned;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ T_B_NED c_t_b_ned;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ T_F_NED d_t_f_ned;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ T_F_NED s_t_f_ned;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ T_F_NED c_t_f_ned;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

  static __device__ T_FA_NED d_t_fa_ned;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
  static __shared__ T_FA_NED s_t_fa_ned;
  // expected-error@-1 {{initialization is not supported for __shared__ variables.}}
  static __constant__ T_FA_NED c_t_fa_ned;
  // expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
}

__host__ __device__ void hd_sema() {
  static int x = 42;
}

inline __host__ __device__ void hd_emitted_host_only() {
  static int x = 42; // no error on device because this is never codegen'ed there.
}
void call_hd_emitted_host_only() { hd_emitted_host_only(); }

// Verify that we also check field initializers in instantiated structs.
struct NontrivialInitializer {
  __host__ __device__ NontrivialInitializer() : x(43) {}
  int x;
};

template <typename T>
__global__ void bar() {
  __shared__ T bad;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
}

void instantiate() {
  bar<NontrivialInitializer><<<1, 1>>>();
// expected-note@-1 {{in instantiation of function template specialization 'bar<NontrivialInitializer>' requested here}}
}
