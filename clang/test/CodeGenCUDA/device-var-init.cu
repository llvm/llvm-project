// REQUIRES: nvptx-registered-target

// Make sure we don't allow dynamic initialization for device
// variables, but accept empty constructors allowed by CUDA.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++11 \
// RUN:     -fno-threadsafe-statics -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++11 \
// RUN:     -emit-llvm -DERROR_CASE -verify -o /dev/null %s

#ifdef __clang__
#include "Inputs/cuda.h"
#endif

// Base classes with different initializer variants.

// trivial constructor -- allowed
struct T {
  int t;
};

// empty constructor
struct EC {
  int ec;
  __device__ EC() {}     // -- allowed
  __device__ EC(int) {}  // -- not allowed
};

// empty templated constructor -- allowed with no arguments
struct ETC {
  template <typename... T> __device__ ETC(T...) {}
};

// undefined constructor -- not allowed
struct UC {
  int uc;
  __device__ UC();
};

// empty constructor w/ initializer list -- not allowed
struct ECI {
  int eci;
  __device__ ECI() : eci(1) {}
};

// non-empty constructor -- not allowed
struct NEC {
  int nec;
  __device__ NEC() { nec = 1; }
};

// no-constructor,  virtual method -- not allowed
struct NCV {
  int ncv;
  __device__ virtual void vm() {}
};

// dynamic in-class field initializer -- not allowed
__device__ int f();
struct NCF {
  int ncf = f();
};

// static in-class field initializer.  NVCC does not allow it, but
// clang generates static initializer for this, so we'll accept it.
struct NCFS {
  int ncfs = 3;
};

// undefined templated constructor -- not allowed
struct UTC {
  template <typename... T> __device__ UTC(T...);
};

// non-empty templated constructor -- not allowed
struct NETC {
  int netc;
  template <typename... T> __device__ NETC(T...) { netc = 1; }
};

__device__ int d_v;
// CHECK: @d_v = addrspace(1) externally_initialized global i32 0,
__shared__ int s_v;
// CHECK: @s_v = addrspace(3) global i32 undef,
__constant__ int c_v;
// CHECK: addrspace(4) externally_initialized global i32 0,

__device__ int d_v_i = 1;
// CHECK: @d_v_i = addrspace(1) externally_initialized global i32 1,
#ifdef ERROR_CASE
__shared__ int s_v_i = 1;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
#endif
__constant__ int c_v_i = 1;
// CHECK: @c_v_i = addrspace(4) externally_initialized global i32 1,

#ifdef ERROR_CASE
__device__ int d_v_f = f();
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ int s_v_f = f();
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ int c_v_f = f();
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
#endif

__device__ T d_t;
// CHECK: @d_t = addrspace(1) externally_initialized global %struct.T zeroinitializer
__shared__ T s_t;
// CHECK: @s_t = addrspace(3) global %struct.T undef,
__constant__ T c_t;
// CHECK: @c_t = addrspace(4) externally_initialized global %struct.T zeroinitializer,

__device__ T d_t_i = {2};
// CHECKL @d_t_i = addrspace(1) externally_initialized global %struct.T { i32 2 },
#ifdef ERROR_CASE
__shared__ T s_t_i = {2};
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
#endif
__constant__ T c_t_i = {2};
// CHECK: @c_t_i = addrspace(4) externally_initialized global %struct.T { i32 2 },

__device__ EC d_ec;
// CHECK: @d_ec = addrspace(1) externally_initialized global %struct.EC zeroinitializer,
__shared__ EC s_ec;
// CHECK: @s_ec = addrspace(3) global %struct.EC undef,
__constant__ EC c_ec;
// CHECK: @c_ec = addrspace(4) externally_initialized global %struct.EC zeroinitializer,

#if ERROR_CASE
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
#endif

__device__ ETC d_etc;
// CHETCK: @d_etc = addrspace(1) externally_initialized global %struct.ETC zeroinitializer,
__shared__ ETC s_etc;
// CHETCK: @s_etc = addrspace(3) global %struct.ETC undef,
__constant__ ETC c_etc;
// CHETCK: @c_etc = addrspace(4) externally_initialized global %struct.ETC zeroinitializer,

#if ERROR_CASE
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

__device__ NCV d_ncv;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ NCV s_ncv;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ NCV c_ncv;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}

__device__ NCF d_ncf;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ NCF s_ncf;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ NCF c_ncf;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
#endif

__device__ NCFS d_ncfs;
// CHECK: @d_ncfs = addrspace(1) externally_initialized global %struct.NCFS { i32 3 }
__constant__ NCFS c_ncfs;
// CHECK: @c_ncfs = addrspace(4) externally_initialized global %struct.NCFS { i32 3 }

#if ERROR_CASE
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
#endif

// Regular base class -- allowed
struct T_B_T : T {};
__device__ T_B_T d_t_b_t;
// CHECK: @d_t_b_t = addrspace(1) externally_initialized global %struct.T_B_T zeroinitializer,
__shared__ T_B_T s_t_b_t;
// CHECK: @s_t_b_t = addrspace(3) global %struct.T_B_T undef,
__constant__ T_B_T c_t_b_t;
// CHECK: @c_t_b_t = addrspace(4) externally_initialized global %struct.T_B_T zeroinitializer,

// Incapsulated object of allowed class -- allowed
struct T_F_T {
  T t;
};
__device__ T_F_T d_t_f_t;
// CHECK: @d_t_f_t = addrspace(1) externally_initialized global %struct.T_F_T zeroinitializer,
__shared__ T_F_T s_t_f_t;
// CHECK: @s_t_f_t = addrspace(3) global %struct.T_F_T undef,
__constant__ T_F_T c_t_f_t;
// CHECK: @c_t_f_t = addrspace(4) externally_initialized global %struct.T_F_T zeroinitializer,

// array of allowed objects -- allowed
struct T_FA_T {
  T t[2];
};
__device__ T_FA_T d_t_fa_t;
// CHECK: @d_t_fa_t = addrspace(1) externally_initialized global %struct.T_FA_T zeroinitializer,
__shared__ T_FA_T s_t_fa_t;
// CHECK: @s_t_fa_t = addrspace(3) global %struct.T_FA_T undef,
__constant__ T_FA_T c_t_fa_t;
// CHECK: @c_t_fa_t = addrspace(4) externally_initialized global %struct.T_FA_T zeroinitializer,


// Calling empty base class initializer is OK
struct EC_I_EC : EC {
  __device__ EC_I_EC() : EC() {}
};
__device__ EC_I_EC d_ec_i_ec;
// CHECK: @d_ec_i_ec = addrspace(1) externally_initialized global %struct.EC_I_EC zeroinitializer,
__shared__ EC_I_EC s_ec_i_ec;
// CHECK: @s_ec_i_ec = addrspace(3) global %struct.EC_I_EC undef,
__constant__ EC_I_EC c_ec_i_ec;
// CHECK: @c_ec_i_ec = addrspace(4) externally_initialized global %struct.EC_I_EC zeroinitializer,

// .. though passing arguments is not allowed.
struct EC_I_EC1 : EC {
  __device__ EC_I_EC1() : EC(1) {}
};
#if ERROR_CASE
__device__ EC_I_EC1 d_ec_i_ec1;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ EC_I_EC1 s_ec_i_ec1;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ EC_I_EC1 c_ec_i_ec1;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
#endif

// Virtual base class -- not allowed
struct T_V_T : virtual T {};
#if ERROR_CASE
__device__ T_V_T d_t_v_t;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
__shared__ T_V_T s_t_v_t;
// expected-error@-1 {{initialization is not supported for __shared__ variables.}}
__constant__ T_V_T c_t_v_t;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables.}}
#endif

// Make sure that we don't allow if we inherit or incapsulate
// something with disallowed initializer.

// Inherited from or incapsulated class with non-empty constructor --
// not allowed
struct T_B_NEC : NEC {};
struct T_F_NEC {
  NEC nec;
};
struct T_FA_NEC {
  NEC nec[2];
};

#if ERROR_CASE
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
#endif

// We should not emit global initializers for device-side variables.
// CHECK-NOT: @__cxx_global_var_init

// Make sure that initialization restrictions do not apply to local
// variables.
__device__ void df() {
  T t;
  EC ec;
  ETC etc;
  UC uc;
  ECI eci;
  NEC nec;
  NCV ncv;
  NCF ncf;
  NCFS ncfs;
  UTC utc;
  NETC netc;
  T_B_T t_b_t;
  T_F_T t_f_t;
  T_FA_T t_fa_t;
  EC_I_EC ec_i_ec;
  EC_I_EC1 ec_i_ec1;
  T_V_T t_v_t;
  T_B_NEC t_b_nec;
  T_F_NEC t_f_nec;
  T_FA_NEC t_fa_nec;
  static __shared__ UC s_uc;
}

// CHECK:   call void @_ZN2ECC1Ev(%struct.EC* %ec)
// CHECK:   call void @_ZN3ETCC1IJEEEDpT_(%struct.ETC* %etc)
// CHECK:   call void @_ZN2UCC1Ev(%struct.UC* %uc)
// CHECK:   call void @_ZN3ECIC1Ev(%struct.ECI* %eci)
// CHECK:   call void @_ZN3NECC1Ev(%struct.NEC* %nec)
// CHECK:   call void @_ZN3NCVC1Ev(%struct.NCV* %ncv)
// CHECK:   call void @_ZN3NCFC1Ev(%struct.NCF* %ncf)
// CHECK:   call void @_ZN4NCFSC1Ev(%struct.NCFS* %ncfs)
// CHECK:   call void @_ZN3UTCC1IJEEEDpT_(%struct.UTC* %utc)
// CHECK:   call void @_ZN4NETCC1IJEEEDpT_(%struct.NETC* %netc)
// CHECK:   call void @_ZN7EC_I_ECC1Ev(%struct.EC_I_EC* %ec_i_ec)
// CHECK:   call void @_ZN8EC_I_EC1C1Ev(%struct.EC_I_EC1* %ec_i_ec1)
// CHECK:   call void @_ZN5T_V_TC1Ev(%struct.T_V_T* %t_v_t) #3
// CHECK:   call void @_ZN7T_B_NECC1Ev(%struct.T_B_NEC* %t_b_nec)
// CHECK:   call void @_ZN7T_F_NECC1Ev(%struct.T_F_NEC* %t_f_nec)
// CHECK:   call void @_ZN8T_FA_NECC1Ev(%struct.T_FA_NEC* %t_fa_nec)
// CHECK:   call void @_ZN2UCC1Ev(%struct.UC* addrspacecast (%struct.UC addrspace(3)* @_ZZ2dfvE4s_uc to %struct.UC*))
// CHECK: ret void

// We should not emit global init function.
// CHECK-NOT: @_GLOBAL__sub_I
