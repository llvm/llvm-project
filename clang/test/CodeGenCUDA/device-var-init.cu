// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// Make sure we don't allow dynamic initialization for device
// variables, but accept empty constructors allowed by CUDA.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++11 \
// RUN:     -fno-threadsafe-statics -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,NVPTX %s

// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -std=c++11 \
// RUN:     -fno-threadsafe-statics -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,AMDGCN %s

#ifdef __clang__
#include "Inputs/cuda.h"
#endif

// Use the types we share with Sema tests.
#include "Inputs/cuda-initializers.h"

__device__ int d_v;
// CHECK: @d_v = addrspace(1) externally_initialized global i32 0,
__shared__ int s_v;
// CHECK: @s_v = addrspace(3) global i32 undef,
__constant__ int c_v;
// CHECK: addrspace(4) externally_initialized global i32 0,

__device__ int d_v_i = 1;
// CHECK: @d_v_i = addrspace(1) externally_initialized global i32 1,

// trivial constructor -- allowed
__device__ T d_t;
// CHECK: @d_t = addrspace(1) externally_initialized global %struct.T zeroinitializer
__shared__ T s_t;
// CHECK: @s_t = addrspace(3) global %struct.T undef,
__constant__ T c_t;
// CHECK: @c_t = addrspace(4) externally_initialized global %struct.T zeroinitializer,

__device__ T d_t_i = {2};
// CHECK: @d_t_i = addrspace(1) externally_initialized global %struct.T { i32 2 },
__constant__ T c_t_i = {2};
// CHECK: @c_t_i = addrspace(4) externally_initialized global %struct.T { i32 2 },

// empty constructor
__device__ EC d_ec;
// CHECK: @d_ec = addrspace(1) externally_initialized global %struct.EC zeroinitializer,
__shared__ EC s_ec;
// CHECK: @s_ec = addrspace(3) global %struct.EC undef,
__constant__ EC c_ec;
// CHECK: @c_ec = addrspace(4) externally_initialized global %struct.EC zeroinitializer,

// empty destructor
__device__ ED d_ed;
// CHECK: @d_ed = addrspace(1) externally_initialized global %struct.ED zeroinitializer,
__shared__ ED s_ed;
// CHECK: @s_ed = addrspace(3) global %struct.ED undef,
__constant__ ED c_ed;
// CHECK: @c_ed = addrspace(4) externally_initialized global %struct.ED zeroinitializer,

__device__ ECD d_ecd;
// CHECK: @d_ecd = addrspace(1) externally_initialized global %struct.ECD zeroinitializer,
__shared__ ECD s_ecd;
// CHECK: @s_ecd = addrspace(3) global %struct.ECD undef,
__constant__ ECD c_ecd;
// CHECK: @c_ecd = addrspace(4) externally_initialized global %struct.ECD zeroinitializer,

// empty templated constructor -- allowed with no arguments
__device__ ETC d_etc;
// CHECK: @d_etc = addrspace(1) externally_initialized global %struct.ETC zeroinitializer,
__shared__ ETC s_etc;
// CHECK: @s_etc = addrspace(3) global %struct.ETC undef,
__constant__ ETC c_etc;
// CHECK: @c_etc = addrspace(4) externally_initialized global %struct.ETC zeroinitializer,

__device__ NCFS d_ncfs;
// CHECK: @d_ncfs = addrspace(1) externally_initialized global %struct.NCFS { i32 3 }
__constant__ NCFS c_ncfs;
// CHECK: @c_ncfs = addrspace(4) externally_initialized global %struct.NCFS { i32 3 }

// Regular base class -- allowed
__device__ T_B_T d_t_b_t;
// CHECK: @d_t_b_t = addrspace(1) externally_initialized global %struct.T_B_T zeroinitializer,
__shared__ T_B_T s_t_b_t;
// CHECK: @s_t_b_t = addrspace(3) global %struct.T_B_T undef,
__constant__ T_B_T c_t_b_t;
// CHECK: @c_t_b_t = addrspace(4) externally_initialized global %struct.T_B_T zeroinitializer,

// Incapsulated object of allowed class -- allowed
__device__ T_F_T d_t_f_t;
// CHECK: @d_t_f_t = addrspace(1) externally_initialized global %struct.T_F_T zeroinitializer,
__shared__ T_F_T s_t_f_t;
// CHECK: @s_t_f_t = addrspace(3) global %struct.T_F_T undef,
__constant__ T_F_T c_t_f_t;
// CHECK: @c_t_f_t = addrspace(4) externally_initialized global %struct.T_F_T zeroinitializer,

// array of allowed objects -- allowed
__device__ T_FA_T d_t_fa_t;
// CHECK: @d_t_fa_t = addrspace(1) externally_initialized global %struct.T_FA_T zeroinitializer,
__shared__ T_FA_T s_t_fa_t;
// CHECK: @s_t_fa_t = addrspace(3) global %struct.T_FA_T undef,
__constant__ T_FA_T c_t_fa_t;
// CHECK: @c_t_fa_t = addrspace(4) externally_initialized global %struct.T_FA_T zeroinitializer,


// Calling empty base class initializer is OK
__device__ EC_I_EC d_ec_i_ec;
// CHECK: @d_ec_i_ec = addrspace(1) externally_initialized global %struct.EC_I_EC zeroinitializer,
__shared__ EC_I_EC s_ec_i_ec;
// CHECK: @s_ec_i_ec = addrspace(3) global %struct.EC_I_EC undef,
__constant__ EC_I_EC c_ec_i_ec;
// CHECK: @c_ec_i_ec = addrspace(4) externally_initialized global %struct.EC_I_EC zeroinitializer,

// CHECK: @_ZZ2dfvE4s_ec = internal addrspace(3) global %struct.EC undef
// CHECK: @_ZZ2dfvE5s_etc = internal addrspace(3) global %struct.ETC undef

// CHECK: @_ZZ2dfvE11const_array = internal addrspace(4) constant [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5]
// CHECK: @_ZZ2dfvE9const_int = internal addrspace(4) constant i32 123

// We should not emit global initializers for device-side variables.
// CHECK-NOT: @__cxx_global_var_init

// Make sure that initialization restrictions do not apply to local
// variables.
__device__ void df() {
  // NVPTX:  %[[ec:.*]] = alloca %struct.EC
  // NVPTX:  %[[ed:.*]] = alloca %struct.ED
  // NVPTX:  %[[ecd:.*]] = alloca %struct.ECD
  // NVPTX:  %[[etc:.*]] = alloca %struct.ETC
  // NVPTX:  %[[uc:.*]] = alloca %struct.UC
  // NVPTX:  %[[ud:.*]] = alloca %struct.UD
  // NVPTX:  %[[eci:.*]] = alloca %struct.ECI
  // NVPTX:  %[[nec:.*]] = alloca %struct.NEC
  // NVPTX:  %[[ned:.*]] = alloca %struct.NED
  // NVPTX:  %[[ncv:.*]] = alloca %struct.NCV
  // NVPTX:  %[[vd:.*]] = alloca %struct.VD
  // NVPTX:  %[[ncf:.*]] = alloca %struct.NCF
  // NVPTX:  %[[ncfs:.*]] = alloca %struct.NCFS
  // NVPTX:  %[[utc:.*]] = alloca %struct.UTC
  // NVPTX:  %[[netc:.*]] = alloca %struct.NETC
  // NVPTX:  %[[ec_i_ec:.*]] = alloca %struct.EC_I_EC
  // NVPTX:  %[[ec_i_ec1:.*]] = alloca %struct.EC_I_EC1
  // NVPTX:  %[[t_v_t:.*]] = alloca %struct.T_V_T
  // NVPTX:  %[[t_b_nec:.*]] = alloca %struct.T_B_NEC
  // NVPTX:  %[[t_f_nec:.*]] = alloca %struct.T_F_NEC
  // NVPTX:  %[[t_fa_nec:.*]] = alloca %struct.T_FA_NEC
  // NVPTX:  %[[t_b_ned:.*]] = alloca %struct.T_B_NED
  // NVPTX:  %[[t_f_ned:.*]] = alloca %struct.T_F_NED
  // NVPTX:  %[[t_fa_ned:.*]] = alloca %struct.T_FA_NED
  // AMDGCN:  %[[ec:.*]] = addrspacecast %struct.EC addrspace(5)* %ec to %struct.EC*
  // AMDGCN:  %[[ed:.*]] = addrspacecast %struct.ED addrspace(5)* %ed to %struct.ED*
  // AMDGCN:  %[[ecd:.*]] = addrspacecast %struct.ECD addrspace(5)* %ecd to %struct.ECD*
  // AMDGCN:  %[[etc:.*]] = addrspacecast %struct.ETC addrspace(5)* %etc to %struct.ETC*
  // AMDGCN:  %[[uc:.*]] = addrspacecast %struct.UC addrspace(5)* %uc to %struct.UC*
  // AMDGCN:  %[[ud:.*]] = addrspacecast %struct.UD addrspace(5)* %ud to %struct.UD*
  // AMDGCN:  %[[eci:.*]] = addrspacecast %struct.ECI addrspace(5)* %eci to %struct.ECI*
  // AMDGCN:  %[[nec:.*]] = addrspacecast %struct.NEC addrspace(5)* %nec to %struct.NEC*
  // AMDGCN:  %[[ned:.*]] = addrspacecast %struct.NED addrspace(5)* %ned to %struct.NED*
  // AMDGCN:  %[[ncv:.*]] = addrspacecast %struct.NCV addrspace(5)* %ncv to %struct.NCV*
  // AMDGCN:  %[[vd:.*]] = addrspacecast %struct.VD addrspace(5)* %vd to %struct.VD*
  // AMDGCN:  %[[ncf:.*]] = addrspacecast %struct.NCF addrspace(5)* %ncf to %struct.NCF*
  // AMDGCN:  %[[ncfs:.*]] = addrspacecast %struct.NCFS addrspace(5)* %ncfs to %struct.NCFS*
  // AMDGCN:  %[[utc:.*]] = addrspacecast %struct.UTC addrspace(5)* %utc to %struct.UTC*
  // AMDGCN:  %[[netc:.*]] = addrspacecast %struct.NETC addrspace(5)* %netc to %struct.NETC*
  // AMDGCN:  %[[ec_i_ec:.*]] = addrspacecast %struct.EC_I_EC addrspace(5)* %ec_i_ec to %struct.EC_I_EC*
  // AMDGCN:  %[[ec_i_ec1:.*]] = addrspacecast %struct.EC_I_EC1 addrspace(5)* %ec_i_ec1 to %struct.EC_I_EC1*
  // AMDGCN:  %[[t_v_t:.*]] = addrspacecast %struct.T_V_T addrspace(5)* %t_v_t to %struct.T_V_T*
  // AMDGCN:  %[[t_b_nec:.*]] = addrspacecast %struct.T_B_NEC addrspace(5)* %t_b_nec to %struct.T_B_NEC*
  // AMDGCN:  %[[t_f_nec:.*]] = addrspacecast %struct.T_F_NEC addrspace(5)* %t_f_nec to %struct.T_F_NEC*
  // AMDGCN:  %[[t_fa_nec:.*]] = addrspacecast %struct.T_FA_NEC addrspace(5)* %t_fa_nec to %struct.T_FA_NEC*
  // AMDGCN:  %[[t_b_ned:.*]] = addrspacecast %struct.T_B_NED addrspace(5)* %t_b_ned to %struct.T_B_NED*
  // AMDGCN:  %[[t_f_ned:.*]] = addrspacecast %struct.T_F_NED addrspace(5)* %t_f_ned to %struct.T_F_NED*
  // AMDGCN:  %[[t_fa_ned:.*]] = addrspacecast %struct.T_FA_NED addrspace(5)* %t_fa_ned to %struct.T_FA_NED*

  T t;
  // CHECK-NOT: call
  EC ec;
  // CHECK:  call void @_ZN2ECC1Ev(%struct.EC* %[[ec]])
  ED ed;
  // CHECK-NOT: call
  ECD ecd;
  // CHECK:  call void @_ZN3ECDC1Ev(%struct.ECD* %[[ecd]])
  ETC etc;
  // CHECK:  call void @_ZN3ETCC1IJEEEDpT_(%struct.ETC* %[[etc]])
  UC uc;
  // undefined constructor -- not allowed
  // CHECK:  call void @_ZN2UCC1Ev(%struct.UC* %[[uc]])
  UD ud;
  // undefined destructor -- not allowed
  // CHECK-NOT: call
  ECI eci;
  // empty constructor w/ initializer list -- not allowed
  // CHECK:  call void @_ZN3ECIC1Ev(%struct.ECI* %[[eci]])
  NEC nec;
  // non-empty constructor -- not allowed
  // CHECK:  call void @_ZN3NECC1Ev(%struct.NEC* %[[nec]])
  // non-empty destructor -- not allowed
  NED ned;
  // no-constructor,  virtual method -- not allowed
  // CHECK:  call void @_ZN3NCVC1Ev(%struct.NCV* %[[ncv]])
  NCV ncv;
  // CHECK-NOT: call
  VD vd;
  // CHECK:  call void @_ZN2VDC1Ev(%struct.VD* %[[vd]])
  NCF ncf;
  // CHECK:   call void @_ZN3NCFC1Ev(%struct.NCF* %[[ncf]])
  NCFS ncfs;
  // CHECK:  call void @_ZN4NCFSC1Ev(%struct.NCFS* %[[ncfs]])
  UTC utc;
  // CHECK:  call void @_ZN3UTCC1IJEEEDpT_(%struct.UTC* %[[utc]])
  NETC netc;
  // CHECK:  call void @_ZN4NETCC1IJEEEDpT_(%struct.NETC* %[[netc]])
  T_B_T t_b_t;
  // CHECK-NOT: call
  T_F_T t_f_t;
  // CHECK-NOT: call
  T_FA_T t_fa_t;
  // CHECK-NOT: call
  EC_I_EC ec_i_ec;
  // CHECK:  call void @_ZN7EC_I_ECC1Ev(%struct.EC_I_EC* %[[ec_i_ec]])
  EC_I_EC1 ec_i_ec1;
  // CHECK:  call void @_ZN8EC_I_EC1C1Ev(%struct.EC_I_EC1* %[[ec_i_ec1]])
  T_V_T t_v_t;
  // CHECK:  call void @_ZN5T_V_TC1Ev(%struct.T_V_T* %[[t_v_t]])
  T_B_NEC t_b_nec;
  // CHECK:  call void @_ZN7T_B_NECC1Ev(%struct.T_B_NEC* %[[t_b_nec]])
  T_F_NEC t_f_nec;
  // CHECK:  call void @_ZN7T_F_NECC1Ev(%struct.T_F_NEC* %[[t_f_nec]])
  T_FA_NEC t_fa_nec;
  // CHECK:  call void @_ZN8T_FA_NECC1Ev(%struct.T_FA_NEC* %[[t_fa_nec]])
  T_B_NED t_b_ned;
  // CHECK-NOT: call
  T_F_NED t_f_ned;
  // CHECK-NOT: call
  T_FA_NED t_fa_ned;
  // CHECK-NOT: call
  static __shared__ EC s_ec;
  // CHECK-NOT: call void @_ZN2ECC1Ev(%struct.EC* addrspacecast (%struct.EC addrspace(3)* @_ZZ2dfvE4s_ec to %struct.EC*))
  static __shared__ ETC s_etc;
  // CHECK-NOT: call void @_ZN3ETCC1IJEEEDpT_(%struct.ETC* addrspacecast (%struct.ETC addrspace(3)* @_ZZ2dfvE5s_etc to %struct.ETC*))

  static const int const_array[] = {1, 2, 3, 4, 5};
  static const int const_int = 123;

  // anchor point separating constructors and destructors
  df(); // CHECK: call void @_Z2dfv()

  // Verify that we only call non-empty destructors
  // CHECK-NEXT: call void @_ZN8T_FA_NEDD1Ev(%struct.T_FA_NED* %[[t_fa_ned]])
  // CHECK-NEXT: call void @_ZN7T_F_NEDD1Ev(%struct.T_F_NED* %[[t_f_ned]])
  // CHECK-NEXT: call void @_ZN7T_B_NEDD1Ev(%struct.T_B_NED* %[[t_b_ned]])
  // CHECK-NEXT: call void @_ZN2VDD1Ev(%struct.VD* %[[vd]])
  // CHECK-NEXT: call void @_ZN3NEDD1Ev(%struct.NED* %[[ned]])
  // CHECK-NEXT: call void @_ZN2UDD1Ev(%struct.UD* %[[ud]])
  // CHECK-NEXT: call void @_ZN3ECDD1Ev(%struct.ECD* %[[ecd]])
  // CHECK-NEXT: call void @_ZN2EDD1Ev(%struct.ED* %[[ed]])

  // CHECK-NEXT: ret void
}

// We should not emit global init function.
// CHECK-NOT: @_GLOBAL__sub_I
