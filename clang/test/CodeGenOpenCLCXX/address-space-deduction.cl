// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -O0 -emit-llvm -o - | FileCheck %s -check-prefixes=COMMON,PTR
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -O0 -emit-llvm -o - -DREF | FileCheck %s -check-prefixes=COMMON,REF

#ifdef REF
#define PTR &
#define ADR(x) x
#else
#define PTR *
#define ADR(x) &x
#endif

//COMMON: @glob = addrspace(1) global i32
int glob;
//PTR: @glob_p = addrspace(1) global i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @glob to i32 addrspace(4)*)
//REF: @glob_p = addrspace(1) constant i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @glob to i32 addrspace(4)*)
int PTR glob_p = ADR(glob);

//COMMON: @_ZZ3fooi{{P|R}}U3AS4iE6loc_st = internal addrspace(1) global i32
//PTR: @_ZZ3fooiPU3AS4iE8loc_st_p = internal addrspace(1) global i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @_ZZ3fooiPU3AS4iE6loc_st to i32 addrspace(4)*)
//REF: @_ZZ3fooiRU3AS4iE8loc_st_p = internal addrspace(1) constant i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @_ZZ3fooiRU3AS4iE6loc_st to i32 addrspace(4)*)
//COMMON: @loc_ext_p = external addrspace(1) {{global|constant}} i32 addrspace(4)*
//COMMON: @loc_ext = external addrspace(1) global i32

//COMMON: define spir_func i32 @_Z3fooi{{P|R}}U3AS4i(i32 %par, i32 addrspace(4)*{{.*}} %par_p)
int foo(int par, int PTR par_p){
  //COMMON: %loc = alloca i32
  int loc;
  //COMMON: %loc_p = alloca i32 addrspace(4)*
  //COMMON: %loc_p_const = alloca i32*
  //COMMON: [[GAS:%[._a-z0-9]*]] = addrspacecast i32* %loc to i32 addrspace(4)*
  //COMMON: store i32 addrspace(4)* [[GAS]], i32 addrspace(4)** %loc_p
  int PTR loc_p = ADR(loc);
  //COMMON: store i32* %loc, i32** %loc_p_const
  const __private int PTR loc_p_const = ADR(loc);

  // CHECK directives for the following code are located above.
  static int loc_st;
  static int PTR loc_st_p = ADR(loc_st);
  extern int loc_ext;
  extern int PTR loc_ext_p;
  (void)loc_ext_p;
  return loc_ext;
}
