// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -target-cpu gfx941 -fopenmp-target-fast -fopenmp-target-ignore-env-vars -fno-openmp-target-big-jump-loop -fopenmp-target-new-runtime -fopenmp-assume-no-thread-state -fopenmp-assume-no-nested-parallelism -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=CHECK1

typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;

typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;

typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;

typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;

extern int printf (const char *__restrict __format, ...);

    typedef enum omp_sync_hint_t {
        omp_sync_hint_none = 0,
        omp_lock_hint_none = omp_sync_hint_none,
        omp_sync_hint_uncontended = 1,
        omp_lock_hint_uncontended = omp_sync_hint_uncontended,
        omp_sync_hint_contended = (1<<1),
        omp_lock_hint_contended = omp_sync_hint_contended,
        omp_sync_hint_nonspeculative = (1<<2),
        omp_lock_hint_nonspeculative = omp_sync_hint_nonspeculative,
        omp_sync_hint_speculative = (1<<3),
        omp_lock_hint_speculative = omp_sync_hint_speculative,
        kmp_lock_hint_hle = (1<<16),
        kmp_lock_hint_rtm = (1<<17),
        kmp_lock_hint_adaptive = (1<<18),
        AMD_fast_fp_atomics = (1<<19),
        AMD_unsafe_fp_atomics = AMD_fast_fp_atomics,
        ompx_fast_fp_atomics = AMD_fast_fp_atomics,
        ompx_unsafe_fp_atomics = AMD_fast_fp_atomics,
        AMD_safe_fp_atomics = (1<<20),
        ompx_safe_fp_atomics = AMD_safe_fp_atomics
    } omp_sync_hint_t;

int main() {
  { // add i8: cas loop via clang codegen
    int8_t add_i8 = 0;
    int8_t h_add_i8 = 0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: add_i8)
    for(int i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      add_i8 += i;
    }

    for(int i = 0; i < 64; i++) {   
      h_add_i8 += i;
    }
    if (add_i8 != h_add_i8) {     
      printf("Err for int8_t atomic add got %d, expected %d\n", add_i8, h_add_i8);
      return 1;
    }
  }

  { // add i32: atomic_add
    int add_i32  = 0;
    int h_add_i32  = 0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: add_i32)
    for(int i = 0; i < 64; i++) {
      // CHECK1: atomicrmw
      #pragma omp atomic
      add_i32 += i;
    }

    for(int i = 0; i < 64; i++) {   
      h_add_i32 += i;
    }
    if (add_i32 != h_add_i32) {     
      printf("Err for int32_t atomic add got %d, expected %d\n", add_i32, h_add_i32);
      return 1;
    }
  }

  { // add long: atomic_add 
    int64_t add_i64 = 0;
    int64_t h_add_i64 = 0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: add_i64)
    for(int64_t i = 0; i < 64; i++) {
      // CHECK1: atomicrmw
      #pragma omp atomic
      add_i64 += i;
    }
    for(int i = 0; i < 64; i++) {   
      h_add_i64 += i;
    }
    if (add_i64 != h_add_i64) {     
      printf("Err for int64_t atomic add got %ld, expected %ld\n", add_i64, h_add_i64);
      return 1;
    }
  }

  { // add float: cas loop via clang codegen
    float add_float = 0.0f;
    float h_add_float = 0.0f;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: add_float)
    for(int i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      add_float += (float)i;
    }
    for(int i = 0; i < 64; i++) {   
      h_add_float += i;
    }
    if (add_float != h_add_float) {
      printf("Err for int64_t atomic add got %f, expected %f\n", add_float, h_add_float);
      return 1;
    }
  }

  { // add double: cas loop via clang codegen
    double add_double = 0.0f;
    double h_add_double = 0.0f;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: add_double)
    for(int i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      add_double += (double)i;
    }
    for(int i = 0; i < 64; i++) {   
      h_add_double += i;
    }
    if (add_double != h_add_double) {
      printf("Err for double atomic add got %lf, expected %lf\n", add_double, h_add_double);
      return 1;
    }
  }

  { // sub int: cas loop via clang codegen
    int sub_int = 2016;
    int h_sub_int = 2016;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: sub_int)
    for(int i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      sub_int -= i;
    }
    for(int i = 0; i < 64; i++) {   
      h_sub_int -= i;
    }
    if (sub_int != h_sub_int) {
      printf("Err for int atomic sub got %d, expected %d\n", sub_int, h_sub_int);
      return 1;
    }
  }

  { // sub int32_t: cas loop via clang codegen
    int32_t sub_int32 = 2016;
    int32_t h_sub_int32 = 2016;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: sub_int32)
    for(int32_t i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      sub_int32 -= i;
    }
    for(int i = 0; i < 64; i++) {   
      h_sub_int32 -= i;
    }
    if (sub_int32 != h_sub_int32) {
      printf("Err for int32 atomic sub got %d, expected %d\n", sub_int32, h_sub_int32);
      return 1;
    }
  }

  { // sub long: cas loop via clang codegen
    long sub_long = 2016;
    long h_sub_long = 2016;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: sub_long)
    for(long i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      sub_long -= i;
    }
    for(int i = 0; i < 64; i++) {   
      h_sub_long -= i;
    }
    if (sub_long != h_sub_long) {
      printf("Err for long atomic sub got %ld, expected %ld\n", sub_long, h_sub_long);
      return 1;
    }
  }

  { // sub int64_t: cas loop via clang codegen
    int64_t sub_int64 = 2016;
    int64_t h_sub_int64 = 2016;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: sub_int64)
    for(int32_t i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      sub_int64 -= i;
    }
    for(int i = 0; i < 64; i++) {   
      h_sub_int64 -= i;
    }
    if (sub_int64 != h_sub_int64) {
      printf("Err for int64_t atomic sub got %ld, expected %ld\n", sub_int64, h_sub_int64);
      return 1;
    }
  }

  { // compare min (i8): cas loop via backend
    int8_t min_i8 = 65;
    int8_t h_min_i8 = 65;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: min_i8)
    for(int i = 0; i < 64; i++) {
      // CHECK1: atomicrmw
      #pragma omp atomic compare
      min_i8 = i < min_i8 ? i : min_i8;
    }
    for(int i = 0; i < 64; i++) {   
      h_min_i8 = i < h_min_i8 ? i : h_min_i8;
    }
    if (min_i8 != h_min_i8) {
      printf("Err for int8_t atomic min got %d, expected %d\n", min_i8, h_min_i8);
      return 1;
    }
  }

  { // compare min (i32): cas loop via runtime
    int min_i32 = 123456;
    int h_min_i32 = 123456;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: min_i32)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMin_int32_t(
      #pragma omp atomic compare
      min_i32 = i < min_i32 ? i : min_i32;
    }
    for(int i = 0; i < 64; i++) {   
      h_min_i32 = i < h_min_i32 ? i : h_min_i32;
    }
    if (min_i32 != h_min_i32) {
      printf("Err for int32_t atomic min got %d, expected %d\n", min_i32, h_min_i32);
      return 1;
    }
  }

  { // compare min (ui32): cas loop via backend
    unsigned int min_ui32 = 128;
    unsigned int h_min_ui32 = 128;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: min_ui32)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMin_uint32_t(
      #pragma omp atomic compare
      min_ui32 = i < min_ui32 ? i : min_ui32;
    }
    for(int i = 0; i < 64; i++) {   
      h_min_ui32 = i < h_min_ui32 ? i : h_min_ui32;
    }
    if (min_ui32 != h_min_ui32) {
      printf("Err for uint32_t atomic min got %d, expected %d\n", min_ui32, h_min_ui32);
      return 1;
    }
  }
  
  { // compare min (i64): cas loop via runtime
    int64_t min_i64 = 123456;
    int64_t h_min_i64 = 123456;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: min_i64)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMin_int64_t(
      #pragma omp atomic compare
      min_i64 = i < min_i64 ? i : min_i64;
    }
    for(int i = 0; i < 64; i++) {   
      h_min_i64 = i < h_min_i64 ? i : h_min_i64;
    }
    if (min_i64 != h_min_i64) {
      printf("Err for int64_t atomic min got %ld, expected %ld\n", min_i64, h_min_i64);
      return 1;
    }
  }

  { // compare min (ui64): cas loop via backend
    uint64_t min_ui64 = 128;
    uint64_t h_min_ui64 = 128;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: min_ui64)
    for(uint64_t i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMin_uint64_t(
      #pragma omp atomic compare
      min_ui64 = i < min_ui64 ? i : min_ui64;
    }
    for(int i = 0; i < 64; i++) {   
      h_min_ui64 = i < h_min_ui64 ? i : h_min_ui64;
    }
    if (min_ui64 != h_min_ui64) {
      printf("Err for uint64_t atomic min got %d, expected %d\n", min_ui64, h_min_ui64);
      return 1;
    }
  }

  { // compare min (double): cas loop via runtime
    double min_d = 123456.0;
    double h_min_d = 123456.0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: min_d)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMin_double(
      #pragma omp atomic compare hint(AMD_fast_fp_atomics)
      min_d = (double)i < min_d ? (double)i : min_d;
    }
    for(int i = 0; i < 64; i++) {   
      h_min_d = i < h_min_d ? i : h_min_d;
    }
    if (min_d != h_min_d) {
      printf("Err for double atomic min got %lf, expected %lf\n", min_d, h_min_d);
      return 1;
    }
  }

  { // compare min (float): cas loop via runtime
    float min_f = 123456.0f;
    float h_min_f = 123456.0f;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: min_f)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMin_float(
      #pragma omp atomic compare hint(AMD_fast_fp_atomics)
      min_f = (float)i < min_f ? (float)i : min_f;
    }
    for(int i = 0; i < 64; i++) {   
      h_min_f = i < h_min_f ? i : h_min_f;
    }
    if (min_f != h_min_f) {
      printf("Err for float atomic min got %f, expected %f\n", min_f, h_min_f);
      return 1;
    }
  }

  { // compare max (int32_t): cas loop via runtime
    int max_i32 = -1;
    int h_max_i32 = -1;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: max_i32)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMax_int32_t(
      #pragma omp atomic compare   
      max_i32 = i > max_i32 ? i : max_i32;
    }
    for(int i = 0; i < 64; i++) {   
      h_max_i32 = i > h_max_i32 ? i : h_max_i32;
    }
    if (max_i32 != h_max_i32) {
      printf("Err for int32_t atomic max got %d, expected %d\n", max_i32, h_max_i32);
      return 1;
    }
  }

  { // compare max (ui32): cas loop via backend
    uint32_t max_ui32 = 128;
    uint32_t h_max_ui32 = 128;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: max_ui32)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMax_uint32_t(
      #pragma omp atomic compare
      max_ui32 = i > max_ui32 ? i : max_ui32;
    }
    for(int i = 0; i < 64; i++) {   
      h_max_ui32 = i > h_max_ui32 ? i : h_max_ui32;
    }
    if (max_ui32 != h_max_ui32) {
      printf("Err for uint32_t atomic max got %d, expected %d\n", max_ui32, h_max_ui32);
      return 1;
    }
  }

  { // compare max (i64): cas loop via runtime
    int64_t max_i64 = -1;
    int64_t h_max_i64 = -1;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: max_i64)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMax_int64_t(
      #pragma omp atomic compare   
      max_i64 = i > max_i64 ? i : max_i64;
    }
    for(int i = 0; i < 64; i++) {   
      h_max_i64 = i > h_max_i64 ? i : h_max_i64;
    }
    if (max_i64 != h_max_i64) {
      printf("Err for int64_t atomic max got %ld, expected %ld\n", max_i64, h_max_i64);
      return 1;
    }
  }

  { // compare max (ui64): cas loop via backend
    uint64_t max_ui64 = 128;
    uint64_t h_max_ui64 = 128;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: max_ui64)
    for(uint64_t i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMax_uint64_t(
      #pragma omp atomic compare
      max_ui64 = i > max_ui64 ? i : max_ui64;
    }
    for(int i = 0; i < 64; i++) {   
      h_max_ui64 = i > h_max_ui64 ? i : h_max_ui64;
    }
    if (max_ui64 != h_max_ui64) {
      printf("Err for uint64_t atomic max got %d, expected %d\n", max_ui64, h_max_ui64);
      return 1;
    }
  }

  { // compare max (double): cas loop via runtime
    double max_d = -1.0;
    double h_max_d = -1.0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: max_d)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMax_double(
      #pragma omp atomic compare hint(AMD_fast_fp_atomics)
      max_d = (double)i > max_d ? (double)i : max_d;
    }
    for(int i = 0; i < 64; i++) {   
      h_max_d = i > h_max_d ? i : h_max_d;
    }
    if (max_d != h_max_d) {
      printf("Err for doublet atomic max got %lf, expected %lf\n", max_d, h_max_d);
      return 1;
    }
  }

  { // compare max (float): cas loop via runtime
    float max_f = -1.0;
    float h_max_f = -1.0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: max_f)
    for(int i = 0; i < 64; i++) {
      // CHECK1: __kmpc_atomicCASLoopMax_float(
      #pragma omp atomic compare hint(AMD_fast_fp_atomics)
      max_f = (float)i > max_f ? (float)i : max_f;
    }
    for(int i = 0; i < 64; i++) {   
      h_max_f= i > h_max_f ? i : h_max_f;
    }
    if (max_f != h_max_f) {
      printf("Err for float atomic max got %f, expected %f\n", max_f, h_max_f);
      return 1;
    }
  }

  { // and i32: cas loop via clang codegen
    int and_i32 = 0;
    int h_and_i32 = 0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: and_i32)
    for(int i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      and_i32 = and_i32 & i;
    }
    for(int i = 0; i < 64; i++) {   
      h_and_i32 = h_and_i32 & i;
    }
    if (and_i32 != h_and_i32) {
      printf("Err for int32_t atomic and got %d, expected %d\n", and_i32, h_and_i32);
      return 1;
    }
  }

  { // and i64: cas loop via clang codegen
    int64_t and_i64 = 0;
    int64_t h_and_i64 = 0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: and_i64)
    for(int64_t i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      and_i64 = and_i64 & i;
    }
    for(int i = 0; i < 64; i++) {
      h_and_i64 = h_and_i64 & i;
    }
    if (and_i64 != h_and_i64) {
      printf("Err for int64_t atomic and got %ld, expected %ld\n", and_i64, h_and_i64);
      return 1;
    }
  }

  { // or i32: cas loop via clang codegen
    int or_i32 = 0;
    int h_or_i32 = 0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: or_i32)
    for(int i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      or_i32 = or_i32 | i;
    }
    for(int i = 0; i < 64; i++) {
      h_or_i32 = h_or_i32 | i;
    }
    if (or_i32 != h_or_i32) {
      printf("Err for int32_t atomic or got %d, expected %d\n", or_i32, h_or_i32);
      return 1;
    }
  }

  { // or i64: cas loop via clang codegen
    int64_t or_i64 = 0;
    int64_t h_or_i64 = 0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: or_i64)
    for(int64_t i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      or_i64 = or_i64 | i;
    }
    for(int i = 0; i < 64; i++) {
      h_or_i64 = h_or_i64 | i;
    }
    if (or_i64 != h_or_i64) {
      printf("Err for int64_t atomic or got %ld, expected %ld\n", or_i64, h_or_i64);
      return 1;
    }
  }

  { // xor i32: cas loop via clang codegen
    int xor_i32 = 0;
    int h_xor_i32 = 0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: xor_i32)
    for(int i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      xor_i32 = xor_i32 ^ i;
    }
    for(int i = 0; i < 64; i++) {
      h_xor_i32 = h_xor_i32 ^ i;
    }
    if (xor_i32 != h_xor_i32) {
      printf("Err for int32_t atomic xor got %d, expected %d\n", xor_i32, h_xor_i32);
      return 1;
    }
  }

  { // xor i64: cas loop via clang codegen
    int64_t xor_i64 = 0;
    int64_t h_xor_i64 = 0;

    // CHECK1-LABEL: define {{.+}}_offloading_{{.+}}_l
    #pragma omp target teams distribute parallel for map(tofrom: xor_i64)
    for(int64_t i = 0; i < 64; i++) {
      // CHECK1: atomic_cont:
      // CHECK1: cmpxchg
      #pragma omp atomic
      xor_i64 = xor_i64 ^ i;
    }
    for(int i = 0; i < 64; i++) {
      h_xor_i64 = h_xor_i64 ^ i;
    }
    if (xor_i64 != h_xor_i64) {
      printf("Err for int64_t atomic xor got %ld, expected %ld\n", xor_i64, h_xor_i64);
      return 1;
    }
  }

  return 0;
}
