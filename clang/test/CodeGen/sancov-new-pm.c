// Test that SanitizerCoverage works under the new pass manager.
// RUN: %clang -target x86_64-linux-gnu -fsanitize=fuzzer %s -fexperimental-new-pass-manager -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-O0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=fuzzer %s -fexperimental-new-pass-manager -O2 -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-O2

extern void *memcpy(void *, const void *, unsigned long);
extern int printf(const char *restrict, ...);

int LLVMFuzzerTestOneInput(const unsigned char *data, unsigned long size) {
  unsigned char buf[4];

  if (size < 8)
    return 0;

  if (data[0] == 'h' && data[1] == 'i' && data[2] == '!') {
    memcpy(buf, data, size);
    printf("test: %.2X\n", buf[0]);
  }

  return 0;
}

// CHECK-DAG: declare void @__sanitizer_cov_pcs_init(i64*, i64*)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_pc_indir(i64)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_cmp1(i8 zeroext, i8 zeroext)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_cmp2(i16 zeroext, i16 zeroext)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_cmp4(i32 zeroext, i32 zeroext)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_cmp8(i64, i64)
// CHECK-O2-NOT: declare void @__sanitizer_cov_trace_const_cmp1(i8 zeroext, i8 zeroext)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_const_cmp2(i16 zeroext, i16 zeroext)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_const_cmp4(i32 zeroext, i32 zeroext)
// CHECK-O2-NOT: declare void @__sanitizer_cov_trace_const_cmp8(i64, i64)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_div4(i32 zeroext)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_div8(i64)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_gep(i64)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_switch(i64, i64*)
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_pc()
// CHECK-O0-DAG: declare void @__sanitizer_cov_trace_pc_guard(i32*)
