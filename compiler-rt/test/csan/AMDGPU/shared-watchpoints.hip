// RUN: %clang_hip_csan -O2 -fPIC -shared -Xarch_host \
// RUN:   -fno-sanitize=concurrency -DLIB_A %s -o %t.a.so %hip_libs
// RUN: %clang_hip_csan -O2 -fPIC -shared -Xarch_host \
// RUN:   -fno-sanitize=concurrency -DLIB_B %s -o %t.b.so %hip_libs
// RUN: %clangxx -x c++ %s -x none %t.a.so %t.b.so \
// RUN:   -Wl,-rpath -Wl,%T -o %t
// RUN: %run %t | FileCheck %s

#if defined(LIB_A) || defined(LIB_B)

#include <stdint.h>

extern "C" __device__ uint64_t *__csan_watchpoint_table;
extern "C" int hipMemcpy(void *, const void *, unsigned long, int);
static constexpr int hipMemcpyDeviceToHost = 2;

#if defined(LIB_A)
#define PRINT_TABLE print_table_a
#define TABLE_LABEL "TABLE_A"
#else
#define PRINT_TABLE print_table_b
#define TABLE_LABEL "TABLE_B"
#endif

__global__ void read_table(uint64_t *out) {
  *out = reinterpret_cast<uint64_t>(__csan_watchpoint_table);
}

extern "C" uint64_t PRINT_TABLE() {
  uint64_t *device_out;
  uint64_t result = 0;
  if (hipMalloc(reinterpret_cast<void **>(&device_out), sizeof(result)))
    return 0;
  read_table<<<1, 1>>>(device_out);
  if (hipMemcpy(&result, device_out, sizeof(result), hipMemcpyDeviceToHost))
    result = 0;
  hipFree(device_out);
  return result;
}

#else

#include <stdint.h>
#include <stdio.h>

extern "C" uint64_t print_table_a();
extern "C" uint64_t print_table_b();

int main() {
  uint64_t a = print_table_a();
  uint64_t b = print_table_b();
  printf("TABLE: %#llx %#llx\n", static_cast<unsigned long long>(a),
         static_cast<unsigned long long>(b));
  return !a || a != b;
}

// CHECK: TABLE: [[TABLE:0x[0-9a-f]+]] [[TABLE]]

#endif
