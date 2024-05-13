// REQUIRES: aarch64-target-arch
// REQUIRES: native-run
// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_aarch64
int main(void) {
  if (__builtin_cpu_supports("fp+simd+pmull+sha2+crc")) {
    if (__builtin_cpu_supports("fp") && __builtin_cpu_supports("simd") &&
        __builtin_cpu_supports("pmull") && __builtin_cpu_supports("sha2") &&
        __builtin_cpu_supports("crc")) {
      return 0;
    } else {
      // Something wrong in feature detection
      return 1;
    }
  }
  return 0;
}
