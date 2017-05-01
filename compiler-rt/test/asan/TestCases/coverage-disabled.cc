// Test that no data is collected without a runtime flag.
//
// RUN: %clangxx_asan -fsanitize-coverage=func %s -o %t
//
// RUN: rm -rf %T/coverage-disabled
//
// RUN: mkdir -p %T/coverage-disabled/normal
// RUN: %env_asan_opts=coverage_direct=0:coverage_dir='"%T/coverage-disabled/normal"':verbosity=1 %run %t
// RUN: not %sancov print %T/coverage-disabled/normal/*.sancov 2>&1
//
// UNSUPPORTED: android

int main(int argc, char **argv) {
  return 0;
}
