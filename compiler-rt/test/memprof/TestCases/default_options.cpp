// RUN: %clangxx_memprof -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

const char *kMemProfDefaultOptions = "verbosity=1 help=1";

extern "C" const char *__memprof_default_options() {
  // CHECK: Available flags for MemProfiler:
  return kMemProfDefaultOptions;
}

int main() {
  return 0;
}
