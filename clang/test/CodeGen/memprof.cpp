// Test if memprof instrumentation and use pass are invoked.
//
// Instrumentation:
// Ensure Pass MemProfilerPass and ModuleMemProfilerPass are invoked.
// RUN: %clang_cc1 -O2 -fmemory-profile %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=INSTRUMENT
// INSTRUMENT: Running pass: MemProfilerPass on main
// INSTRUMENT: Running pass: ModuleMemProfilerPass on [module]

// Avoid failures on big-endian systems that can't read the raw profile properly
// REQUIRES: x86_64-linux

// TODO: Use text profile inputs once that is available for memprof.
//
// To update the inputs below, run Inputs/update_memprof_inputs.sh
// RUN: llvm-profdata merge %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.memprofdata

// Profile use:
// Ensure Pass PGOInstrumentationUse is invoked with the memprof-only profile.
// RUN: %clang_cc1 -O2 -fmemory-profile-use=%t.memprofdata %s -fdebug-pass-manager  -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=USE
// USE: Running pass: MemProfUsePass on [module]

char *foo() {
  return new char[10];
}
int main() {
  char *a = foo();
  delete[] a;
  return 0;
}
