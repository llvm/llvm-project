// Test if memprof instrumentation and use pass are invoked.
//
// Instrumentation:
// Ensure Pass MemProfilerPass and ModuleMemProfilerPass are invoked.
// RUN: %clang_cc1 -O2 -fmemory-profile %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=INSTRUMENT
// INSTRUMENT: Running pass: MemProfilerPass on main
// INSTRUMENT: Running pass: ModuleMemProfilerPass on [module]

// TODO: Use text profile inputs once that is available for memprof.
//
// The following commands were used to compile the source to instrumented
// executables and collect raw binary format profiles:
//
// # Collect memory profile:
// $ clang++ -fuse-ld=lld -no-pie -Wl,--no-rosegment -gmlt \
//      -fdebug-info-for-profiling -mno-omit-leaf-frame-pointer \
//      -fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id \
//      memprof.cpp -o memprof.exe -fmemory-profile
// $ env MEMPROF_OPTIONS=log_path=stdout ./memprof.exe > memprof.memprofraw
//
// RUN: llvm-profdata merge %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.memprofdata

// Profile use:
// Ensure Pass PGOInstrumentationUse is invoked with the memprof-only profile.
// RUN: %clang_cc1 -O2 -fprofile-instrument-use-path=%t.memprofdata %s -fdebug-pass-manager  -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=USE
// USE: Running pass: PGOInstrumentationUse on [module]

char *foo() {
  return new char[10];
}
int main() {
  char *a = foo();
  delete[] a;
  return 0;
}
