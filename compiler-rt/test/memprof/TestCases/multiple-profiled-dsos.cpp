// REQUIRES: x86_64-linux, memprof-dynamic-runtime

// Only the two shared libraries are MemProf-instrumented. -Bsymbolic ensures
// each DSO calls its own externally visible shared_allocate definition.
// RUN: split-file %s %t
// RUN: %clangxx_memprof -O1 -g -fPIC -shared -Wl,--build-id \
// RUN:     -Wl,-Bsymbolic %t/lib1.cpp -o %t.lib1.so
// RUN: %clangxx_memprof -O1 -g -fPIC -shared -Wl,--build-id \
// RUN:     -Wl,-Bsymbolic %t/lib2.cpp -o %t.lib2.so
// RUN: %clangxx -O1 %t/main.cpp %t.lib1.so %t.lib2.so -Wl,-rpath,%T \
// RUN:     -o %t.main
// RUN: rm -f %t.raw.*
// RUN: env MEMPROF_OPTIONS=log_path=%t.raw LD_PRELOAD=%shared_libmemprof \
// RUN:     %run %t.main
// RUN: cp %t.lib1.so %t.duplicate.so
// RUN: not llvm-profdata show --memory %t.raw.* \
// RUN:     --profiled-binary %t.lib1.so \
// RUN:     --profiled-binary %t.duplicate.so -o /dev/null 2>&1 | \
// RUN:     FileCheck %s --check-prefix=DUPLICATE
// RUN: llvm-profdata show --memory %t.raw.* \
// RUN:     --profiled-binary %t.lib1.so --profiled-binary %t.lib2.so \
// RUN:     -o %t.yaml
// RUN: FileCheck %s --check-prefix=RAW < %t.yaml
// RUN: llvm-profdata merge %t.raw.* --profiled-binary %t.lib1.so \
// RUN:     --profiled-binary %t.lib2.so -o %t.memprofdata
// RUN: llvm-profdata show --memory %t.memprofdata | \
// RUN:     FileCheck %s --check-prefix=INDEXED

// The shared_allocate linkage name is identical in both DSOs. Two allocation
// functions proves their module-qualified GUIDs did not collapse into one key.
// RAW:      NumMibInfo: 2
// RAW-NEXT: NumAllocFunctions: 2
// RAW-DAG:  SymbolName: shared_allocate
// RAW-DAG:  TotalSize: 111
// RAW-DAG:  SymbolName: shared_allocate
// RAW-DAG:  TotalSize: 222

// INDEXED: Total contexts: 2
// INDEXED-DAG: TotalSize: 111
// INDEXED-DAG: TotalSize: 222

// DUPLICATE: error: Duplicate profiled binary build id:

//--- main.cpp

extern "C" void lib1_entry();
extern "C" void lib2_entry();

int main() {
  lib1_entry();
  lib2_entry();
}

//--- lib1.cpp

#include <cstdlib>

extern "C" __attribute__((noinline)) void shared_allocate() {
  volatile char *P = static_cast<char *>(std::malloc(111));
  for (int I = 0; I != 111; ++I)
    P[I] = static_cast<char>(I);
  std::free(const_cast<char *>(P));
}

extern "C" __attribute__((noinline)) void lib1_entry() { shared_allocate(); }

//--- lib2.cpp

#include <cstdlib>

extern "C" __attribute__((noinline)) void shared_allocate() {
  volatile char *P = static_cast<char *>(std::malloc(222));
  for (int I = 0; I != 222; ++I)
    P[I] = static_cast<char>(I);
  std::free(const_cast<char *>(P));
}

extern "C" __attribute__((noinline)) void lib2_entry() { shared_allocate(); }
