// Test if memprof instrumentation and use pass are invoked.
// RUN: rm -rf %t && split-file %s %t

// Instrumentation:
// Ensure Pass MemProfilerPass and ModuleMemProfilerPass are invoked.
// RUN: %clang_cc1 -O2 -fmemory-profile %t/a.cpp -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=INSTRUMENT
// INSTRUMENT: Running pass: MemProfilerPass on main
// INSTRUMENT: Running pass: ModuleMemProfilerPass on [module]

// Profile use:
// Ensure Pass PGOInstrumentationUse is invoked with the memprof-only profile.
// RUN: llvm-profdata merge %t/a.yaml -o %t/a.memprofdata
// RUN: %clang_cc1 -O2 -fmemory-profile-use=%t/a.memprofdata %t/a.cpp -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=USE
// USE: Running pass: MemProfUsePass on [module]

//--- a.cpp
char *foo() {
  return new char[10];
}
int main() {
  char *a = foo();
  delete[] a;
  return 0;
}

//--- a.yaml
---
HeapProfileRecords:
  - GUID:            main
    AllocSites:
      - Callstack:
          - { Function: main, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z3foov, LineOffset: 1, Column: 13, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 0
  - GUID:            _Z3foov
    CallSites:
      - Frames:
          - { Function: _Z3foov, LineOffset: 1, Column: 13, IsInlineFrame: false }
...
