// REQUIRES: remote-cache-service

// Need a short path for the unix domain socket (and unique for this test file).
// RUN: rm -f %{remote-cache-dir}/%basename_t
// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang_cc1 -cc1 -triple x86_64-apple-macosx12 -analyze -analyzer-checker=deadcode -analyzer-output plist %s -o %t/regular.plist 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG
// RUN: FileCheck %s --input-file=%t/regular.plist --check-prefix=CHECK-PLIST

// CHECK-DIAG: Value stored to 'v' during its initialization is never read
// CHECK-PLIST: Value stored to &apos;v&apos; during its initialization is never read

// RUN: llvm-remote-cache-test -socket-path=%{remote-cache-dir}/%basename_t -cache-path=%t/cache -- env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macosx12 --analyze --analyzer-output plist %s -o %t/cached.plist
// RUN: rm %t/cached.plist
// RUN: llvm-remote-cache-test -socket-path=%{remote-cache-dir}/%basename_t -cache-path=%t/cache -- env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macosx12 --analyze --analyzer-output plist %s -o %t/cached.plist -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-HIT

// CHECK-HIT: remark: compile job cache hit
// CHECK-HIT: Value stored to 'v' during its initialization is never read

// RUN: diff -u %t/regular.plist %t/cached.plist

// Check cache is skipped for analyzer html output.
// RUN: llvm-remote-cache-test -socket-path=%{remote-cache-dir}/%basename_t -cache-path=%t/cache -- env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macosx12 --analyze --analyzer-output html %s -o %t/analysis -Rcompile-job-cache -Wclang-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-HTML
// RUN: llvm-remote-cache-test -socket-path=%{remote-cache-dir}/%basename_t -cache-path=%t/cache -- env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macosx12 --analyze --analyzer-output html %s -o %t/analysis -Rcompile-job-cache -Wclang-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-HTML

// CHECK-HTML: remark: compile job cache miss
// CHECK-HTML: remark: compile job cache skipped
// FIXME: `analyse` action passes `-w` for `-cc1` args and the "caching disabled" warning doesn't show up.

void foo(int *p) {
  int v = p[0];
}
