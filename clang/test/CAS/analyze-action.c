// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang_cc1 -cc1 -triple x86_64-apple-macosx12 -analyze -analyzer-checker=deadcode -analyzer-output plist %s -o %t/regular.plist 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG
// RUN: FileCheck %s --input-file=%t/regular.plist --check-prefix=CHECK-PLIST

// CHECK-DIAG: Value stored to 'v' during its initialization is never read
// CHECK-PLIST: Value stored to &apos;v&apos; during its initialization is never read

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macosx12 -fcas-path %t/cas -analyze -analyzer-checker=deadcode -analyzer-output plist %s -o %t/cached.plist
// RUN: %clang @%t/t.rsp

// RUN: rm %t/cached.plist
// RUN: %clang @%t/t.rsp -Rcompile-job-cache 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HIT

// CHECK-HIT: remark: compile job cache hit
// CHECK-HIT: Value stored to 'v' during its initialization is never read

// RUN: diff -u %t/regular.plist %t/cached.plist

// Check cache is skipped for analyzer html output.
// RUN: %clang -cc1depscan -fdepscan=inline -o %t/t2.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macosx12 -fcas-path %t/cas -analyze -analyzer-checker=deadcode -analyzer-output html %s -o %t/analysis
// RUN: %clang @%t/t2.rsp -Rcompile-job-cache 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HTML
// RUN: %clang @%t/t2.rsp -Rcompile-job-cache 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HTML

// CHECK-HTML: remark: compile job cache miss
// CHECK-HTML: warning: caching disabled because analyzer output is not supported
// CHECK-HTML: remark: compile job cache skipped

void foo(int *p) {
  int v = p[0];
}
