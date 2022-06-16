// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest --data %s > %t/casid

// RUN: %clang -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas \
// RUN:   -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Wimplicit-function-declaration \
// RUN:   -Wno-error=implicit-function-declaration \
// RUN:   -Rcompile-job-cache-hit -emit-obj -o %t/output.o \
// RUN:   -serialize-diagnostic-file %t/diags %s 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS

// RUN: c-index-test -read-diagnostics %t/diags 2>&1 | FileCheck %s --check-prefix=SERIALIZED-MISS

// RUN: ls %t/output.o && rm %t/output.o

// RUN: %clang -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas \
// RUN:   -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Wimplicit-function-declaration \
// RUN:   -Wno-error=implicit-function-declaration \
// RUN:   -Rcompile-job-cache-hit -emit-obj -o %t/output.o \
// RUN:   -serialize-diagnostic-file %t/diags %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT

// RUN: c-index-test -read-diagnostics %t/diags 2>&1 | FileCheck %s --check-prefix=SERIALIZED-HIT

// CACHE-HIT: remark: compile job cache hit
// CACHE-HIT: warning: call to undeclared function

// CACHE-MISS: warning: call to undeclared function
// CACHE-MISS-NOT: remark: compile job cache hit

// FIXME: serialized diagnostics should match the text diagnostics rdar://85234207
// SERIALIZED-HIT: warning: compile job cache hit for
// SERIALIZED-HIT: Number of diagnostics: 1
// SERIALIZED-MISS: Number of diagnostics: 0

void foo(void) {
  bar();
}
