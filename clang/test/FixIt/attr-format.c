// RUN: not %clang_cc1 %s -fsyntax-only -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

// CHECK: fix-it:"{{.*}}attr-format.c":{4:36-4:37}:"0"
__attribute__((format(strftime, 1, 1)))
void my_strftime(const char *fmt);

// CHECK: fix-it:"{{.*}}attr-format.c":{8:34-8:36}:"2"
__attribute__((format(printf, 1, 10)))
void my_strftime(const char *fmt, ...);
