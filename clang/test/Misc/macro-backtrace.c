// Tests for macro expansion backtraces. The RUN and CHECK lines are grouped
// below the test code to reduce noise when updating them.

#define M1(A, B) ((A) < (B))
#define M2(A, B) M1(A, B)
#define M3(A, B) M2(A, B)
#define M4(A, B) M3(A, B)
#define M5(A, B) M4(A, B)
#define M6(A, B) M5(A, B)
#define M7(A, B) M6(A, B)
#define M8(A, B) M7(A, B)
#define M9(A, B) M8(A, B)
#define M10(A, B) M9(A, B)
#define M11(A, B) M10(A, B)
#define M12(A, B) M11(A, B)

void f(int *ip, float *fp) {
  if (M12(ip, fp)) { }
  // RUN: %clang_cc1 -fsyntax-only -fmacro-backtrace-limit=5 %s 2>&1 \
  // RUN:   | FileCheck %s -check-prefix=CHECK-LIMIT
  // CHECK-LIMIT: macro-backtrace.c:18:7: warning: comparison of distinct pointer types ('int *' and 'float *')
  // CHECK-LIMIT: if (M12(ip, fp)) { }
  // CHECK-LIMIT: macro-backtrace.c:15:19: note: expanded from macro 'M12'
  // CHECK-LIMIT: #define M12(A, B) M11(A, B)
  // CHECK-LIMIT: macro-backtrace.c:14:19: note: expanded from macro 'M11'
  // CHECK-LIMIT: #define M11(A, B) M10(A, B)
  // CHECK-LIMIT: note: (skipping 7 expansions in backtrace; use -fmacro-backtrace-limit=0 to see all)
  // CHECK-LIMIT: macro-backtrace.c:6:18: note: expanded from macro 'M3'
  // CHECK-LIMIT: #define M3(A, B) M2(A, B)
  // CHECK-LIMIT: macro-backtrace.c:5:18: note: expanded from macro 'M2'
  // CHECK-LIMIT: #define M2(A, B) M1(A, B)
  // CHECK-LIMIT: macro-backtrace.c:4:23: note: expanded from macro 'M1'
  // CHECK-LIMIT: #define M1(A, B) ((A) < (B))

  // RUN: %clang_cc1 -fsyntax-only -fno-caret-diagnostics %s 2>&1 \
  // RUN:   | FileCheck %s -check-prefix=CHECK-NO-CARETS
  // CHECK-NO-CARETS: macro-backtrace.c:18:7: warning: comparison of distinct pointer types ('int *' and 'float *')
  // CHECK-NO-CARETS-NEXT: macro-backtrace.c:15:19: note: expanded from macro 'M12'
  // CHECK-NO-CARETS-NEXT: macro-backtrace.c:14:19: note: expanded from macro 'M11'
  // CHECK-NO-CARETS-NEXT: macro-backtrace.c:13:19: note: expanded from macro 'M10'
  // CHECK-NO-CARETS-NEXT: note: (skipping 6 expansions in backtrace; use -fmacro-backtrace-limit=0 to see all)
  // CHECK-NO-CARETS-NEXT: macro-backtrace.c:6:18: note: expanded from macro 'M3'
  // CHECK-NO-CARETS-NEXT: macro-backtrace.c:5:18: note: expanded from macro 'M2'
  // CHECK-NO-CARETS-NEXT: macro-backtrace.c:4:23: note: expanded from macro 'M1'

  // Check that the expansion notes respect the same formatting options as
  // other diagnostics.
  // RUN: %clang_cc1 -fsyntax-only -fdiagnostics-format vi %s 2>&1 \
  // RUN:   | FileCheck %s -check-prefix=CHECK-NOTE-FORMAT
  // CHECK-NOTE-FORMAT: macro-backtrace.c +18:7: warning:
  // CHECK-NOTE-FORMAT: macro-backtrace.c +15:19: note:
  // CHECK-NOTE-FORMAT: macro-backtrace.c +14:19: note:
  // CHECK-NOTE-FORMAT: note:
  // CHECK-NOTE-FORMAT: macro-backtrace.c +6:18: note:
  // CHECK-NOTE-FORMAT: macro-backtrace.c +5:18: note:
  // CHECK-NOTE-FORMAT: macro-backtrace.c +4:23: note:
}
