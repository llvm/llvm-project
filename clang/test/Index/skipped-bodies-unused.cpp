// RUN: env CINDEXTEST_SKIP_FUNCTION_BODIES=1 c-index-test -test-load-source all %s -Wunused-parameter 2>&1 \
// RUN: | FileCheck %s

// No 'unused parameter' warnings should be shown when skipping the function bodies.
inline int foo(int used, int unused) {
    used = 100;
}
// CHECK-NOT: warning: unused parameter
