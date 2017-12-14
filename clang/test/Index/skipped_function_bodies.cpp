// RUN: env CINDEXTEST_SKIP_FUNCTION_BODIES=1 c-index-test -test-load-source all %s 2>&1 \
// RUN: | FileCheck %s

inline int with_body() { return 10; }
inline int without_body();

int x = with_body() + without_body();
// CHECK: warning: inline function 'without_body' is not defined
// CHECK-NOT: warning: inline function 'with_body' is not defined
