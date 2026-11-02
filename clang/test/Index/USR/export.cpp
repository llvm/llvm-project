// RUN: c-index-test core -print-source-symbols -- -std=c++20 -xc++-module %s | FileCheck %s

// Export declarations are skipped in USRs for enclosed items.
export module m;

namespace ns {
export void f();
// CHECK: [[@LINE-1]]:13 | function/C | f | c:@N@ns@F@f# |
export {
int X;
// CHECK: [[@LINE-1]]:5 | variable/C | X | c:@N@ns@X |
}
export namespace inner {
// CHECK: [[@LINE-1]]:18 | namespace/C++ | inner | c:@N@ns@N@inner |
void g();
// CHECK: [[@LINE-1]]:6 | function/C | g | c:@N@ns@N@inner@F@g# |
}
}
