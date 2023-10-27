// XFAIL: target={{.*}}-aix{{.*}}
// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-unknown-unknown 2>&1 | FileCheck %s

namespace ns {
//  void foo();
}

using ns::foo __attribute__((using_if_exists));
// CHECK: [[@LINE-1]]:11 | using/C++ | foo | c:@UD@foo | <no-cgname> | Decl | rel: 0
// CHECK-NOT: <unknown>
