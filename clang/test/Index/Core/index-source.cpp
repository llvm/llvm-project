// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

template <typename TemplArg>
class TemplCls {
// CHECK: [[@LINE-1]]:7 | class/C++ | TemplCls | c:@ST>1#T@TemplCls | <no-cgname> | Def | rel: 0
  TemplCls(int x);
  // CHECK: [[@LINE-1]]:3 | constructor/C++ | TemplCls | c:@ST>1#T@TemplCls@F@TemplCls#I# | <no-cgname> | Decl,RelChild | rel: 1
  // CHECK-NEXT: RelChild | TemplCls | c:@ST>1#T@TemplCls
};

template <typename T>
class BT {
  struct KLR {
    int idx;
  };

  // CHECK: [[@LINE+1]]:7 | instance-method/C++ | foo |
  KLR foo() {
    return { .idx = 0 }; // Make sure this doesn't trigger a crash.
  }
};
