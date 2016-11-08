// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

template <typename TemplArg>
class TemplCls {
// CHECK: [[@LINE-1]]:7 | class(Gen)/C++ | TemplCls | c:@ST>1#T@TemplCls | <no-cgname> | Def | rel: 0
public:
  TemplCls(int x);
  // CHECK: [[@LINE-1]]:3 | constructor/C++ | TemplCls | c:@ST>1#T@TemplCls@F@TemplCls#I# | <no-cgname> | Decl,RelChild | rel: 1
  // CHECK-NEXT: RelChild | TemplCls | c:@ST>1#T@TemplCls
};

TemplCls<int> gtv(0);
// CHECK: [[@LINE-1]]:1 | class(Gen)/C++ | TemplCls | c:@ST>1#T@TemplCls | <no-cgname> | Ref | rel: 0

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

// CHECK: [[@LINE+1]]:23 | type-alias/C | size_t |
typedef unsigned long size_t;
// CHECK: [[@LINE+2]]:7 | function/C | operator new | c:@F@operator new#l# | __Znwm |
// CHECK: [[@LINE+1]]:20 | type-alias/C | size_t | {{.*}} | Ref |
void* operator new(size_t sz);
