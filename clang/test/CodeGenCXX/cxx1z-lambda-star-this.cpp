// RUN: %clang_cc1 -std=c++1y -triple i686-pc-windows-msvc -emit-llvm %s -o - | FileCheck %s
//CHECK: %[[A_LAMBDA:.*]] = type { %struct.A }
//CHECK: %[[B_LAMBDA:.*]] = type { ptr }
struct A {
  double a = 111;
  auto foo() { return [*this] { return a; }; }
};

namespace ns1 {
int X = A{}.foo()();
} //end ns1

//CHECK: @"?foo@A@@QAE?A?<auto>@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%class.anon) align 8 %[[A_LAMBDA_RETVAL:.*]])
// get the first object with the closure type, which is of type 'struct.A'
//CHECK: %[[I0:.+]] = getelementptr inbounds %[[A_LAMBDA]], ptr %[[A_LAMBDA_RETVAL]], i32 0, i32 0
// copy the contents ...
//CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 8 %[[I0]], ptr align 8 %this1, i32 8, i1 false)

struct B {
  double b = 222;
  auto bar() { return [this] { return b; }; };
};

namespace ns2 {
int X = B{}.bar()();
}
//CHECK: @"?bar@B@@QAE?A?<auto>@@XZ"(ptr {{[^,]*}} %this, ptr noalias sret(%class.anon.0) align 4 %agg.result)
//CHECK: %[[I20:.+]] = getelementptr inbounds %class.anon.0, ptr %agg.result, i32 0, i32 0
//CHECK: store ptr %this1, ptr %[[I20]], align 4
