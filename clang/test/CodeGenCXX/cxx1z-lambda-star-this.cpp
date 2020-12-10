// RUN: %clang_cc1 -std=c++1y -triple i686-pc-windows-msvc -emit-llvm %s -o - | FileCheck %s
//CHECK: %[[A_LAMBDA:.*]] = type { %struct.A }
//CHECK: %[[B_LAMBDA:.*]] = type { %struct.B* }
struct A {
  double a = 111;
  auto foo() { return [*this] { return a; }; }
};

namespace ns1 {
int X = A{}.foo()();
} //end ns1

//CHECK: @"?foo@A@@QAE?A?<auto>@@XZ"(%struct.A* {{[^,]*}} %this, %class.anon* noalias sret(%class.anon) align 8 %[[A_LAMBDA_RETVAL:.*]])
// get the first object with the closure type, which is of type 'struct.A'
//CHECK: %[[I0:.+]] = getelementptr inbounds %[[A_LAMBDA]], %[[A_LAMBDA]]* %[[A_LAMBDA_RETVAL]], i32 0, i32 0
//CHECK: %[[I1:.+]] = bitcast %struct.A* %[[I0]] to i8*
//CHECK: %[[I2:.+]] = bitcast %struct.A* %this1 to i8*
// copy the contents ...
//CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %[[I1]], i8* align 8 %[[I2]], i32 8, i1 false)

struct B {
  double b = 222;
  auto bar() { return [this] { return b; }; };
};

namespace ns2 {
int X = B{}.bar()();
}
//CHECK: @"?bar@B@@QAE?A?<auto>@@XZ"(%struct.B* {{[^,]*}} %this, %class.anon.0* noalias sret(%class.anon.0) align 4 %agg.result)
//CHECK: %[[I20:.+]] = getelementptr inbounds %class.anon.0, %class.anon.0* %agg.result, i32 0, i32 0
//CHECK: store %struct.B* %this1, %struct.B** %[[I20]], align 4
