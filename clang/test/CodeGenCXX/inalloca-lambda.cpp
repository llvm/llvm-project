// RUN: %clang_cc1 -triple i686-windows-msvc -emit-llvm -o - %s  2>&1 | FileCheck %s

struct A {
  A();
  A(const A &);
  int x;
};
void decayToFp(int (*f)(A));
void test() {
  auto ld = [](A a) {
    static int calls = 0;
    ++calls;
    return a.x + calls;
  };
  decayToFp(ld);
  ld(A{});
}

// CHECK: define internal x86_thiscallcc noundef i32
// CHECK-SAME: @"??R<lambda_0>@?0??test@@YAXXZ@QBE?A?<auto>@@UA@@@Z"
// CHECK-SAME: (ptr noundef %this, ptr inalloca(<{ %struct.A }>) %[[ARG:.*]])
// CHECK: %[[V:.*]] = getelementptr inbounds <{ %struct.A }>, ptr %[[ARG]], i32 0, i32 0
// CHECK: %call = call x86_thiscallcc noundef i32
// CHECK-SAME: @"?__impl@<lambda_0>@?0??test@@YAXXZ@QBE?A?<auto>@@UA@@@Z"
// CHECK-SAME: (ptr noundef %this, ptr noundef %[[V]])

// CHECK: define internal noundef i32
// CHECK-SAME: @"?__invoke@<lambda_0>@?0??test@@YAXXZ@CA?A?<auto>@@UA@@@Z"
// CHECK-SAME: (ptr inalloca(<{ %struct.A }>) %[[ARG:.*]])
// CHECK: %unused.capture = alloca %class.anon, align 1
// CHECK: %[[VAR:.*]] = getelementptr inbounds <{ %struct.A }>, ptr %[[ARG]], i32 0, i32 0
// CHECK: %call = call x86_thiscallcc noundef i32
// CHECK-SAME: @"?__impl@<lambda_0>@?0??test@@YAXXZ@QBE?A?<auto>@@UA@@@Z"
// CHECK-SAME: (ptr noundef %unused.capture, ptr noundef %[[VAR]])
// CHECK: ret i32 %call

// CHECK: define internal x86_thiscallcc noundef i32
// CHECK-SAME: @"?__impl@<lambda_0>@?0??test@@YAXXZ@QBE?A?<auto>@@UA@@@Z"
// CHECK-SAME: (ptr noundef %this, ptr noundef %[[ARG:.*]])
// CHECK: %this.addr = alloca ptr, align 4
// CHECK: store ptr %this, ptr %this.addr, align 4
// CHECK: %this1 = load ptr, ptr %this.addr, align 4
// CHECK: %{{.*}} = load i32, ptr @"?calls@?1???R<lambda_0>
// CHECK: %inc = add nsw i32 %{{.*}}, 1
// CHECK: store i32 %inc, ptr @"?calls@?1???R<lambda_0>
// CHECK: %{{.*}} = getelementptr inbounds %struct.A, ptr %{{.*}}, i32 0, i32 0
// CHECK: %{{.*}} = load i32, ptr %{{.*}}, align 4
// CHECK: %{{.*}} = load i32, ptr @"?calls@?1???R<lambda_0>
// CHECK: %add = add nsw i32 %{{.*}}, %{{.*}}
// CHECK: ret i32 %add

// Make sure we don't try to copy an uncopyable type.
struct B {
  B();
  B(B &);
  void operator=(B);
  long long x;
} b;

void f() {
  [](B) {}(b);
}

