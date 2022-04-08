// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu -emit-llvm --std=c++17 %s -o - | FileCheck %s

struct stream {
  friend const stream &operator<<(const stream &, const float &);
};

void foo() {
  constexpr float f_zero = 0.0f;
  constexpr float f_one = 1.0f;
  constexpr float f_two = 2.0f;

  stream s;
  [=]() {
    s << f_zero << f_one << f_two;
  }();
}

// CHECK: define{{.*}} void @_Z3foov
// CHECK: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 0
// CHECK-NEXT: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 1
// CHECK-NEXT: store float 0.000
// CHECK-NEXT: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 2
// CHECK-NEXT: store float 1.000
// CHECK-NEXT: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 3
// CHECK-NEXT: store float 2.000

// The lambda body.  Reverse iteration when the captures aren't deterministic
// causes these to be laid out differently in the lambda.
// CHECK: define internal void
// CHECK: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 0
// CHECK: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 1
// CHECK: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 2
// CHECK: getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i32 0, i32 3
