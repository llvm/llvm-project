// RUN: %clang_cc1 -emit-llvm -std=c++1z %s -o - -triple=x86_64-linux-gnu | FileCheck %s

struct S { S(); ~S(); };

// CHECK-LABEL: define {{.*}}global_var_init
// CHECK-NOT: br
// CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @global)
S global;

// CHECK-LABEL: define {{.*}}global_var_init
// FIXME: Do we really need thread-safe initialization here? We don't run
// global ctors on multiple threads. (If we were to do so, we'd need thread-safe
// init for B<int>::member and B<int>::inline_member too.)
// CHECK: load atomic i8, i8* bitcast (i64* @_ZGV13inline_global to i8*) acquire,
// CHECK: icmp eq i8 {{.*}}, 0
// CHECK: br i1
// CHECK-NOT: !prof
// CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @inline_global)
inline S inline_global;

// CHECK-LABEL: define {{.*}}global_var_init
// CHECK-NOT: br
// CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @thread_local_global)
thread_local S thread_local_global;

// CHECK-LABEL: define {{.*}}global_var_init
// CHECK: load i8, i8* bitcast (i64* @_ZGV26thread_local_inline_global to i8*)
// CHECK: icmp eq i8 {{.*}}, 0
// CHECK: br i1
// CHECK-NOT: !prof
// CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @thread_local_inline_global)
thread_local inline S thread_local_inline_global;

struct A {
  static S member;
  static thread_local S thread_local_member;

  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: load atomic i8, i8* bitcast (i64* @_ZGVN1A13inline_memberE to i8*) acquire,
  // CHECK: icmp eq i8 {{.*}}, 0
  // CHECK: br i1
  // CHECK-NOT: !prof
  // CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @_ZN1A13inline_memberE)
  static inline S inline_member;

  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: load i8, i8* bitcast (i64* @_ZGVN1A26thread_local_inline_memberE to i8*)
  // CHECK: icmp eq i8 {{.*}}, 0
  // CHECK: br i1
  // CHECK-NOT: !prof
  // CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @_ZN1A26thread_local_inline_memberE)
  static thread_local inline S thread_local_inline_member;
};

// CHECK-LABEL: define void @_Z1fv()
void f() {
  // CHECK: load atomic i8, i8* bitcast (i64* @_ZGVZ1fvE12static_local to i8*) acquire,
  // CHECK: icmp eq i8 {{.*}}, 0
  // CHECK: br i1 {{.*}}, !prof ![[WEIGHTS_LOCAL:[0-9]*]]
  static S static_local;

  // CHECK: load i8, i8* @_ZGVZ1fvE19static_thread_local,
  // CHECK: icmp eq i8 {{.*}}, 0
  // CHECK: br i1 {{.*}}, !prof ![[WEIGHTS_THREAD_LOCAL:[0-9]*]]
  static thread_local S static_thread_local;
}

// CHECK-LABEL: define {{.*}}global_var_init
// CHECK-NOT: br
// CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @_ZN1A6memberE)
S A::member;

// CHECK-LABEL: define {{.*}}global_var_init
// CHECK-NOT: br
// CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @_ZN1A19thread_local_memberE)
thread_local S A::thread_local_member;

template <typename T> struct B {
  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: load i8, i8* bitcast (i64* @_ZGVN1BIiE6memberE to i8*)
  // CHECK: icmp eq i8 {{.*}}, 0
  // CHECK: br i1
  // CHECK-NOT: !prof
  // CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @_ZN1BIiE6memberE)
  static S member;

  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: load i8, i8* bitcast (i64* @_ZGVN1BIiE13inline_memberE to i8*)
  // CHECK: icmp eq i8 {{.*}}, 0
  // CHECK: br i1
  // CHECK-NOT: !prof
  // CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @_ZN1BIiE13inline_memberE)
  static inline S inline_member;

  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: load i8, i8* bitcast (i64* @_ZGVN1BIiE19thread_local_memberE to i8*)
  // CHECK: icmp eq i8 {{.*}}, 0
  // CHECK: br i1
  // CHECK-NOT: !prof
  // CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @_ZN1BIiE19thread_local_memberE)
  static thread_local S thread_local_member;

  // CHECK-LABEL: define {{.*}}global_var_init
  // CHECK: load i8, i8* bitcast (i64* @_ZGVN1BIiE26thread_local_inline_memberE to i8*)
  // CHECK: icmp eq i8 {{.*}}, 0
  // CHECK: br i1
  // CHECK-NOT: !prof
  // CHECK: call void @_ZN1SC1Ev({{.*}}* {{[^,]*}} @_ZN1BIiE26thread_local_inline_memberE)
  static thread_local inline S thread_local_inline_member;
};
template<typename T> S B<T>::member;
template<typename T> thread_local S B<T>::thread_local_member;

template<typename ...T> void use(T &...);
void use_b() {
  use(B<int>::member, B<int>::inline_member, B<int>::thread_local_member,
      B<int>::thread_local_inline_member);
}

// CHECK-LABEL: define {{.*}}tls_init()
// CHECK: load i8, i8* @__tls_guard, align 1
// CHECK: icmp eq i8 {{.*}}, 0
// CHECK: br i1 {{.*}}, !prof ![[WEIGHTS_THREAD_LOCAL]]

// CHECK-DAG: ![[WEIGHTS_THREAD_LOCAL]] = !{!"branch_weights", i32 1, i32 1023}
// CHECK-DAG: ![[WEIGHTS_LOCAL]] = !{!"branch_weights", i32 1, i32 1048575}
