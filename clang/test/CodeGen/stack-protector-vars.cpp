// RUN: %clang -target x86_64-apple-darwin -emit-llvm -S -o - %s | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -emit-llvm -S -o - %s -fstack-protector | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -emit-llvm -S -o - %s -fstack-protector-all | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin -emit-llvm -Xclang -verify -fstack-protector-all %s -o %t -c

typedef __SIZE_TYPE__ size_t;

int printf(const char * _Format, ...);
char *strcpy(char *s1, const char *s2);

struct S {
  S();
  int a[4];
};

// CHECK: define {{.*}} @_Z5test1PKc
// CHECK: %{{.*}} = alloca [1000 x i8], align {{.*}}, !stack-protector ![[A:.*]]
void test1(const char *msg) {
  __attribute__((stack_protector_ignore))
  char a[1000]; // expected-warning {{'stack_protector_ignore' attribute ignored due to '-fstack-protector-all' option}}
  strcpy(a, msg);
  printf("%s\n", a);
}

// CHECK: define {{.*}} @_Z5test2
// CHECK-NOT: %{{.*}} = alloca [1000 x i8], align {{.*}}, !stack-protector
void test2(const char *msg) {
  char b[1000];
  strcpy(b, msg);
  printf("%s\n", b);
}

// CHECK: define {{.*}} @_Z5test3v
// CHECK: %{{.*}} = alloca %struct.S, align {{.*}}, !stack-protector ![[A:.*]]
S test3() {
  __attribute__((stack_protector_ignore))
  S s; // expected-warning {{'stack_protector_ignore' attribute ignored due to '-fstack-protector-all' option}}
  return s;
}

// CHECK: define {{.*}} @_Z5test4b
// CHECK: %{{.*}} = alloca %struct.S, align {{.*}}, !stack-protector ![[A:.*]]
// CHECK: call void @_ZN1SC1Ev
S test4(bool b) {
  __attribute__((stack_protector_ignore))
  S s; // expected-warning {{'stack_protector_ignore' attribute ignored due to '-fstack-protector-all' option}}
  if ( b )
    return s;
  else
    return s;
}

// CHECK: define {{.*}} @_Z5test5b
// CHECK: %{{.*}} = alloca %struct.S, align {{.*}}
// CHECK-NOT: stack-protector
// CHECK: %{{.*}} = alloca %struct.S, align {{.*}}, !stack-protector ![[A:.*]]
// CHECK: %{{.*}} = alloca %struct.S, align {{.*}}
// CHECK-NOT: stack-protector
// CHECK: call void @_ZN1SC1Ev
// CHECK: call void @_ZN1SC1Ev
S test5(bool b) {
  __attribute__((stack_protector_ignore))
  S s1; // expected-warning {{'stack_protector_ignore' attribute ignored due to '-fstack-protector-all' option}}
  S s2;
  if ( b )
    return s1;
  else
    return s2;
}

// CHECK: ![[A]] = !{i32 0}
