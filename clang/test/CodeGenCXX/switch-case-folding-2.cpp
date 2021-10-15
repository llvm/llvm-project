// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

extern int printf(const char*, ...);

// CHECK-LABEL: @_Z4testi(
int test(int val){
 switch (val) {
 case 4:
   do {
     switch (6) {
       // CHECK: call noundef i32 (i8*, ...) @_Z6printfPKcz
       case 6: do { case 5: printf("bad\n"); } while (0);
     };
   } while (0);
 }
 return 0;
}

int main(void) {
 return test(5);
}

// CHECK-LABEL: @_Z10other_testv(
void other_test() {
  switch(0) {
  case 0:
    do {
    default:;
    } while(0);
  }
}

struct X { X(); ~X(); };

void dont_call();
void foo();

// CHECK-LABEL: @_Z13nested_scopesv(
void nested_scopes() {
  switch (1) {
  case 0:
    // CHECK-NOT: @_Z9dont_callv(
    dont_call();
    break;

  default:
    // CHECK: call {{.*}} @_ZN1XC1Ev(
    // CHECK: call {{.*}} @_Z3foov(
    // CHECK-NOT: call {{.*}} @_Z3foov(
    // CHECK: call {{.*}} @_ZN1XD1Ev(
    { X x; foo(); }

    // CHECK: call {{.*}} @_ZN1XC1Ev(
    // CHECK: call {{.*}} @_Z3foov(
    // CHECK: call {{.*}} @_ZN1XD1Ev(
    { X x; foo(); }

    // CHECK: call {{.*}} @_ZN1XC1Ev(
    // CHECK: call {{.*}} @_Z3foov(
    // CHECK: call {{.*}} @_ZN1XD1Ev(
    { X x; foo(); }
    break;
  }
}

// CHECK-LABEL: @_Z17scope_fallthroughv(
void scope_fallthrough() {
  switch (1) {
    // CHECK: call {{.*}} @_ZN1XC1Ev(
    // CHECK-NOT: call {{.*}} @_Z3foov(
    // CHECK: call {{.*}} @_ZN1XD1Ev(
    { default: X x; }
    // CHECK: call {{.*}} @_Z3foov(
    foo();
    break;
  }
}

// CHECK-LABEL: @_Z12hidden_breakb(
void hidden_break(bool b) {
  switch (1) {
  default:
    // CHECK: br
    if (b)
      break;
    // CHECK: call {{.*}} @_Z3foov(
    foo();
    break;
  }
}

// CHECK-LABEL: @_Z10hidden_varv(
int hidden_var() {
  switch (1) {
  // CHECK: %[[N:.*]] = alloca i32
  case 0: int n;
  // CHECK: store i32 0, i32* %[[N]]
  // CHECK: load i32, i32* %[[N]]
  // CHECK: ret
  default: n = 0; return n;
  }
}

// CHECK-LABEL: @_Z13case_in_labelv(
void case_in_label() {
  // CHECK: br label %
  switch (1) case 1: foo: case 0: goto foo;
}
