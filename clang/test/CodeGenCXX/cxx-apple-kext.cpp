// RUN: %clangxx -target x86_64-apple-darwin10 %s -flto -S -o - |\
// RUN:   FileCheck --check-prefix=CHECK-NO-KEXT %s
// RUN: %clangxx -target x86_64-apple-darwin10 %s -fapple-kext -flto -S -o - |\
// RUN:   FileCheck --check-prefix=CHECK-KEXT %s

// CHECK-NO-KEXT-NOT: _GLOBAL__D_a
// CHECK-NO-KEXT: @is_hosted = global
// CHECK-NO-KEXT: @_ZTI3foo = {{.*}} @_ZTVN10__cxxabiv117
// CHECK-NO-KEXT: call i32 @__cxa_atexit({{.*}} @_ZN3fooD1Ev
// CHECK-NO-KEXT: declare i32 @__cxa_atexit

// CHECK-KEXT: @_ZTV3foo = 
// CHECK-KEXT-NOT: @_ZTVN10__cxxabiv117
// CHECK-KEXT-NOT: call i32 @__cxa_atexit({{.*}} @_ZN3fooD1Ev
// CHECK-KEXT-NOT: declare i32 @__cxa_atexit
// CHECK-KEXT: @is_freestanding = global
// CHECK-KEXT: _GLOBAL__D_a
// CHECK-KEXT: call void @_ZN3fooD1Ev(%class.foo* @a)

class foo {
public:
  foo();
  virtual ~foo();
};

foo a;
foo::~foo() {}

#if !(__STDC_HOSTED__ == 1)
int is_freestanding = 1;
#else
int is_hosted = 1;
#endif

extern "C" void f1() {
}
