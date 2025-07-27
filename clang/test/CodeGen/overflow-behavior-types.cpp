// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -foverflow-behavior-types \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT


// Test the __attribute__((overflow_behavior())) for C++

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __no_wrap __attribute__((overflow_behavior(no_wrap)))

class Foo {
public:
  unsigned long other;
  char __wrap a;

  Foo() = delete;
  Foo(char _a) : a(_a) {}

  decltype(a) getA() const { return a; }
};
/* define dso_local void @_Z12test_membersc(i8 noundef signext %some) #0 {
entry:
  %some.addr = alloca i8, align 1
  %foo = alloca %class.Foo, align 8
  store i8 %some, ptr %some.addr, align 1
  %0 = load i8, ptr %some.addr, align 1
  call void @_ZN3FooC2Ec(ptr noundef nonnull align 8 dereferenceable(9) %foo, i8 noundef signext %0)
  %a = getelementptr inbounds nuw %class.Foo, ptr %foo, i32 0, i32 1
  %1 = load i8, ptr %a, align 8
  %inc = add i8 %1, 1
  store i8 %inc, ptr %a, align 8
  %call = call noundef i8 @_ZNK3Foo4getAEv(ptr noundef nonnull align 8 dereferenceable(9) %foo)
  %add = add i8 %call, 1
  ret void
}
*/

// DEFAULT-LABEL: define {{.*}} @_Z12test_membersc
void test_members(char some) {
  Foo foo{some};

  // DEFAULT: %[[A:.*]] = getelementptr inbounds nuw %class.Foo, ptr %foo, i32 0, i32 1
  // DEFAULT-NEXT: %[[T1:.*]] = load i8, ptr %[[A]], align 8
  // DEFAULT-NEXT: %inc{{\d*}} = add i8 %[[T1]], 1
  (++foo.a);

  // DEFAULT: %[[CALL:.*]] = call noundef i8 @_ZNK3Foo4getAEv
  // DEFAULT-NEXT: add i8 %[[CALL]], 1
  (void)(foo.getA() + 1);
}

// DEFAULT-LABEL: define {{.*}} @_Z9test_autoU11ObtWrap_c
void test_auto(char __wrap a) {
  auto b = a;

  // DEFAULT: %[[T1:.*]] = load i8, ptr %b
  // DEFAULT-NEXT: sub i8 %[[T1]], 1
  (b - 1); // no instrumentation
}


int overloadme(__no_wrap int a) { return 0; }
int overloadme(int a) { return 1; } // make sure we pick this one
// DEFAULT-LABEL: define {{.*}}test_overload_set_exact_match
int test_overload_set_exact_match(int a) {
  // DEFAULT: call {{.*}} @_Z10overloadmei
  return overloadme(a);
}
