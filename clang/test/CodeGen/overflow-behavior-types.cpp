// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -fexperimental-overflow-behavior-types \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __trap __attribute__((overflow_behavior(trap)))

class Foo {
public:
  unsigned long other;
  char __ob_wrap a;

  Foo() = delete;
  Foo(char _a) : a(_a) {}

  decltype(a) getA() const { return a; }
};

// DEFAULT-LABEL: define {{.*}} @_Z12test_membersc
void test_members(char some) {
  Foo foo{some};

  // DEFAULT: %[[A:.*]] = getelementptr inbounds nuw %class.Foo, ptr %foo, i32 0, i32 1
  // DEFAULT-NEXT: %[[T1:.*]] = load i8, ptr %[[A]]
  // DEFAULT-NEXT: %[[CONV:.*]] = sext i8 %[[T1]] to i32
  // DEFAULT-NEXT: %inc{{\d*}} = add i32 %[[CONV]], 1
  (++foo.a);

  // DEFAULT: %[[CALL:.*]] = call noundef signext i8 @_ZNK3Foo4getAEv
  // DEFAULT-NEXT: sext i8 %[[CALL]] to i32
  // DEFAULT-NEXT: add i32 {{.*}}, 1
  (void)(foo.getA() + 1);
}

// DEFAULT-LABEL: define {{.*}} @_Z9test_autoU8ObtWrap_c
void test_auto(char __ob_wrap a) {
  auto b = a;

  // DEFAULT: %[[T1:.*]] = load i8, ptr %b
  // DEFAULT: sub i32 {{.*}}, 1
  (b - 1); // no instrumentation
}


int overloadme(__ob_trap int a) { return 0; }
int overloadme(int a) { return 1; } // make sure we pick this one
// DEFAULT-LABEL: define {{.*}}test_overload_set_exact_match
int test_overload_set_exact_match(int a) {
  // DEFAULT: call {{.*}} @_Z10overloadmei
  return overloadme(a);
}
