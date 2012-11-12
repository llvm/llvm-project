// RUN: %clang_cc1 -fsyntax-only -Wc++11-compat -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// Verify that the appropriate fixits are emitted for narrowing conversions in
// initializer lists.

typedef short int16_t;

void fixits() {
  int x = 999;
  struct {char c;} c2 = {x};
  // CHECK: warning:{{.*}} cannot be narrowed
  // CHECK: fix-it:{{.*}}:26}:"static_cast<char>("
  // CHECK: fix-it:{{.*}}:27}:")"
  struct {int16_t i;} i16 = {70000};
  // CHECK: warning:{{.*}} cannot be narrowed
  // CHECK: fix-it:{{.*}}:30}:"static_cast<int16_t>("
  // CHECK: fix-it:{{.*}}:35}:")"
}

template<typename T>
void maybe_shrink_int(T t) {
  struct {T t;} t2 = {700};
}

void test_template() {
  maybe_shrink_int((char)3);
  // CHECK: warning:{{.*}} cannot be narrowed
  // CHECK: note:{{.*}} in instantiation
  // CHECK: note:{{.*}} override
  // FIXME: This should be static_cast<T>.
  // CHECK: fix-it:{{.*}}"static_cast<char>("
  // CHECK: fix-it:{{.*}}")"
}
