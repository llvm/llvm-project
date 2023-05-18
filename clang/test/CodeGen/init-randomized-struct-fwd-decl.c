// RUN: %clang_cc1 -triple=x86_64-unknown-linux -emit-llvm -frandomize-layout-seed=1234567890abcdef < %s | FileCheck %s
// PR60349

// Clang will add a forward declaration of "struct bar" and "enum qux" to the
// structures. This shouldn't prevent these structures from being randomized.
// So the 'f' element shouldn't be at the start of the structure anymore.

struct foo {
  struct bar *(*f)(void);
  struct bar *(*g)(void);
  struct bar *(*h)(void);
  struct bar *(*i)(void);
  struct bar *(*j)(void);
  struct bar *(*k)(void);
};

// CHECK-LABEL: define {{.*}}@t1(
// CHECK-NOT: getelementptr inbounds %struct.foo, ptr %3, i32 0, i32 0
struct bar *t1(struct foo *z) {
  return z->f();
}

struct baz {
  enum qux *(*f)(void);
  enum qux *(*g)(void);
  enum qux *(*h)(void);
  enum qux *(*i)(void);
  enum qux *(*j)(void);
  enum qux *(*k)(void);
};

// CHECK-LABEL: define {{.*}}@t2(
// CHECK-NOT: getelementptr inbounds %struct.baz, ptr %3, i32 0, i32 0
enum qux *t2(struct baz *z) {
  return z->f();
}
