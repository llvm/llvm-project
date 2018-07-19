// RUN: not %clang -cc1 -fsyntax-only %s 2>&1 | FileCheck %s

#define a b
#define b c
#define c(x) d(x)
#define d(x) x*1

#define e f
#define f g
#define g(x) h(x)
#define h(x) x

void PR16799() {
  const char str[] = "string";
  a(str);
  // CHECK: :15:3: error: invalid operands to binary expression
  // CHECK:       ('const char *' and 'int')
  // CHECK:   a(str);
  // CHECK:   ^~~~~~
  // CHECK: :3:11: note: expanded from macro 'a'
  // CHECK: #define a b
  // CHECK:           ^
  // CHECK: :4:11: note: expanded from macro 'b'
  // CHECK: #define b c
  // CHECK:           ^
  // CHECK: :5:14: note: expanded from macro 'c'
  // CHECK: #define c(x) d(x)
  // CHECK:              ^~~~
  // CHECK: :6:15: note: expanded from macro 'd'
  // CHECK: #define d(x) x*1
  // CHECK:              ~^~

  e(str);
  // CHECK: :33:5: warning: expression result unused
  // CHECK:   e(str);
  // CHECK:     ^~~
  // CHECK: :8:11: note: expanded from macro 'e'
  // CHECK: #define e f
  // CHECK:           ^
  // CHECK: :9:11: note: expanded from macro 'f'
  // CHECK: #define f g
  // CHECK:           ^
  // CHECK: :10:16: note: expanded from macro 'g'
  // CHECK: #define g(x) h(x)
  // CHECK:                ^
  // CHECK: :11:14: note: expanded from macro 'h'
  // CHECK: #define h(x) x
  // CHECK:              ^
}
// CHECK: 1 warning and 1 error generated.
