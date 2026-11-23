// RUN: clang-reorder-fields --extra-arg="-std=c++20" -record-name Bar -fields-order c,a,b %s -- | FileCheck %s

class Foo {
public:
  const int* x;
  int y;        
};

class Bar {
public:
  char a; // CHECK:      {{^  int c;}}
  Foo b;  // CHECK-NEXT: {{^  char a;}}
  int c;  // CHECK-NEXT: {{^  Foo b;}}
};

int main() {
  const int x = 13;
  Bar bar1 = { 'a', { &x, 0 }, 123 };              // CHECK: {{^ Bar bar1 = { 123, 'a', { &x, 0 } };}}
  Bar bar2 = { .a = 'a', { &x, 0 }, 123 };         // CHECK: {{^ Bar bar2 = { .c = 123, .a = 'a', .b = { &x, 0 } };}}
  Bar bar3 = { 'a', .b { &x, 0 }, 123 };           // CHECK: {{^ Bar bar3 = { .c = 123, .a = 'a', .b = { &x, 0 } };}}
  Bar bar4 = { .c = 123, .b { &x, 0 }, .a = 'a' }; // CHECK: {{^ Bar bar4 = { .c = 123, .a = 'a', .b = { &x, 0 } };}}

  return 0;
}
