// RUN: %clang_cc1 -triple thumbv7-apple-ios %s -emit-llvm -o %t -fblocks
// RUN: grep "_Block_object_dispose" %t | count 12
// RUN: grep "__copy_helper_block_" %t | count 9
// RUN: grep "__destroy_helper_block_" %t | count 9
// RUN: grep "__Block_byref_object_copy_" %t | count 2
// RUN: grep "__Block_byref_object_dispose_" %t | count 2
// RUN: grep "i32 135)" %t | count 2
// RUN: grep "_Block_object_assign" %t | count 5

// RUN: %clang_cc1 -triple thumbv7-unknown-windows %s -emit-llvm -o %t -fblocks
// RUN: grep "_Block_object_dispose" %t | count 12
// RUN: grep "__copy_helper_block_" %t | count 11
// RUN: grep "__destroy_helper_block_" %t | count 11
// RUN: grep "__Block_byref_object_copy_" %t | count 2
// RUN: grep "__Block_byref_object_dispose_" %t | count 2
// RUN: grep "i32 135)" %t | count 2
// RUN: grep "_Block_object_assign" %t | count 5

int printf(const char *, ...);

void test1() {
  __block int a;
  int b=2;
  a=1;
  printf("a is %d, b is %d\n", a, b);
  ^{ a = 10; printf("a is %d, b is %d\n", a, b); }(); // needs copy/dispose
  printf("a is %d, b is %d\n", a, b);
  a = 1;
  printf("a is %d, b is %d\n", a, b);
}

void test2() {
  __block int a;
  a=1;
  printf("a is %d\n", a);
  ^{ // needs copy/dispose
    ^{ // needs copy/dispose
      a = 10;
    }();
  }();
  printf("a is %d\n", a);
  a = 1;
  printf("a is %d\n", a);
}

void test3() {
  __block int k;
  __block int (^j)(int);
  ^{j=0; k=0;}(); // needs copy/dispose
}

int test4() {
  extern int g;
  static int i = 1;
  ^(int j){ i = j; g = 0; }(0); // does not need copy/dispose
  return i + g;
}

int g;

void test5() {
  __block struct { int i; } i;
  ^{ (void)i; }(); // needs copy/dispose
}

void test6() {
  __block int i;
  ^{ i=1; }(); // needs copy/dispose
  ^{}(); // does not need copy/dispose
}

void test7() {
  ^{ // does not need copy/dispose
    __block int i;
    ^{ i = 1; }(); // needs copy/dispose
  }();
}

int main() {
  int rv = 0;
  test1();
  test2();
  test3();
  rv += test4();
  test5();
  return rv;
}
