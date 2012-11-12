// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: grep "hello" %t | count 3
// RUN: grep 'c"hello\\00"' %t | count 2
// RUN: grep 'c"hello\\00\\00\\00"' %t | count 1
// RUN: grep 'c"ola"' %t | count 1

/* Should be 3 hello string, two global (of different sizes), the rest
   are shared. */

void f0() {
  bar("hello");
}

void f1() {
  static char *x = "hello";
  bar(x);
}

void f2() {
  static char x[] = "hello";
  bar(x);
}

void f3() {
  static char x[8] = "hello";
  bar(x);
}

void f4() {
  static struct s {
    char *name;
  } x = { "hello" };
  gaz(&x);
}

char x[3] = "ola";
