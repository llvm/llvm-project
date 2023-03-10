// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct AStruct { 
  int i;
  char *s;
  double d;
};

void f(void) {
  static int i = 42;
  static int is[] = { 1, 2, 3, 4 };
  static char* str = "forty-two";
  static char* strs[] = { "one", "two", "three", "four" };
  static struct AStruct myStruct = { 1, "two", 3.0 };
}

// CHECK: @g.b = internal global ptr @g.a
void g(void) {
  static char a[10];
  static char *b = a;
}

struct s { void *p; };

void foo(void) {
  static struct s var = {((void*)&((char*)0)[0])};
}

// CHECK: @f1.l0 = internal global i32 ptrtoint (ptr @f1 to i32)
int f1(void) { static int l0 = (unsigned) f1; }

// PR7044
char *f2(char key) {
  switch (key) {
    static char _msg[40];
  case '\014':
    return _msg;
  }

  return 0;
}
