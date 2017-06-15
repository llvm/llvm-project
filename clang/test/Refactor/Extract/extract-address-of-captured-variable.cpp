
void takesVoidPtr(void *x) { }
void takesPtr(int *x) { }
void takesPtrPtr(int **x) { }

void addressOfVariableImpliesPossibleMutation(int x, int *ip) {
  takesPtr(&x);
// CHECK1-C: (int *x) {\ntakesPtr(x);\n}
// CHECK1-CPP: (int &x) {\ntakesPtr(&x);\n}
  takesVoidPtr(&x);
// CHECK1-C: (int *x) {\ntakesVoidPtr(x);\n}
// CHECK1-CPP: (int &x) {\ntakesVoidPtr(&x);\n}
  &x;
// CHECK1-C: (int *x) {\nreturn x;\n}
// CHECK1-CPP: (int &x) {\nreturn &x;\n}
  *(&x) = 0;
// CHECK1-C: extracted(int *x) {\n*(x) = 0;\n}
// CHECK1-CPP: extracted(int &x) {\n*(&x) = 0;\n}
  takesPtrPtr(&ip);
// CHECK1-C: (int **ip) {\ntakesPtrPtr(ip);\n}
// CHECK1-CPP: (int *&ip) {\ntakesPtrPtr(&ip);\n}
  takesPtr(ip);
// CHECK1-C: (int *ip) {\ntakesPtr(ip);\n}
// CHECK1-CPP: (int *ip) {\ntakesPtr(ip);\n}
  takesVoidPtr(ip);
// CHECK1-C: (int *ip) {\ntakesVoidPtr(ip);\n}
// CHECK1-CPP: (int *ip) {\ntakesVoidPtr(ip);\n}
  takesPtr(&((x)));
// CHECK1-C: (int *x) {\ntakesPtr(((x)));\n}
// CHECK1-CPP: (int &x) {\ntakesPtr(&((x)));\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:7:3-7:15 -selected=%s:10:3-10:19 -selected=%s:13:3-13:5 -selected=%s:16:3-16:12 -selected=%s:19:3-19:19 -selected=%s:22:3-22:15 -selected=%s:25:3-25:19 -selected=%s:28:3-28:19 %s -x c | FileCheck --check-prefix=CHECK1-C %s
// RUN: clang-refactor-test perform -action extract -selected=%s:7:3-7:15 -selected=%s:10:3-10:19 -selected=%s:13:3-13:5 -selected=%s:16:3-16:12 -selected=%s:19:3-19:19 -selected=%s:22:3-22:15 -selected=%s:25:3-25:19 -selected=%s:28:3-28:19 %s | FileCheck --check-prefix=CHECK1-CPP %s

typedef struct {
  int width, height;
} Rectangle;

void takesStructPtr(Rectangle *sp) { }

#ifdef __cplusplus

void addressOfRef(int &x, int *&ip, Rectangle &r) {
  takesPtr(&x);
// CHECK2: (int &x) {\ntakesPtr(&x);\n}
  takesPtrPtr(&ip);
// CHECK2: (int *&ip) {\ntakesPtrPtr(&ip);\n}
  takesStructPtr(&r);
// CHECK2: (Rectangle &r) {\ntakesStructPtr(&r);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:45:3-45:15 -selected=%s:47:3-47:19 -selected=%s:49:3-49:21  %s | FileCheck --check-prefix=CHECK2 %s

#endif

void addressOfArray(int x[]) {
  takesPtrPtr(&x);
// CHECK3-C: (int **x) {\ntakesPtrPtr(x);\n}
// CHECK3-CPP: (int *&x) {\ntakesPtrPtr(&x);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:58:3-58:18 %s -x c | FileCheck --check-prefix=CHECK3-C %s
// RUN: clang-refactor-test perform -action extract -selected=%s:58:3-58:18 %s | FileCheck --check-prefix=CHECK3-CPP %s

typedef struct {
  Rectangle r;
} RectangleInStruct;

void addressOfMember(Rectangle r, RectangleInStruct rs) {
  takesPtr(&r.width);
// CHECK4-C: (Rectangle *r) {\ntakesPtr(&r->width);\n}
// CHECK4-CPP: (Rectangle &r) {\ntakesPtr(&r.width);\n}
  takesPtr(&(rs).r.width);
// CHECK4-C: (RectangleInStruct *rs) {\ntakesPtr(&(rs)->r.width);\n}
// CHECK4-CPP: (RectangleInStruct &rs) {\ntakesPtr(&(rs).r.width);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:71:3-71:21 -selected=%s:74:3-74:26 %s -x c | FileCheck --check-prefix=CHECK4-C %s
// RUN: clang-refactor-test perform -action extract -selected=%s:71:3-71:21 -selected=%s:74:3-74:26 %s | FileCheck --check-prefix=CHECK4-CPP %s

void takesConstPtr(const int *x) { }

#ifdef __cplusplus

void addressOfMember(const Rectangle &r) {
  takesConstPtr(&r.width);
// CHECK5: (const Rectangle &r) {\ntakesConstPtr(&r.width);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:87:3-87:26 %s | FileCheck --check-prefix=CHECK5 %s

class PrivateInstanceVariables {
  int x;
  Rectangle r;

  void method() {
    takesPtr(&x);
// CHECK6: extracted(int &x) {\ntakesPtr(&x);\n}
    takesStructPtr(&(r));
// CHECK6: extracted(Rectangle &r) {\ntakesStructPtr(&(r));\n}
    takesPtr(&((r).width));
// CHECK6: extracted(Rectangle &r) {\ntakesPtr(&((r).width));\n}
  }
};

// RUN: clang-refactor-test perform -action extract -selected=%s:98:5-98:17 -selected=%s:100:5-100:25 -selected=%s:102:5-102:27 %s | FileCheck --check-prefix=CHECK6 %s

#endif

void takesCVoidPtr(const void *x) { }
void takesCPtr(const int *x) { }
void takesCPtrPtr(int * const *x) { }
void takesBothPtrs(const int *x, int *y) {}
void addressForConstUseShouldPassAsConst(int x, int *ip) {
  takesCPtr(&x);
// CHECK7-C: (const int *x) {\ntakesCPtr(x);\n}
// CHECK7-CPP: (const int &x) {\ntakesCPtr(&x);\n}
  takesCVoidPtr((&(x)));
// CHECK7-C: (const int *x) {\ntakesCVoidPtr(((x)));\n}
// CHECK7-CPP: (const int &x) {\ntakesCVoidPtr((&(x)));\n}
  takesCPtrPtr(&ip);
// CHECK7-C: (int *const *ip) {\ntakesCPtrPtr(ip);\n}
// CHECK7-CPP: (int *const &ip) {\ntakesCPtrPtr(&ip);\n}
  takesCPtr(ip);
// CHECK7-C: (int *ip) {\ntakesCPtr(ip);\n}
// CHECK7-CPP: (int *ip) {\ntakesCPtr(ip);\n}
  takesBothPtrs(&x, &x);
// CHECK7-C: (int *x) {\ntakesBothPtrs(x, x);\n}
// CHECK7-CPP: (int &x) {\ntakesBothPtrs(&x, &x);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:116:3-116:16 -selected=%s:119:3-119:22 -selected=%s:122:3-122:20 -selected=%s:125:3-125:16 -selected=%s:128:3-128:24 %s -x c | FileCheck --check-prefix=CHECK7-C %s
// RUN: clang-refactor-test perform -action extract -selected=%s:116:3-116:16 -selected=%s:119:3-119:22 -selected=%s:122:3-122:20 -selected=%s:125:3-125:16 -selected=%s:128:3-128:24 %s | FileCheck --check-prefix=CHECK7-CPP %s

void addressOfConstUseAndMutation(int x) {
  x = 0;
  takesCPtr(&x);
  x = 1;
}
// CHECK8: extracted(int &x) {\nx = 0;\n  takesCPtr(&x);\n}
// CHECK8: extracted(int &x) {\ntakesCPtr(&x);\n  x = 1;\n}

// RUN: clang-refactor-test perform -action extract -selected=%s:137:3-138:16 -selected=%s:138:3-139:8 %s | FileCheck --check-prefix=CHECK8 %s

void takesCStructPtr(const Rectangle *r) { }

void constAddressOfMember(Rectangle r, RectangleInStruct rs) {
  takesCStructPtr(&r);
// CHECK9-C: extracted(const Rectangle *r) {\ntakesCStructPtr(r);\n}
// CHECK9-CPP: extracted(const Rectangle &r) {\ntakesCStructPtr(&r);\n}
  takesCPtr(&r.width);
// CHECK9-C: (const Rectangle *r) {\ntakesCPtr(&r->width);\n}
// CHECK9-CPP: (const Rectangle &r) {\ntakesCPtr(&r.width);\n}
  takesCPtr((&(rs).r.height));
// CHECK9-C: (const RectangleInStruct *rs) {\ntakesCPtr((&(rs)->r.height));\n}
// CHECK9-CPP: (const RectangleInStruct &rs) {\ntakesCPtr((&(rs).r.height));\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:149:3-149:22 -selected=%s:152:3-152:22 -selected=%s:155:3-155:30 %s -x c | FileCheck --check-prefix=CHECK9-C %s
// RUN: clang-refactor-test perform -action extract -selected=%s:149:3-149:22 -selected=%s:152:3-152:22 -selected=%s:155:3-155:30  %s | FileCheck --check-prefix=CHECK9-CPP %s

#ifdef __cplusplus

// RUN: clang-refactor-test perform -action extract -selected=%s:87:3-87:26 %s | FileCheck --check-prefix=CHECK5 %s

class PrivateInstanceVariablesConstAddress {
  int x;
  Rectangle r;

  void method() {
    takesCPtr(&x);
// CHECK10: extracted(const int &x) {\ntakesCPtr(&x);\n}
    takesCStructPtr(&r);
// CHECK10: extracted(const Rectangle &r) {\ntakesCStructPtr(&r);\n}
  }
};

// RUN: clang-refactor-test perform -action extract -selected=%s:172:5-172:18 -selected=%s:174:5-174:24 %s | FileCheck --check-prefix=CHECK10 %s

#endif

void rewriteToPtrWithDerefParensForArrow(Rectangle *r) {
  int y = r->width;
  r = 0;
// CHECK11: (Rectangle **r) {\nint y = (*r)->width;\n  *r = 0;\n}\n\n"
}
// RUN: clang-refactor-test perform -action extract -selected=%s:184:3-185:8 %s -x c | FileCheck --check-prefix=CHECK11 %s

void constAddressWithConditionalOperator(int x, int y) {
  takesCPtr(&(x == 0 ? x : y));
// CHECK12: (const int &x, const int &y) {\ntakesCPtr(&(x == 0 ? x : y));\n}
}
// RUN: clang-refactor-test perform -action extract -selected=%s:191:3-191:31 %s | FileCheck --check-prefix=CHECK12 %s
