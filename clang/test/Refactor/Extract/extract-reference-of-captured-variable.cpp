
void takesPtr(int *x) { }

typedef struct {
  int width, height;
} Rectangle;

void takesStructPtr(Rectangle *sp) { }

void variableTakesRef(int x, Rectangle r) {
  int &y = x;
  takesPtr(&y);
// CHECK1: (int &x) {\nint &y = x;\n  takesPtr(&y);\n}
  Rectangle p = r;
  Rectangle &rp = p;
  takesStructPtr(&rp);
// CHECK1: (const Rectangle &r) {\nRectangle p = r;\n  Rectangle &rp = p;\n  takesStructPtr(&rp);\n}
// CHECK1: (Rectangle &p) {\nRectangle &rp = p;\n  takesStructPtr(&rp);\n}
  int &member = ((r).width);
  int z = member;
// CHECK1: (Rectangle &r) {\nint &member = ((r).width);\n  int z = member;\n}

// Even though y takes a reference to x, we still want to pass it by value here.
  int a = x;
// CHECK1: (int x) {\nint a = x;\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:11:3-12:15 -selected=%s:14:3-16:22 -selected=%s:15:3-16:22 -selected=%s:19:3-20:17 -selected=%s:24:3-24:12 %s | FileCheck --check-prefix=CHECK1 %s

class PrivateInstanceVariables {
  int x;
  Rectangle r;

  void method() {
    int &y = x;
// CHECK2: extracted(int &x) {\nint &y = x;\n}
    Rectangle &rr = r;
// CHECK2: extracted(Rectangle &r) {\nRectangle &rr = r;\n}
    int &z = ((r).width);
// CHECK2: extracted(Rectangle &r) {\nint &z = ((r).width);\n}
  }
};

// RUN: clang-refactor-test perform -action extract -selected=%s:35:5-35:15 -selected=%s:37:5-37:22 -selected=%s:39:5-39:25 %s | FileCheck --check-prefix=CHECK2 %s

#ifdef USECONST
#define CONST const
#else
#define CONST
#endif

void takesRef(CONST int &x) { }

void takesStructRef(CONST Rectangle &r) { }

void takesValue(int x) { }

struct ConsTakesRef {
  ConsTakesRef(CONST int &x) { }

  void takesRef(CONST int &x) const { }
  void takesValue(int x) const { }
};

int operator << (CONST Rectangle &r, CONST int &x) { return 0; }

void callTakesRef(int x, Rectangle r) {
  takesRef(x);
// CHECK3: extracted(int &x) {\ntakesRef(x);\n}
// CHECK4: extracted(int x) {\ntakesRef(x);\n}
  takesValue(x);
// CHECK3: extracted(int x) {\ntakesValue(x);\n}
// CHECK4: extracted(int x) {\ntakesValue(x);\n}
  auto k = ConsTakesRef(x); auto y = ConsTakesRef(x);
// CHECK3: extracted(int &x) {\nauto k = ConsTakesRef(x);\n}
// CHECK4: extracted(int x) {\nauto k = ConsTakesRef(x);\n}
  y.takesRef((x));
// CHECK3: extracted(int &x, const ConsTakesRef &y) {\ny.takesRef((x));\n}
// CHECK4: extracted(int x, const ConsTakesRef &y) {\ny.takesRef((x));\n}
  y.takesValue(x);
// CHECK3: extracted(int x, const ConsTakesRef &y) {\ny.takesValue(x);\n}
// CHECK4: extracted(int x, const ConsTakesRef &y) {\ny.takesValue(x);\n}
  takesStructRef((r));
// CHECK3: extracted(Rectangle &r) {\ntakesStructRef((r));\n}
// CHECK4: extracted(const Rectangle &r) {\ntakesStructRef((r));\n}
  takesRef((r).height);
// CHECK3: extracted(Rectangle &r) {\ntakesRef((r).height);\n}
// CHECK4: extracted(const Rectangle &r) {\ntakesRef((r).height);\n}
  y.takesRef(r.width);
// CHECK3: extracted(Rectangle &r, const ConsTakesRef &y) {\ny.takesRef(r.width);\n}
// CHECK4: extracted(const Rectangle &r, const ConsTakesRef &y) {\ny.takesRef(r.width);\n}
  takesValue(r.width);
// CHECK3: extracted(const Rectangle &r) {\ntakesValue(r.width);\n}
// CHECK4: extracted(const Rectangle &r) {\ntakesValue(r.width);\n}
  r << x;
// CHECK3: extracted(Rectangle &r, int &x) {\nreturn r << x;\n}
// CHECK4: extracted(const Rectangle &r, int x) {\nreturn r << x;\n}

  int &r1 = x;
  takesRef(x);
// CHECK3: extracted(int &x) {\nint &r1 = x;\n  takesRef(x);\n}
// CHECK4: extracted(int &x) {\nint &r1 = x;\n  takesRef(x);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:68:3-68:14 -selected=%s:71:3-71:16 -selected=%s:74:3-74:27 -selected=%s:77:3-77:18 -selected=%s:80:3-80:18 -selected=%s:83:3-83:22 -selected=%s:86:3-86:23 -selected=%s:89:3-89:22 -selected=%s:92:3-92:22 -selected=%s:95:3-95:9 -selected=%s:99:3-100:14 %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:68:3-68:14 -selected=%s:71:3-71:16 -selected=%s:74:3-74:27 -selected=%s:77:3-77:18 -selected=%s:80:3-80:18 -selected=%s:83:3-83:22 -selected=%s:86:3-86:23 -selected=%s:89:3-89:22 -selected=%s:92:3-92:22 -selected=%s:95:3-95:9 -selected=%s:99:3-100:14 %s -DUSECONST | FileCheck --check-prefix=CHECK4 %s

void takesConstRef(const int &x) { }

void callTakeRefAndConstRef(int x) {
  takesRef(x);
  takesConstRef(x);
// CHECK5: extracted(int &x) {\ntakesRef(x);\n  takesConstRef(x);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:111:3-112:19 %s | FileCheck --check-prefix=CHECK5 %s

class PrivateInstanceVariablesCallRefs {
  int x;
  Rectangle r;

  void callsTakeRef() {
    takesRef(x);
// CHECK6: extracted(int &x) {\ntakesRef(x);\n}
// CHECK7: extracted(int x) {\ntakesRef(x);\n}
    takesStructRef(r);
// CHECK6: extracted(Rectangle &r) {\ntakesStructRef(r);\n}
// CHECK7: extracted(const Rectangle &r) {\ntakesStructRef(r);\n}
    takesRef(r.width);
// CHECK6: extracted(Rectangle &r) {\ntakesRef(r.width);\n}
// CHECK7: extracted(const Rectangle &r) {\ntakesRef(r.width);\n}
  }
};

// RUN: clang-refactor-test perform -action extract -selected=%s:123:5-123:16 -selected=%s:126:5-126:22 -selected=%s:129:5-129:22 %s | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:123:5-123:16 -selected=%s:126:5-126:22 -selected=%s:129:5-129:22 %s -DUSECONST | FileCheck --check-prefix=CHECK7 %s

void variableTakesConstRef(int x, Rectangle r) {
  const int &y = x;
// CHECK8: extracted(int x) {\nconst int &y = x;\n}
  const Rectangle &p = r;
// CHECK8: extracted(const Rectangle &r) {\nconst Rectangle &p = r;\n}
  const int &z = r.width;
// CHECK8: extracted(const Rectangle &r) {\nconst int &z = r.width;\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:139:3-139:19 -selected=%s:141:3-141:25 -selected=%s:143:3-143:25 %s | FileCheck --check-prefix=CHECK8 %s

class ClassWithMethod {
public:
  int method() CONST { return 0; }
  int operator + (int x) CONST { return x; }
};

void nonConstMethodCallImpliesNonConstReceiver(ClassWithMethod x) {
  x.method();
// CHECK10: extracted(ClassWithMethod &x) {\nreturn x.method();\n}
// CHECK11: extracted(const ClassWithMethod &x) {\nreturn x.method();\n}
  x.operator +(2);
// CHECK10: extracted(ClassWithMethod &x) {\nreturn x.operator +(2);\n}
// CHECK11: extracted(const ClassWithMethod &x) {\nreturn x.operator +(2);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:156:3-156:14 -selected=%s:159:3-159:18 %s | FileCheck --check-prefix=CHECK10 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:156:3-156:14 -selected=%s:159:3-159:18 %s -DUSECONST | FileCheck --check-prefix=CHECK11 %s

void ignoreMethodCallsOnPointer(ClassWithMethod *x) {
  x->method();
// CHECK12: extracted(ClassWithMethod *x) {\nreturn x->method();\n}
  x->operator +(2);
// CHECK12: extracted(ClassWithMethod *x) {\nreturn x->operator +(2);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:168:3-168:13 -selected=%s:170:3-170:19 %s | FileCheck --check-prefix=CHECK12 %s

void takesRValueRef(int &&x) { }
void takesRValueStructRef(Rectangle &&r) { }

void callTakesRValueRef(int x) {
  takesRValueRef(static_cast<int&&>(x));
// CHECK13: extracted(int &x) {\ntakesRValueRef(static_cast<int&&>(x));\n}
  Rectangle r;
  takesRValueStructRef((static_cast<Rectangle&&>(r)));
// CHECK13: extracted(Rectangle &r) {\ntakesRValueStructRef((static_cast<Rectangle&&>(r)));\n}
  int &&y = static_cast<int&&>(r.height);
// CHECK13: extracted(Rectangle &r) {\nint &&y = static_cast<int&&>(r.height);\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:180:3-180:20 -selected=%s:183:3-183:54 -selected=%s:185:3-185:41 %s | FileCheck --check-prefix=CHECK13 %s

void referencesInConditionalOperator(int x, int y) {
  takesRef(x == 0 ? x : y);
// CHECK14: (int &x, int &y) {\ntakesRef(x == 0 ? x : y);\n}
// CHECK15: (int x, int y) {\ntakesRef(x == 0 ? x : y);\n}
  Rectangle a, b;
  takesStructRef(y == 0 ? (a) : b);
// CHECK14: (Rectangle &a, Rectangle &b, int y) {\ntakesStructRef(y == 0 ? (a) : b);\n}
// CHECK15: (const Rectangle &a, const Rectangle &b, int y) {\ntakesStructRef(y == 0 ? (a) : b);\n}
  takesRef(x == 0 ? (a).width : (y == 0 ? y : b.height));
// CHECK14: (Rectangle &a, Rectangle &b, int x, int &y) {\ntakesRef(x == 0 ? (a).width : (y == 0 ? y : b.height));\n}
// CHECK15: (const Rectangle &a, const Rectangle &b, int x, int y) {\ntakesRef(x == 0 ? (a).width : (y == 0 ? y : b.height));\n}
  takesRef((x == 0 ? a : (b)).width);
// CHECK14: (Rectangle &a, Rectangle &b, int x) {\ntakesRef((x == 0 ? a : (b)).width);\n}
// CHECK15: (const Rectangle &a, const Rectangle &b, int x) {\ntakesRef((x == 0 ? a : (b)).width);\n}
  takesRef(x == 0 ? y : y);
// CHECK14: (int x, int &y) {\ntakesRef(x == 0 ? y : y);\n}
// CHECK15: (int x, int y) {\ntakesRef(x == 0 ? y : y);\n}
  ClassWithMethod caller1, caller2;
  (x == 0 ? caller1 : caller2).method();
// CHECK14: (ClassWithMethod &caller1, ClassWithMethod &caller2, int x) {\nreturn (x == 0 ? caller1 : caller2).method();\n}
// CHECK15: (const ClassWithMethod &caller1, const ClassWithMethod &caller2, int x) {\nreturn (x == 0 ? caller1 : caller2).method();\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:192:3-192:27 -selected=%s:196:3-196:35 -selected=%s:199:3-199:57 -selected=%s:202:3-202:37 -selected=%s:205:3-205:27 -selected=%s:209:3-209:40 %s | FileCheck --check-prefix=CHECK14 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:192:3-192:27 -selected=%s:196:3-196:35 -selected=%s:199:3-199:57 -selected=%s:202:3-202:37 -selected=%s:205:3-205:27 -selected=%s:209:3-209:40 %s -DUSECONST | FileCheck --check-prefix=CHECK15 %s

class PrivateInstanceVariablesConditionalOperatorRefs {
  int x;
  Rectangle r;

  void callsTakeRef(int y) {
    takesRef(y == 0 ? x : r.width);
  }
};
// CHECK16: (Rectangle &r, int &x, int y) {\ntakesRef(y == 0 ? x : r.width);\n}
// RUN: clang-refactor-test perform -action extract -selected=%s:222:5-222:35 %s | FileCheck --check-prefix=CHECK16 %s

class ReferencesInCommaOperator {
  int x;

  void callsTakeRef(int y, Rectangle r) {
    takesRef((x, y));
// CHECK17: (int x, int &y) {\ntakesRef((x, y));\n}
    takesRef((y, x));
// CHECK17: (int &x, int y) {\ntakesRef((y, x));\n}
    takesStructRef((takesValue(x), r));
// CHECK17: (Rectangle &r, int x) {\ntakesStructRef((takesValue(x), r));\n}
  }
};

// RUN: clang-refactor-test perform -action extract -selected=%s:232:5-232:21 -selected=%s:234:5-234:21 -selected=%s:236:5-236:39 %s | FileCheck --check-prefix=CHECK17 %s

struct StaticMember {
  static int staticMember;
};

void memberMustBeNonStaticField(StaticMember s) {
  takesRef(s.staticMember);
// CHECK18: (const StaticMember &s) {\ntakesRef(s.staticMember);\n}
}
// RUN: clang-refactor-test perform -action extract -selected=%s:248:3-248:27 %s | FileCheck --check-prefix=CHECK18 %s

class ClassWithMethod2 {
public:
  ClassWithMethod member;
};

class ClassWithMethod;

void nonConstMethodCallImpliesNonConstReceiver2(ClassWithMethod2 x) {
  x.member.method();
// CHECK19: (ClassWithMethod2 &x) {\nreturn x.member.method();\n}
// CHECK20: (const ClassWithMethod2 &x) {\nreturn x.member.method();\n}
}
// RUN: clang-refactor-test perform -action extract -selected=%s:261:3-261:20 %s | FileCheck --check-prefix=CHECK19 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:261:3-261:20 %s -DUSECONST | FileCheck --check-prefix=CHECK20 %s

class PrivateInstaceVariablesCallRefsBase {
  int x;
};

class PrivateInstanceVariablesCallRefs2: public PrivateInstaceVariablesCallRefsBase {
  int y;
  Rectangle r;

  void callsTakeRef() {
    takesRef(this->x);
    takesRef((this)->y);
    takesRef(static_cast<PrivateInstaceVariablesCallRefsBase *>(this)->x);
    takesRef((0, ((const_cast<PrivateInstanceVariablesCallRefs2 *>(this)->r.width))));
  }
// CHECK21: (PrivateInstanceVariablesCallRefs2 &object) {\ntakesRef(object.x);\n}
// CHECK21: (PrivateInstanceVariablesCallRefs2 &object) {\ntakesRef((object).y);\n}
// CHECK21: (PrivateInstanceVariablesCallRefs2 &object) {\ntakesRef(static_cast<PrivateInstaceVariablesCallRefsBase *>(&object)->x);\n}
// CHECK21: (PrivateInstanceVariablesCallRefs2 &object) {\ntakesRef((0, ((const_cast<PrivateInstanceVariablesCallRefs2 *>(&object)->r.width))));\n}
};
// RUN: clang-refactor-test perform -action extract -selected=%s:277:5-277:22 -selected=%s:278:5-278:24 -selected=%s:279:5-279:74 -selected=%s:280:5-280:86 %s | FileCheck --check-prefix=CHECK21 %s
