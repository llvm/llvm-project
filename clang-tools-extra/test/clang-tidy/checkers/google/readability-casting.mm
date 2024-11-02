// RUN: clang-tidy %s -checks=-*,google-readability-casting -- \
// RUN:   -xobjective-c++ -fobjc-abi-version=2 -fobjc-arc | count 0

// Note: this test expects no diagnostics, but FileCheck cannot handle that,
// hence the use of | count 0.

bool g() { return false; }

enum Enum { Enum1 };
struct X {};
struct Y : public X {};

void f(int a, double b, const char *cpc, const void *cpv, X *pX) {

  typedef const char *Typedef1;
  typedef const char *Typedef2;
  Typedef1 t1;
  (Typedef2)t1;
  (const char*)t1;
  (Typedef1)cpc;

  typedef char Char;
  char *pc;
  Char *pChar = (Char*)pc;

  (Char)*cpc;

  (char)*pChar;

  (const char*)cpv;

  char *pc2 = (char*)(cpc + 33);

  const char &crc = *cpc;
  char &rc = (char&)crc;

  char &rc2 = (char&)*cpc;

  char ** const* const* ppcpcpc;
  char ****ppppc = (char****)ppcpcpc;

  char ***pppc = (char***)*(ppcpcpc);

  char ***pppc2 = (char***)(*ppcpcpc);

  char *pc5 = (char*)(const char*)(cpv);

  int b1 = (int)b;
  b1 = (const int&)b;

  b1 = (int) b;

  b1 = (int)         b;

  b1 = (int) (b);

  b1 = (int)         (b);

  Y *pB = (Y*)pX;
  Y &rB = (Y&)*pX;

  const char *pc3 = (const char*)cpv;

  char *pc4 = (char*)cpv;

  b1 = (int)Enum1;

  Enum e = (Enum)b1;

  int b2 = int(b);
  int b3 = static_cast<double>(b);
  int b4 = b;
  double aa = a;
  (void)b2;
  return (void)g();
}

template <typename T>
void template_function(T t, int n) {
  int i = (int)t;
}

template <typename T>
struct TemplateStruct {
  void f(T t, int n) {
    int k = (int)t;
  }
};

void test_templates() {
  template_function(1, 42);
  template_function(1.0, 42);
  TemplateStruct<int>().f(1, 42);
  TemplateStruct<double>().f(1.0, 42);
}

extern "C" {
void extern_c_code(const char *cpc) {
  char *pc = (char*)cpc;
}
}

#define CAST(type, value) (type)(value)
void macros(double d) {
  int i = CAST(int, d);
}

enum E { E1 = 1 };
template <E e>
struct A {
  // Usage of template argument e = E1 is represented as (E)1 in the AST for
  // some reason. We have a special treatment of this case to avoid warnings
  // here.
  static const E ee = e;
};
struct B : public A<E1> {};


void overloaded_function();
void overloaded_function(int);

template<typename Fn>
void g(Fn fn) {
  fn();
}

void function_casts() {
  typedef void (*FnPtrVoid)();
  typedef void (&FnRefVoid)();
  typedef void (&FnRefInt)(int);

  g((void (*)())overloaded_function);
  g((void (*)())&overloaded_function);
  g((void (&)())overloaded_function);

  g((FnPtrVoid)overloaded_function);
  g((FnPtrVoid)&overloaded_function);
  g((FnRefVoid)overloaded_function);

  FnPtrVoid fn0 = (void (*)())&overloaded_function;
  FnPtrVoid fn1 = (void (*)())overloaded_function;
  FnPtrVoid fn1a = (FnPtrVoid)overloaded_function;
  FnRefInt fn2 = (void (&)(int))overloaded_function;
  auto fn3 = (void (*)())&overloaded_function;
  auto fn4 = (void (*)())overloaded_function;
  auto fn5 = (void (&)(int))overloaded_function;

  void (*fn6)() = (void (*)())&overloaded_function;
  void (*fn7)() = (void (*)())overloaded_function;
  void (*fn8)() = (FnPtrVoid)overloaded_function;
  void (&fn9)(int) = (void (&)(int))overloaded_function;

  void (*correct1)() = static_cast<void (*)()>(overloaded_function);
  FnPtrVoid correct2 = static_cast<void (*)()>(&overloaded_function);
  FnRefInt correct3 = static_cast<void (&)(int)>(overloaded_function);
}

struct S {
    S(const char *);
};
struct ConvertibleToS {
  operator S() const;
};
struct ConvertibleToSRef {
  operator const S&() const;
};

void conversions() {
  //auto s1 = (const S&)"";
  auto s2 = (S)"";
  auto s2a = (struct S)"";
  auto s2b = (const S)"";
  ConvertibleToS c;
  auto s3 = (const S&)c;
  auto s4 = (S)c;
  ConvertibleToSRef cr;
  auto s5 = (const S&)cr;
  auto s6 = (S)cr;
}
