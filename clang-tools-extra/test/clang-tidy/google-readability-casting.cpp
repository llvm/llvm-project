// RUN: %check_clang_tidy %s google-readability-casting %t

bool g() { return false; }

enum Enum { Enum1 };
struct X {};
struct Y : public X {};

void f(int a, double b, const char *cpc, const void *cpv, X *pX) {
  const char *cpc2 = (const char*)cpc;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: redundant cast to the same type [google-readability-casting]
  // CHECK-FIXES: const char *cpc2 = cpc;

  typedef const char *Typedef1;
  typedef const char *Typedef2;
  Typedef1 t1;
  (Typedef2)t1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: possibly redundant cast between typedefs of the same type [google-readability-casting]
  // CHECK-FIXES: {{^}}  (Typedef2)t1;
  (const char*)t1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: possibly redundant cast {{.*}}
  // CHECK-FIXES: {{^}}  (const char*)t1;
  (Typedef1)cpc;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: possibly redundant cast {{.*}}
  // CHECK-FIXES: {{^}}  (Typedef1)cpc;
  (Typedef1)t1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant cast to the same type
  // CHECK-FIXES: {{^}}  t1;

  char *pc = (char*)cpc;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: C-style casts are discouraged; use const_cast [google-readability-casting]
  // CHECK-FIXES: char *pc = const_cast<char*>(cpc);

  char *pc2 = (char*)(cpc + 33);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}}; use const_cast {{.*}}
  // CHECK-FIXES: char *pc2 = const_cast<char*>(cpc + 33);

  const char &crc = *cpc;
  char &rc = (char&)crc;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: {{.*}}; use const_cast {{.*}}
  // CHECK-FIXES: char &rc = const_cast<char&>(crc);

  char &rc2 = (char&)*cpc;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}}; use const_cast {{.*}}
  // CHECK-FIXES: char &rc2 = const_cast<char&>(*cpc);

  char ** const* const* ppcpcpc;
  char ****ppppc = (char****)ppcpcpc;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: {{.*}}; use const_cast {{.*}}
  // CHECK-FIXES: char ****ppppc = const_cast<char****>(ppcpcpc);

  char ***pppc = (char***)*(ppcpcpc);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: {{.*}}; use const_cast {{.*}}
  // CHECK-FIXES: char ***pppc = const_cast<char***>(*(ppcpcpc));

  char ***pppc2 = (char***)(*ppcpcpc);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: {{.*}}; use const_cast {{.*}}
  // CHECK-FIXES: char ***pppc2 = const_cast<char***>(*ppcpcpc);

  char *pc5 = (char*)(const char*)(cpv);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}}; use const_cast {{.*}}
  // CHECK-MESSAGES: :[[@LINE-2]]:22: warning: {{.*}}; use reinterpret_cast {{.*}}
  // CHECK-FIXES: char *pc5 = const_cast<char*>(reinterpret_cast<const char*>(cpv));

  int b1 = (int)b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: {{.*}}; use static_cast {{.*}}
  // CHECK-FIXES: int b1 = static_cast<int>(b);

  Y *pB = (Y*)pX;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}}; use static_cast/const_cast/reinterpret_cast {{.*}}
  Y &rB = (Y&)*pX;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}}; use static_cast/const_cast/reinterpret_cast {{.*}}

  const char *pc3 = (const char*)cpv;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: {{.*}}; use reinterpret_cast {{.*}}
  // CHECK-FIXES: const char *pc3 = reinterpret_cast<const char*>(cpv);

  char *pc4 = (char*)cpv;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}}; use static_cast/const_cast/reinterpret_cast {{.*}}
  // CHECK-FIXES: char *pc4 = (char*)cpv;

  b1 = (int)Enum1;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: {{.*}}; use static_cast {{.*}}
  // CHECK-FIXES: b1 = static_cast<int>(Enum1);

  Enum e = (Enum)b1;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: {{.*}}; use static_cast {{.*}}
  // CHECK-FIXES: Enum e = static_cast<Enum>(b1);

  // CHECK-MESSAGES-NOT: warning:
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
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}}; use static_cast/const_cast/reinterpret_cast {{.*}}
  // CHECK-FIXES: int i = (int)t;
  int j = (int)n;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant cast to the same type
  // CHECK-FIXES: int j = n;
}

template <typename T>
struct TemplateStruct {
  void f(T t, int n) {
    int k = (int)t;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}}; use static_cast/const_cast/reinterpret_cast
    // CHECK-FIXES: int k = (int)t;
    int l = (int)n;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant cast to the same type
    // CHECK-FIXES: int l = n;
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
  const char *cpc2 = (const char*)cpc;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: redundant cast to the same type
  // CHECK-FIXES: const char *cpc2 = cpc;
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
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: g(static_cast<void (*)()>(overloaded_function));
  g((void (*)())&overloaded_function);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: g(static_cast<void (*)()>(&overloaded_function));
  g((void (&)())overloaded_function);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: g(static_cast<void (&)()>(overloaded_function));

  g((FnPtrVoid)overloaded_function);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: g(static_cast<FnPtrVoid>(overloaded_function));
  g((FnPtrVoid)&overloaded_function);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: g(static_cast<FnPtrVoid>(&overloaded_function));
  g((FnRefVoid)overloaded_function);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: g(static_cast<FnRefVoid>(overloaded_function));

  FnPtrVoid fn0 = (void (*)())&overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: FnPtrVoid fn0 = static_cast<void (*)()>(&overloaded_function);
  FnPtrVoid fn1 = (void (*)())overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: FnPtrVoid fn1 = static_cast<void (*)()>(overloaded_function);
  FnPtrVoid fn1a = (FnPtrVoid)overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: FnPtrVoid fn1a = static_cast<FnPtrVoid>(overloaded_function);
  FnRefInt fn2 = (void (&)(int))overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: FnRefInt fn2 = static_cast<void (&)(int)>(overloaded_function);
  auto fn3 = (void (*)())&overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: auto fn3 = static_cast<void (*)()>(&overloaded_function);
  auto fn4 = (void (*)())overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: auto fn4 = static_cast<void (*)()>(overloaded_function);
  auto fn5 = (void (&)(int))overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: auto fn5 = static_cast<void (&)(int)>(overloaded_function);

  void (*fn6)() = (void (*)())&overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: void (*fn6)() = static_cast<void (*)()>(&overloaded_function);
  void (*fn7)() = (void (*)())overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: void (*fn7)() = static_cast<void (*)()>(overloaded_function);
  void (*fn8)() = (FnPtrVoid)overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: void (*fn8)() = static_cast<FnPtrVoid>(overloaded_function);
  void (&fn9)(int) = (void (&)(int))overloaded_function;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: C-style casts are discouraged; use static_cast [
  // CHECK-FIXES: void (&fn9)(int) = static_cast<void (&)(int)>(overloaded_function);

  void (*correct1)() = static_cast<void (*)()>(overloaded_function);
  FnPtrVoid correct2 = static_cast<void (*)()>(&overloaded_function);
  FnRefInt correct3 = static_cast<void (&)(int)>(overloaded_function);
}
