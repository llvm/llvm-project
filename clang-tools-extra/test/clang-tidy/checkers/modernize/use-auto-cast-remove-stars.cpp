// RUN: %check_clang_tidy %s modernize-use-auto %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-auto.RemoveStars: 'true' , modernize-use-auto.MinTypeNameLength: '0'}}" \
// RUN:   -- -frtti

struct A {
  virtual ~A() {}
};

struct B : public A {};

struct C {};

void f_static_cast() {
  long l = 1;
  int i1 = static_cast<int>(l);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  i1 = static_cast<int>(l);

  const int i2 = static_cast<int>(l);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: const auto  i2 = static_cast<int>(l);

  long long ll = static_cast<long long>(l);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  ll = static_cast<long long>(l);
  unsigned long long ull = static_cast<unsigned long long>(l);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  ull = static_cast<unsigned long long>(l);
  unsigned int ui = static_cast<unsigned int>(l);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  ui = static_cast<unsigned int>(l);
  long double ld = static_cast<long double>(l);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  ld = static_cast<long double>(l);

  A *a = new B();
  B *b1 = static_cast<B *>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto b1 = static_cast<B *>(a);

  B *const b2 = static_cast<B *>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto const b2 = static_cast<B *>(a);

  const B *b3 = static_cast<const B *>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: const auto b3 = static_cast<const B *>(a);

  B &b4 = static_cast<B &>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &b4 = static_cast<B &>(*a);

  const B &b5 = static_cast<const B &>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: const auto  &b5 = static_cast<const B &>(*a);

  B &b6 = static_cast<B &>(*a), &b7 = static_cast<B &>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &b6 = static_cast<B &>(*a), &b7 = static_cast<B &>(*a);

  // Don't warn when non-cast involved
  long double cast = static_cast<long double>(l), noncast = 5;

  // Don't warn when auto is already being used.
  auto i3 = static_cast<int>(l);
  auto *b8 = static_cast<B *>(a);
  auto &b9 = static_cast<B &>(*a);
}

void f_dynamic_cast() {
  A *a = new B();
  B *b1 = dynamic_cast<B *>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto b1 = dynamic_cast<B *>(a);

  B &b2 = dynamic_cast<B &>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &b2 = dynamic_cast<B &>(*a);
}

void f_reinterpret_cast() {
  auto *a = new A();
  C *c1 = reinterpret_cast<C *>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto c1 = reinterpret_cast<C *>(a);

  C &c2 = reinterpret_cast<C &>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &c2 = reinterpret_cast<C &>(*a);
}

void f_const_cast() {
  const A *a1 = new A();
  A *a2 = const_cast<A *>(a1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto a2 = const_cast<A *>(a1);
  A &a3 = const_cast<A &>(*a1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &a3 = const_cast<A &>(*a1);
}

typedef unsigned char xmlChar;
#define BAD_CAST (xmlChar *)

#define XMLCHAR_CAST(x) (xmlChar *)(x)

#define CAST_IN_MACRO(x)         \
  do {                           \
    xmlChar *s = (xmlChar *)(x); \
  } while (false);
// CHECK-FIXES: xmlChar *s = (xmlChar *)(x); {{\\}}

void f_cstyle_cast() {
  auto *a = new A();
  C *c1 = (C *)a;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto c1 = (C *)a;

  C &c2 = (C &)*a;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &c2 = (C &)*a;

  xmlChar  *s = BAD_CAST "xml";
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto s = BAD_CAST "xml";
  xmlChar  *t = XMLCHAR_CAST("xml");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto t = XMLCHAR_CAST("xml");
  CAST_IN_MACRO("xml");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
}

void f_functional_cast() {
  long l = 1;
  int i1 = int(l);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  i1 = int(l);

  B b;
  A a = A(b);
}

class StringRef
{
public:
  StringRef(const char *);
  const char *begin() const;
  const char *end() const;
};

template <typename T, typename U>
T template_value_cast(const U &u);

template <typename T, typename U>
T *template_pointer_cast(U *u);

template <typename T, typename U>
T &template_reference_cast(U &u);

template <typename T, typename U>
const T *template_const_pointer_cast(const U *u);

template <typename T, typename U>
const T &template_const_reference_cast(const U &u);

template <typename T>
T template_value_get(StringRef s);

struct S {
  template <typename T>
  const T *template_member_get();
};

template <typename T>
T max(T t1, T t2);

void f_template_cast()
{
  double d = 0;
  int i1 = template_value_cast<int>(d);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: auto  i1 = template_value_cast<int>(d);

  A *a = new B();
  B *b1 = template_value_cast<B *>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: auto b1 = template_value_cast<B *>(a);
  B &b2 = template_value_cast<B &>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &b2 = template_value_cast<B &>(*a);
  B *b3 = template_pointer_cast<B>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: auto b3 = template_pointer_cast<B>(a);
  B &b4 = template_reference_cast<B>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &b4 = template_reference_cast<B>(*a);
  const B *b5 = template_const_pointer_cast<B>(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: const auto b5 = template_const_pointer_cast<B>(a);
  const B &b6 = template_const_reference_cast<B>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: const auto  &b6 = template_const_reference_cast<B>(*a);
  B *b7 = template_value_get<B *>("foo");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: auto b7 = template_value_get<B *>("foo");
  B *b8 = template_value_get<B *>("foo"), *b9 = template_value_get<B *>("bar");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: auto b8 = template_value_get<B *>("foo"), b9 = template_value_get<B *>("bar");

  S s;
  const B *b10 = s.template_member_get<B>();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use auto when initializing with a template cast to avoid duplicating the type name
  // CHECK-FIXES: const auto b10 = s.template_member_get<B>();

  // Don't warn when auto is already being used.
  auto i2 = template_value_cast<int>(d);
  auto *i3 = template_value_cast<int *>(d);
  auto **i4 = template_value_cast<int **>(d);
  auto &i5 = template_reference_cast<int>(d);

  // Don't warn for implicit template arguments.
  int i6 = max(i1, i2);

  // Don't warn for mismatched var and initializer types.
  A *a1 = template_value_cast<B *>(a);

  // Don't warn for mismatched var types.
  B *b11 = template_value_get<B *>("foo"), b12 = template_value_get<B>("bar");

  // Don't warn for implicit variables.
  for (auto &c : template_reference_cast<StringRef>(*a)) {
  }
}
