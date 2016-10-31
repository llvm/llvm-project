// RUN: %check_clang_tidy %s modernize-use-auto %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-use-auto.RemoveStars, value: '1'}]}" \
// RUN:   -- -std=c++11 -frtti

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
  // CHECK-FIXES: auto b3 = static_cast<const B *>(a);

  B &b4 = static_cast<B &>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &b4 = static_cast<B &>(*a);

  const B &b5 = static_cast<const B &>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: const auto  &b5 = static_cast<const B &>(*a);

  B &b6 = static_cast<B &>(*a), &b7 = static_cast<B &>(*a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing with a cast to avoid duplicating the type name
  // CHECK-FIXES: auto  &b6 = static_cast<B &>(*a), &b7 = static_cast<B &>(*a);

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
// CHECK-FIXES: xmlChar *s = (xmlChar *)(x);

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
