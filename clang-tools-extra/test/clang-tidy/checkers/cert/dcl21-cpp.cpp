// RUN: %check_clang_tidy %s cert-dcl21-cpp %t

class A {};

A operator++(A &, int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: overloaded 'operator++' returns a non-constant object instead of a constant object type [cert-dcl21-cpp]
// CHECK-FIXES: {{^}}const A operator++(A &, int);

A operator--(A &, int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: overloaded 'operator--' returns a no
// CHECK-FIXES: {{^}}const A operator--(A &, int);

class B {};

B &operator++(B &);
const B operator++(B &, int);

B &operator--(B &);
const B operator--(B &, int);


class D {
D &operator++();
const D operator++(int);

D &operator--();
const D operator--(int);
};

class C {
C operator++(int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: overloaded 'operator++' returns a no
// CHECK-FIXES: {{^}}const C operator++(int);

C operator--(int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: overloaded 'operator--' returns a no
// CHECK-FIXES: {{^}}const C operator--(int);
};

class E {};

E &operator++(E &, int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: overloaded 'operator++' returns a reference instead of a constant object type [cert-dcl21-cpp]
// CHECK-FIXES: {{^}}const E operator++(E &, int);

E &operator--(E &, int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: overloaded 'operator--' returns a re
// CHECK-FIXES: {{^}}const E operator--(E &, int);

class G {
G &operator++(int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: overloaded 'operator++' returns a re
// CHECK-FIXES: {{^}}const G operator++(int);

G &operator--(int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: overloaded 'operator--' returns a re
// CHECK-FIXES: {{^}}const G operator--(int);
};

class F {};

const F &operator++(F &, int);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: overloaded 'operator++' returns a re
// CHECK-FIXES: {{^}}const F operator++(F &, int);

const F &operator--(F &, int);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: overloaded 'operator--' returns a re
// CHECK-FIXES: {{^}}const F operator--(F &, int);

class H {
const H &operator++(int);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: overloaded 'operator++' returns a re
// CHECK-FIXES: {{^}}const H operator++(int);

const H &operator--(int);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: overloaded 'operator--' returns a re
// CHECK-FIXES: {{^}}const H operator--(int);
};


#define FROM_MACRO P&
class P {
const FROM_MACRO operator++(int);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: overloaded 'operator++' returns a re
// CHECK-FIXES: {{^}}const FROM_MACRO operator++(int);
};


template<typename T>
class Q {
const Q &operator++(int);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: overloaded 'operator++' returns a re
// CHECK-FIXES: {{^}}const Q<T> operator++(int);

const Q &operator--(int);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: overloaded 'operator--' returns a re
// CHECK-FIXES: {{^}}const Q<T> operator--(int);
};

void foobar() {
  Q<int> a;
  Q<float> b;
  (void)a;
  (void)b;
}

struct S {};
typedef S& SRef;

SRef operator++(SRef, int);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: overloaded 'operator++' returns a re
// CHECK-FIXES: {{^}}SRef operator++(SRef, int);

struct T {
  typedef T& TRef;
  
  TRef operator++(int);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: overloaded 'operator++' returns a re
// CHECK-FIXES: {{^}}  TRef operator++(int);
};

struct U {
  typedef const U& ConstURef;
  
  ConstURef& operator++(int);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: overloaded 'operator++' returns a re
// CHECK-FIXES: {{^}}  ConstURef& operator++(int);
};

struct V {
  V *operator++(int);
  V *const operator--(int);
};

