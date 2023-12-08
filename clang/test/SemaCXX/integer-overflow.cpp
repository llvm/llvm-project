// RUN: %clang_cc1 %s -verify -fsyntax-only -std=gnu++98 -triple x86_64-pc-linux-gnu
// RUN: %clang_cc1 %s -verify -fsyntax-only -std=gnu++2a -triple x86_64-pc-linux-gnu

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

// Check integer sizes.
int array64[sizeof(uint64_t) == 8 ? 1 : -1];
int array32[sizeof(uint32_t) == 4 ? 1 : -1];
int arrayint[sizeof(int) < sizeof(uint64_t) ? 1 : -1];

uint64_t f0(uint64_t);
uint64_t f1(uint64_t, uint32_t);
uint64_t f2(uint64_t, ...);

static const uint64_t overflow = 1 * 4608 * 1024 * 1024; // expected-warning {{overflow in expression; result is 536870912 with type 'int'}}

uint64_t check_integer_overflows(int i) { //expected-note 0+{{declared here}}
// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  uint64_t overflow = 4608 * 1024 * 1024,
// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
           overflow2 = (uint64_t)(4608 * 1024 * 1024),
// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
           overflow3 = (uint64_t)(4608 * 1024 * 1024 * i),
// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
           overflow4 =  (1ULL * ((4608) * ((1024) * (1024))) + 2ULL),
// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
           overflow5 = static_cast<uint64_t>(4608 * 1024 * 1024),
// expected-warning@+1 2{{overflow in expression; result is 536870912 with type 'int'}}
           multi_overflow = (uint64_t)((uint64_t)(4608 * 1024 * 1024) * (uint64_t)(4608 * 1024 * 1024));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  overflow += overflow2 = overflow3 = (uint64_t)(4608 * 1024 * 1024);
// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  overflow += overflow2 = overflow3 = 4608 * 1024 * 1024;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  overflow += overflow2 = overflow3 = static_cast<uint64_t>(4608 * 1024 * 1024);

  uint64_t not_overflow = 4608 * 1024 * 1024ULL;
  uint64_t not_overflow2 = (1ULL * ((uint64_t)(4608) * (1024 * 1024)) + 2ULL);

// expected-warning@+1 2{{overflow in expression; result is 536870912 with type 'int'}}
  overflow = 4608 * 1024 * 1024 ?  4608 * 1024 * 1024 : 0;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  overflow =  0 ? 0 : 4608 * 1024 * 1024;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  if (4608 * 1024 * 1024)
    return 0;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  if ((uint64_t)(4608 * 1024 * 1024))
    return 1;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  if (static_cast<uint64_t>(4608 * 1024 * 1024))
    return 1;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  if ((uint64_t)(4608 * 1024 * 1024))
    return 2;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  if ((uint64_t)(4608 * 1024 * 1024 * i))
    return 3;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  if ((1ULL * ((4608) * ((1024) * (1024))) + 2ULL))
    return 4;

// expected-warning@+1 2{{overflow in expression; result is 536870912 with type 'int'}}
  if ((uint64_t)((uint64_t)(4608 * 1024 * 1024) * (uint64_t)(4608 * 1024 * 1024)))
    return 5;

#if __cplusplus < 201103L
  switch (i) {
// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  case 4608 * 1024 * 1024:
    return 6;
// expected-warning@+1 {{overflow in expression; result is 537919488 with type 'int'}}
  case (uint64_t)(4609 * 1024 * 1024):
    return 7;
// expected-warning@+1 {{overflow in expression; result is 537919488 with type 'int'}}
  case 1 + static_cast<uint64_t>(4609 * 1024 * 1024):
    return 7;
// expected-error@+1 {{expression is not an integral constant expression}}
  case ((uint64_t)(4608 * 1024 * 1024 * i)):
    return 8;
// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  case ((1ULL * ((4608) * ((1024) * (1024))) + 2ULL)):
    return 9;
// expected-warning@+2 2{{overflow in expression; result is 536870912 with type 'int'}}
// expected-warning@+1 {{overflow converting case value to switch condition type (288230376151711744 to 0)}}
  case ((uint64_t)((uint64_t)(4608 * 1024 * 1024) * (uint64_t)(4608 * 1024 * 1024))):
    return 10;
  }
#endif

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  while (4608 * 1024 * 1024);

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  while ((uint64_t)(4608 * 1024 * 1024));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  while (static_cast<uint64_t>(4608 * 1024 * 1024));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  while ((uint64_t)(4608 * 1024 * 1024));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  while ((uint64_t)(4608 * 1024 * 1024 * i));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  while ((1ULL * ((4608) * ((1024) * (1024))) + 2ULL));

// expected-warning@+1 2{{overflow in expression; result is 536870912 with type 'int'}}
  while ((uint64_t)((uint64_t)(4608 * 1024 * 1024) * (uint64_t)(4608 * 1024 * 1024)));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  do { } while (4608 * 1024 * 1024);

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  do { } while ((uint64_t)(4608 * 1024 * 1024));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  do { } while (static_cast<uint64_t>(4608 * 1024 * 1024));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  do { } while ((uint64_t)(4608 * 1024 * 1024));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  do { } while ((uint64_t)(4608 * 1024 * 1024 * i));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  do { } while ((1ULL * ((4608) * ((1024) * (1024))) + 2ULL));

// expected-warning@+1 2{{overflow in expression; result is 536870912 with type 'int'}}
  do { } while ((uint64_t)((uint64_t)(4608 * 1024 * 1024) * (uint64_t)(4608 * 1024 * 1024)));

// expected-warning@+3 {{overflow in expression; result is 536870912 with type 'int'}}
// expected-warning@+3 {{overflow in expression; result is 536870912 with type 'int'}}
// expected-warning@+3 {{overflow in expression; result is 536870912 with type 'int'}}
  for (uint64_t i = 4608 * 1024 * 1024;
       (uint64_t)(4608 * 1024 * 1024);
       i += (uint64_t)(4608 * 1024 * 1024 * i));

// expected-warning@+3 {{overflow in expression; result is 536870912 with type 'int'}}
// expected-warning@+3 2{{overflow in expression; result is 536870912 with type 'int'}}
// expected-warning@+3 2{{overflow in expression; result is 536870912 with type 'int'}}
  for (uint64_t i = (1ULL * ((4608) * ((1024) * (1024))) + 2ULL);
       ((uint64_t)((uint64_t)(4608 * 1024 * 1024) * (uint64_t)(4608 * 1024 * 1024)));
       i = ((4608 * 1024 * 1024) + ((uint64_t)(4608 * 1024 * 1024))));

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  _Complex long long x = 4608 * 1024 * 1024;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  (__real__ x) = 4608 * 1024 * 1024;

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  (__imag__ x) = 4608 * 1024 * 1024;

// expected-warning@+2 {{overflow in expression; result is 536870912 with type 'int'}}
  uint64_t a[10];
  a[4608 * 1024 * 1024] = 1;
#if __cplusplus < 201103L
// expected-warning@-2 {{array index 536870912 is past the end of the array (that has type 'uint64_t[10]' (aka 'unsigned long long[10]'))}}
// expected-note@-4 {{array 'a' declared here}}
#endif

// expected-warning@+1 2{{overflow in expression; result is 536870912 with type 'int'}}
  return ((4608 * 1024 * 1024) + ((uint64_t)(4608 * 1024 * 1024)));
}

void check_integer_overflows_in_function_calls() {
// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  (void)f0(4608 * 1024 * 1024);

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  uint64_t x = f0(4608 * 1024 * 1024);

// expected-warning@+2 {{overflow in expression; result is 536870912 with type 'int'}}
  uint64_t (*f0_ptr)(uint64_t) = &f0;
  (void)(*f0_ptr)(4608 * 1024 * 1024);

// expected-warning@+1 {{overflow in expression; result is 536870912 with type 'int'}}
  (void)f2(0, f0(4608 * 1024 * 1024));
}

// Tests that ensure that evaluation-for-overflow of random expressions doesn't
// crash.
namespace EvaluationCrashes {
  namespace VirtualCallWithVbase {
    struct A {};
    struct B : virtual A {
      virtual bool f(const void *, int);
    };
    struct C : B {
      bool f(const void *, int);
    };
    int d;
    bool e(C c) {
      if (c.f(&d, d)) {}
      return true;
    }
  }
}

namespace GH31643 {
void f() {
  int a = -(1<<31); // expected-warning {{overflow in expression; result is -2147483648 with type 'int'}}
}
}

#if __cplusplus >= 201103L
namespace GH63629 {
typedef long long int64_t;

template<typename T>
class u_ptr {
  T *ptr;
public:
  u_ptr(const u_ptr&) = delete;
  u_ptr &operator=(const u_ptr&) = delete;
  u_ptr(u_ptr &&other) : ptr(other.ptr) { other.ptr = 0; }
  u_ptr(T *ptr) : ptr(ptr) { }
  ~u_ptr() { delete ptr; }
};

u_ptr<bool> Wrap(int64_t x) {
    return nullptr;
}

int64_t Pass(int64_t x) { return x; }

int m() {
    int64_t x = Pass(30 * 24 * 60 * 59 * 1000);  // expected-warning {{overflow in expression; result is -1746167296 with type 'int'}}
    auto r = Wrap(Pass(30 * 24 * 60 * 59 * 1000));  // expected-warning {{overflow in expression; result is -1746167296 with type 'int'}}
    return 0;
}
}
#endif
