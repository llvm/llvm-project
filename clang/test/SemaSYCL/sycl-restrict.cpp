// RUN: %clang_cc1 -fcxx-exceptions -fsycl-is-device -Wno-return-type -verify -fsyntax-only -x c++ -emit-llvm-only -std=c++17 %s
// RUN: %clang_cc1 -fcxx-exceptions -fsycl-is-device -fno-sycl-allow-func-ptr -Wno-return-type -verify -fsyntax-only -x c++ -emit-llvm-only -std=c++17 %s
// RUN: %clang_cc1 -fcxx-exceptions -fsycl-is-device -DALLOW_FP=1 -fsycl-allow-func-ptr -Wno-return-type -verify -fsyntax-only -x c++ -emit-llvm-only -std=c++17 %s


namespace std {
  class type_info;
  typedef __typeof__(sizeof(int)) size_t;
}
namespace Check_User_Operators {
class Fraction
{
    int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
    int n, d;
public:
    Fraction(int n, int d = 1) : n(n/gcd(n, d)), d(d/gcd(n, d)) { }
    int num() const { return n; }
    int den() const { return d; }
};
bool operator==(const Fraction& lhs, const Fraction& rhs)
{
    // expected-error@+1 {{SYCL kernel cannot allocate storage}}
    new int;
    return lhs.num() == rhs.num() && lhs.den() == rhs.den();
}}

namespace Check_VLA_Restriction {
void no_restriction(int p) {
  int index[p+2];
}
void restriction(int p) {
  // expected-error@+1 {{variable length arrays are not supported for the current target}}
  int index[p+2];
}
}

void* operator new (std::size_t size, void* ptr) throw() { return ptr; };
namespace Check_RTTI_Restriction {
// expected-error@+1 5{{No class with a vtable can be used in a SYCL kernel or any code included in the kernel}}
struct A {
  virtual ~A(){};
};

struct B : public A {
  B() : A() {}
};

struct OverloadedNewDelete {
  // This overload allocates storage, give diagnostic.
  void *operator new(std::size_t size) throw() {
  // expected-error@+1 {{SYCL kernel cannot allocate storage}}
    float *pt = new float;
    return 0;}
  // This overload does not allocate: no diagnostic.
  void *operator new[](std::size_t size) throw() {return 0;}
  void operator delete(void *){};
  void operator delete[](void *){};
};

bool isa_B(A *a) {
  Check_User_Operators::Fraction f1(3, 8), f2(1, 2), f3(10, 2);
  if (f1 == f2) return false;

  Check_VLA_Restriction::restriction(7);
  // expected-error@+1 {{SYCL kernel cannot allocate storage}}
  int *ip = new int;
  int i; int *p3 = new(&i) int; // no error on placement new
  OverloadedNewDelete *x = new( struct OverloadedNewDelete );
  auto y = new struct OverloadedNewDelete [5];
  // expected-error@+1 {{SYCL kernel cannot use rtti}}
  (void)typeid(int);
  // expected-error@+2 {{SYCL kernel cannot use rtti}}
  // expected-note@+1{{used here}}
  return dynamic_cast<B *>(a) != 0;
}

__attribute__((sycl_kernel)) void kernel1(void) {
  // expected-note@+1{{used here}}
  A *a;
  // expected-note@+1 3{{used here}}
  isa_B(a);
}
}
// expected-error@+1 {{No class with a vtable can be used in a SYCL kernel or any code included in the kernel}}
typedef struct Base {
  virtual void f() const {}
} b_type;

typedef struct A {
  static int stat_member;
  const static int const_stat_member;
  constexpr static int constexpr_stat_member=0;

  int fm(void)
  {
    // expected-error@+1 {{SYCL kernel cannot use a non-const static data variable}}
    return stat_member;
  }
} a_type;


b_type b;

using myFuncDef = int(int,int);

void eh_ok(void)
{
  try {
    ;
  } catch (...) {
    ;
  }
  throw 20;
}

void eh_not_ok(void)
{
  // expected-error@+1 {{SYCL kernel cannot use exceptions}}
  try {
    ;
  // expected-error@+1 {{SYCL kernel cannot use exceptions}}
  } catch (...) {
    ;
  }
  // expected-error@+1 {{SYCL kernel cannot use exceptions}}
  throw 20;
}

void usage(  myFuncDef functionPtr ) {

  eh_not_ok();

#if ALLOW_FP
  // No error message for function pointer.
#else
  // expected-error@+2 {{SYCL kernel cannot call through a function pointer}}
#endif
  if ((*functionPtr)(1,2))
  // expected-note@+3{{used here}}
  // expected-error@+2 {{SYCL kernel cannot use a global variable}}
  // expected-error@+1 {{SYCL kernel cannot call a virtual function}}
    b.f();
  Check_RTTI_Restriction::kernel1();
}

namespace ns {
  int glob;
}
extern "C++" {
  int another_global = 5;
  namespace AnotherNS {
    int moar_globals = 5;
   }
}

int use2 ( a_type ab, a_type *abp ) {

  if (ab.constexpr_stat_member) return 2;
  if (ab.const_stat_member) return 1;
  // expected-error@+1 {{SYCL kernel cannot use a non-const static data variable}}
  if (ab.stat_member) return 0;
  // expected-error@+1 {{SYCL kernel cannot use a non-const static data variable}}
  if (abp->stat_member) return 0;
  if (ab.fm()) return 0;
  // expected-error@+1 {{SYCL kernel cannot use a global variable}}
  return another_global ;
  // expected-error@+1 {{SYCL kernel cannot use a global variable}}
  return ns::glob +
  // expected-error@+1 {{SYCL kernel cannot use a global variable}}
    AnotherNS::moar_globals;
}

int addInt(int n, int m) {
    return n+m;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
  a_type ab;
  a_type *p;
  use2(ab, p);
}

int main() {
  a_type ab;
  kernel_single_task<class fake_kernel>([]() { usage(  &addInt ); });
  return 0;
}

