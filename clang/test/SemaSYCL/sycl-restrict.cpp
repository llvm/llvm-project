// RUN: %clang_cc1 -fcxx-exceptions -fsycl-is-device -Wno-return-type -verify -fsyntax-only -x c++ -emit-llvm-only -std=c++17 %s


namespace std {
  class type_info;
  typedef __typeof__(sizeof(int)) size_t;
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
  // These overloads do not allocate.
  void *operator new(std::size_t size) throw() {return 0;}
  void *operator new[](std::size_t size) throw() {return 0;}
  void operator delete(void *){};
  void operator delete[](void *){};
};

bool isa_B(A *a) {

  // expected-error@+1 {{SYCL kernel cannot allocate storage}}
  int *ip = new int;
  int i; int *p3 = new(&i) int; // no error on placement new
  //FIXME call to overloaded new should not get error message
  // expected-error@+1 {{SYCL kernel cannot allocate storage}}
  OverloadedNewDelete *x = new( struct OverloadedNewDelete );
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

  // expected-error@+1 {{SYCL kernel cannot call through a function pointer}}
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

int use2 ( a_type ab ) {

  if (ab.constexpr_stat_member) return 2;
  if (ab.const_stat_member) return 1;
  // expected-error@+1 {{SYCL kernel cannot use a non-const static data variable}}
  if (ab.stat_member) return 0;
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
  use2(ab);
}

int main() {
  a_type ab;
  kernel_single_task<class fake_kernel>([]() { usage(  &addInt ); });
  return 0;
}

