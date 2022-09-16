// RUN: %clang_cc1 -fsyntax-only -verify -xobjective-c++ %s -std=c++20 -fms-extensions -fblocks -fobjc-arc -fobjc-runtime-has-weak -fenable-matrix -Wno-dynamic-exception-spec -Wno-c++17-compat-mangling
// RUN: %clang_cc1 -fsyntax-only -verify -xobjective-c++ %s -std=c++14 -fms-extensions -fblocks -fobjc-arc -fobjc-runtime-has-weak -fenable-matrix -Wno-dynamic-exception-spec -Wno-c++17-compat-mangling

namespace std {
template<typename T> struct initializer_list {
  const T *begin, *end;
  initializer_list();
};
} // namespace std

enum class N {};

using Animal = int;

using AnimalPtr = Animal *;

using Man = Animal;
using Dog = Animal;

using ManPtr = Man *;
using DogPtr = Dog *;

using SocratesPtr = ManPtr;

using ConstMan = const Man;
using ConstDog = const Dog;

using Virus = void;
using SARS = Virus;
using Ebola = Virus;

using Bacteria = float;
using Bacilli = Bacteria;
using Vibrio = Bacteria;

struct Plant;
using Gymnosperm = Plant;
using Angiosperm = Plant;

namespace variable {

auto x1 = Animal();
N t1 = x1; // expected-error {{lvalue of type 'Animal' (aka 'int')}}

auto x2 = AnimalPtr();
N t2 = x2; // expected-error {{lvalue of type 'AnimalPtr' (aka 'int *')}}

auto *x3 = AnimalPtr();
N t3 = x3; // expected-error {{lvalue of type 'Animal *' (aka 'int *')}}

// Each variable deduces separately.
auto x4 = Man(), x5 = Dog();
N t4 = x4; // expected-error {{lvalue of type 'Man' (aka 'int')}}
N t5 = x5; // expected-error {{lvalue of type 'Dog' (aka 'int')}}

auto x6 = { Man(), Dog() };
N t6 = x6; // expected-error {{from 'std::initializer_list<Animal>' (aka 'initializer_list<int>')}}

} // namespace variable

namespace function_basic {

auto f1() { return Animal(); }
auto x1 = f1();
N t1 = x1; // expected-error {{lvalue of type 'Animal' (aka 'int')}}

decltype(auto) f2() { return Animal(); }
auto x2 = f2();
N t2 = x2; // expected-error {{lvalue of type 'Animal' (aka 'int')}}

auto x3 = [a = Animal()] { return a; }();
N t3 = x3; // expected-error {{lvalue of type 'Animal' (aka 'int')}}

} // namespace function_basic

namespace function_multiple_basic {

N t1 = [] { // expected-error {{rvalue of type 'Animal' (aka 'int')}}
  if (true)
    return Man();
  return Dog();
}();

N t2 = []() -> decltype(auto) { // expected-error {{rvalue of type 'Animal' (aka 'int')}}
  if (true)
    return Man();
  return Dog();
}();

N t3 = [] { // expected-error {{rvalue of type 'Animal' (aka 'int')}}
  if (true)
    return Dog();
  auto x = Man();
  return x;
}();

N t4 = [] { // expected-error {{rvalue of type 'int'}}
  if (true)
    return Dog();
  return 1;
}();

N t5 = [] { // expected-error {{rvalue of type 'Virus' (aka 'void')}}
  if (true)
    return Ebola();
  return SARS();
}();

N t6 = [] { // expected-error {{rvalue of type 'void'}}
  if (true)
    return SARS();
  return;
}();

} // namespace function_multiple_basic

#define TEST_AUTO(X, A, B) \
  static_assert(__is_same(A, B), ""); \
  auto X(A a, B b) {       \
    if (0)                 \
      return a;            \
    if (0)                 \
      return b;            \
    return N();            \
  }
#define TEST_DAUTO(X, A, B)     \
  static_assert(__is_same(A, B), ""); \
  decltype(auto) X(A a, B b) {  \
    if (0)                      \
      return static_cast<A>(a); \
    if (0)                      \
      return static_cast<B>(b); \
    return N();                 \
  }

namespace misc {

TEST_AUTO(t1, ManPtr, DogPtr)      // expected-error {{but deduced as 'Animal *' (aka 'int *')}}
TEST_AUTO(t2, ManPtr, int *)       // expected-error {{but deduced as 'int *'}}
TEST_AUTO(t3, SocratesPtr, ManPtr) // expected-error {{but deduced as 'ManPtr' (aka 'int *')}}

TEST_AUTO(t4, _Atomic(Man), _Atomic(Dog)) // expected-error {{but deduced as '_Atomic(Animal)'}}

using block_man = void (^)(Man);
using block_dog = void (^)(Dog);
TEST_AUTO(t5, block_man, block_dog) // expected-error {{but deduced as 'void (^__strong)(Animal)'}}

#if __cplusplus >= 201500
using fp1 = SARS (*)(Man, DogPtr) throw(Vibrio);
using fp2 = Ebola (*)(Dog, ManPtr) throw(Bacilli);
TEST_AUTO(t6, fp1, fp2); // expected-error {{but deduced as 'Virus (*)(Animal, Animal *) throw(Bacteria)' (aka 'void (*)(int, int *) throw(Bacteria)')}}

using fp3 = SARS (*)() throw(Man);
using fp4 = Ebola (*)() throw(Vibrio);
auto t7(fp3 a, fp4 b) {
  if (false)
    return true ? a : b;
  if (false)
    return a;
  return N(); // expected-error {{but deduced as 'SARS (*)() throw(Man, Vibrio)' (aka 'void (*)() throw(Man, Vibrio)')}}
}
#endif

using fp5 = void (*)(const Man);
using fp6 = void (*)(Dog);
TEST_AUTO(t8, fp5, fp6); // expected-error {{but deduced as 'void (*)(Animal)' (aka 'void (*)(int)')}}

using fp7 = void (*)(ConstMan);
using fp8 = void (*)(ConstDog);
TEST_AUTO(t9, fp7, fp8); // expected-error {{but deduced as 'void (*)(const Animal)' (aka 'void (*)(const int)')}}

using fp9 = void (*)(ConstMan);
using fp10 = void (*)(const Dog);
TEST_AUTO(t10, fp9, fp10); // expected-error {{but deduced as 'void (*)(const Animal)' (aka 'void (*)(const int)')}}

using fp11 = void (*)(__strong block_man);
using fp12 = void (*)(__weak block_dog);
TEST_AUTO(t11, fp11, fp12); // expected-error {{but deduced as 'void (*)(void (^)(Animal))'}}

TEST_AUTO(t12, Man Angiosperm::*, Dog Gymnosperm::*) // expected-error {{but deduced as 'Animal Plant::*'}}

TEST_DAUTO(t13, const Man &, const Dog &) // expected-error {{but deduced as 'const Animal &' (aka 'const int &')}}

TEST_DAUTO(t14, Man &&, Dog &&) // expected-error {{but deduced as 'Animal &&' (aka 'int &&')}}

using matrix_man = Man __attribute__((matrix_type(4, 4)));
using matrix_dog = Dog __attribute__((matrix_type(4, 4)));
TEST_AUTO(t15, matrix_man, matrix_dog) // expected-error {{but deduced as 'Animal __attribute__((matrix_type(4, 4)))'}}

using vector_man = Man __attribute__((vector_size(4)));
using vector_dog = Dog __attribute__((vector_size(4)));
TEST_AUTO(t16, vector_man, vector_dog) // expected-error {{but deduced as '__attribute__((__vector_size__(1 * sizeof(Animal)))) Animal' (vector of 1 'Animal' value)}}

using ext_vector_man = Man __attribute__((ext_vector_type(4)));
using ext_vector_dog = Dog __attribute__((ext_vector_type(4)));
TEST_AUTO(t17, ext_vector_man, ext_vector_dog) // expected-error {{but deduced as 'Animal __attribute__((ext_vector_type(4)))' (vector of 4 'Animal' values)}}

using TwoDogs = Dog[2];
using ConstTwoDogsPtr = const TwoDogs*;
using ConstTwoMenPtr = const Man(*)[2];
TEST_AUTO(t18, ConstTwoDogsPtr, ConstTwoMenPtr); // expected-error {{but deduced as 'const Animal (*)[2]' (aka 'const int (*)[2]')}}

} // namespace misc

namespace exception_spec {

void none();
void dyn_none() throw();
void dyn() throw(int);
void ms_any() throw(...);
void __declspec(nothrow) nothrow();
void noexcept_basic() noexcept;
void noexcept_true() noexcept(true);
void noexcept_false() noexcept(false);

#if __cplusplus < 201500
TEST_AUTO(t1, decltype(&noexcept_false), decltype(&noexcept_true)) // expected-error {{but deduced as 'void (*)() noexcept(false)'}}
TEST_AUTO(t2, decltype(&noexcept_basic), decltype(&noexcept_true)) // expected-error {{but deduced as 'void (*)() noexcept(true)'}}
TEST_AUTO(t3, decltype(&none), decltype(&ms_any)) // expected-error {{but deduced as 'void (*)()'}}
TEST_AUTO(t4, decltype(&noexcept_false), decltype(&ms_any)) // expected-error {{but deduced as 'void (*)() throw(...)'}}
TEST_AUTO(t5, decltype(&nothrow), decltype(&noexcept_false)) // expected-error {{but deduced as 'void (*)() noexcept(false)'}}
TEST_AUTO(t6, decltype(&dyn_none), decltype(&nothrow)) // expected-error {{but deduced as 'void (*)() throw()'}}
TEST_AUTO(t7, decltype(&noexcept_true), decltype(&dyn)) // expected-error {{but deduced as 'void (*)() throw(int)'}}
#endif
} // namespace exception_spec

namespace non_deduced {
  void f();
  void g();
  void g(int);
  auto h() {
    if (false) return f;
    return g;
    // expected-error@-1 {{returned value of type '<overloaded function type>'}}
  }
} // namespace non_deduced
