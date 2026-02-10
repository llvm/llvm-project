// RUN: %clang_cc1 -fcxx-exceptions -std=c++26 -fexperimental-new-constant-interpreter -verify %s

namespace std {
  class exception {
  public:
    constexpr exception() noexcept {};
    // constexpr exception(const exception&) noexcept;
    // constexpr exception& operator=(const exception&) noexcept;
    constexpr virtual ~exception() {};
    // constexpr virtual const char* what() const noexcept;
  };

  template <typename T> struct remove_reference { using type = T; };
  template <typename T> struct remove_reference<T &> { using type = T; };
  template <typename T> struct remove_reference<T &&> { using type = T; };
  template <typename T>
  constexpr typename std::remove_reference<T>::type&& move(T &&t) noexcept {
    return static_cast<typename std::remove_reference<T>::type &&>(t);
  }
};


class Bad : std::exception {};

namespace Simple {
  constexpr int a() {
    try {
    } catch(int e){
      return 12;
    }
    return -2;
  }
  static_assert(a() == -2);

  constexpr int b() {
    try {
      throw 12;
    } catch(int e){
      return 12;
    }
    return -2;
  }
  static_assert(b() == 12);

  constexpr int c() {
    int m = 12;
    try {
      throw 12;
    } catch(int e){
      m = 140;
    }
    return m;
  }
  static_assert(c() == 140);

  constexpr int d() {
    int m = 12;
    try {
      throw 12;
    } catch(int e){
      m = 140;
    } catch (float f) {
      m = 15;
    }
    return m;
  }
  static_assert(d() == 140);

  constexpr int e() {
    int m = 12;
    try {
      throw 12;
    } catch(int e){
      m = e + 2;
    }
    return m;
  }
  static_assert(e() == 14);

  constexpr int f() {
    int m = 12;
    try {
      throw 12;
    } catch(...){
      m = 100;
    }
    return m;
  }
  static_assert(f() == 100);

  constexpr int g() {
    int m = 12;
    try {
      throw Bad();
    } catch(Bad &B){
      m = 100;
    }
    return m;
  }
  static_assert(g() == 100);

  constexpr int h() {
    int m = 12;
    try {
      throw ++m;
    } catch(...){
    }
    return m;
  }
  static_assert(h() == 13);

  constexpr int i(bool b) {
    try {
      if (b)
        throw 12;
      else
        throw 14.0f;
    } catch (int) {
      return 100;
    } catch (float) {
      return 200;
    }
    return 0;
  }
  static_assert(i(true) == 100);
  static_assert(i(false) == 200);
}

namespace Uncaught {

  constexpr int a() {
    throw 12; // expected-note {{uncaught exception of type 'int': '12'}}
    return 0;
  }
  static_assert(a() == 13); // expected-error {{not an integral constant expression}}
}

namespace NoFrame {
  static_assert((1, throw 2, 3) == 1); // expected-error {{not an integral constant expression}} \
                                       // expected-note {{uncaught exception of type 'int': '2'}} \
                                       // expected-warning {{left operand of comma operator has no effect}}

}

namespace CleanupAfterThrowingCall {
  constexpr int a() {
    throw 12;
    return -12;
  }
  constexpr int test() {
    try {
      a();
    } catch (int i) {
      return 26;
    }

    return 120;
  }
  static_assert(test() == 26);

  constexpr int b2() {
    throw 1.0;
    return 1;
  }
  constexpr int a2() {
    b2();
    throw 12;
    return -12;
  }
  constexpr int test2() {

    try {
      a2();
    } catch (int i) {
      return 26;
    } catch (double d ){
      return (int)(d * 2);
    }

    return 120;
  }
  static_assert(test2() == 2);
}

namespace Dtors {
  class Inc {
    public:
    int &m;
    constexpr Inc(int &m) : m(m) {}
    constexpr ~Inc() { ++m; }
  };

  constexpr int test1() {
    int m = 10;
    Inc _(m);

    try {
      throw 12;
    } catch (int) {
      return m;
    }
  }
  static_assert(test1() == 10);

  constexpr int test2() {
    int m = 10;
    try {
      Inc _(m);
      throw 12;
    } catch (int) {
    }
    return m;
  }
  static_assert(test2() == 11);

  struct checker {
    int & counter;
    constexpr ~checker() {
      ++counter;
    }
  };

  constexpr int test() {
    int counter = 0;
    {
      try {
        auto c1 = checker{counter};
        throw 42;
      } catch (...) {
        return counter * 7;
      }
    }
    return counter * 3;
  }

  constexpr int destruction_counter = test();
  static_assert(destruction_counter == 7);
}

namespace CatchArray {
  template <typename T> consteval T test(T head, auto... tail) {
    const T array[] = {head, tail...};
    try {
      throw array;
    } catch (const T (&arr)[5]) {
      return -2;
    } catch (const T * ptr) {
      return *ptr;
    } catch (...) {
      return -1;
    }
  }

  constexpr auto r0 = test(1,2,3,4,5,6);
  static_assert(r0 == 1);

  constexpr auto r1 = test(1,2,3,4,5);
  static_assert(r1 == 1);

  constexpr auto r2 = test(1,2,3,4);
  static_assert(r2 == 1);

  constexpr auto r3 = test(7,1,2,3,4,5,6);
  static_assert(r3 == 7);

  constexpr auto r4 = test(8,1,2,3,4,5);
  static_assert(r4 == 8);

  constexpr auto r5 = test(9,1,2,3,4);
  static_assert(r5 == 9);
}

namespace CatchVoidPtr {
  consteval int test() {
    int p = 3;
    try {
      throw &p;
    } catch (void * ptr) {
      return *static_cast<int *>(ptr);
    }
  }
  static_assert(test() == 3);

  constexpr void t() {
    int p = 3; // expected-note {{declared here}}
    throw &p;
  }
  consteval int test2() {
    try {
      t();
    } catch (void * ptr) {
      return *static_cast<int *>(ptr); // expected-note {{read of object outside its lifetime is not allowed in a constant expression}}
    }
  }
  static_assert(test2() == 3); // expected-error {{not an integral constant expression}} \
                               // expected-note {{in call to}}
}

namespace Nullptr {
  consteval int test_nullptr() {
    try {
      throw nullptr;
    } catch (const int * ex) {
      return true;
    } catch (...) {
      return false;
    }
  }
  static_assert(test_nullptr());

  consteval int test_zero() {
    try {
      throw 0;
    } catch (const int * ex) {
      return false;
    } catch (...) {
      return true;
    }
  }
  static_assert(test_zero());
}

namespace CatchAll {
  template <typename T, typename... Args> consteval int test(Args && ... args) {
    try {
      throw T{args...};
    } catch (unsigned v) {
      return static_cast<int>(v) * 2;
    } catch (int v) {
      return v * 3;
    } catch (bool v) {
      return static_cast<int>(v) * 5;
    } catch (...) {
      return -1;
    }
    return 0;
  }

  static_assert(test<unsigned>(42u) == 84);
  static_assert(test<int>(13) == 39);
  static_assert(test<bool>(true) == 5);
  static_assert(test<long>(42) == -1);
}

namespace Copy {
  class Child {};
  constexpr int test() {

    try {
      throw Child{};
    } catch (Child C) {
      return 20;
    }
    return 30;
  }
  static_assert(test() == 20);

  constexpr int a(){
      throw Child{};
  }
  constexpr int test2() {

    try {
      a();
    } catch (Child C) {
      return 20;
    }
    return 30;
  }
  static_assert(test2() == 20);
}

namespace Inheritance {
  class Parent2 {
  public:
    int F = 5;
    constexpr int getFive() { return F; }
  };
  class Parent : public Parent2{
  };
  class Child : public Parent {};

  constexpr int foo() {
    try {
      throw Child{};
    } catch (Parent2 P) {
      return P.getFive() + 9;
    }
    return 0;
  }
  static_assert(foo() == 14);

  constexpr int foo2() {
    Child C{};
    try {
      throw &C;
    } catch (Parent2 *P) {
      return P->getFive() + 12;
    }
    return 0;
  }
  static_assert(foo2() == 17);

  constexpr int a() {
    throw Child{};
  };
  constexpr int foo3() {
    try {
      a();
    } catch (Parent2 P) {
      return P.getFive();
    }
    return 0;
  }
  static_assert(foo3() == 5);

  constexpr int b(Child *C) {
    throw C;
  };

  constexpr int foo4() {
    Child C{};
    try {
      b(&C);
    } catch (Parent *P) {
      return P->getFive();
    }
    return 0;
  }
  static_assert(foo4() == 5);
}

namespace Pointer {
  static constexpr auto via_const_catch = 2;
  static constexpr auto via_childs_get_value = 3;
  static constexpr auto via_special_child_catch = 5;

  struct parent {
    int value;
    explicit constexpr parent(int v) noexcept: value{v} { }
    constexpr virtual int get_value() const noexcept {
      return value;
    }
    constexpr virtual ~parent() = default;
  };

  struct modifying_child: parent {
    explicit constexpr modifying_child(int v) noexcept: parent{v} { }
    constexpr int get_value() const noexcept override {
      return value * via_childs_get_value;
    }
  };

  struct ordinary_child: parent {
    explicit constexpr ordinary_child(int v) noexcept: parent{v} { }
  };

  struct special_child: parent {
    explicit constexpr special_child(int v) noexcept: parent{v} { }
  };

  consteval int test(void (*fnc)()) {
    int result = 0;
    try {
      fnc();
    } catch (special_child * sch) {
      result = sch->get_value() * via_special_child_catch;
      delete sch;
    } catch (const special_child * sch) {
      result = sch->get_value() * via_special_child_catch * via_const_catch;
      delete sch;
    } catch (parent * exc) {
      result = exc->get_value();
      delete exc;
    } catch (const parent * exc) {
      result = exc->get_value() * via_const_catch;
      delete exc;
    }
    return result;
  }

  constexpr auto r1 = test([] { throw new parent{1}; });
  static_assert(r1 == 1);

  constexpr auto r2 = test([] { throw new modifying_child{3}; });
  static_assert(r2 == 3 * via_childs_get_value);

  constexpr auto r3 = test([] { throw new ordinary_child{5}; });
  static_assert(r3 == 5);

  constexpr auto r4 = test([] { throw new special_child{17}; });
  static_assert(r4 == 17 * via_special_child_catch);
}

namespace References1 {
  class Parent {
  public:
    int F = 10;
    constexpr int getTen() { return F; }
  };

  class Child : public Parent {
    public:
    int F = 5;
    constexpr int getFive() { return F; }

  };

  constexpr int foo() {
    try {
      throw Child{};
    } catch (Child &C) {
      return C.getFive();
    }
    return 0;
  }
  static_assert(foo() == 5);

  constexpr int nested() {
    throw Child{};
    return 1;
  };
  constexpr int foo2() {
    try {
      nested();
    } catch (Child &C) {
      return C.getFive();
    }
    return 0;
  }
  static_assert(foo2() == 5);

  constexpr int foo3() {
    try {
      throw Child{};
    } catch (Parent &P) {
      return P.getTen();
    }
    return 0;
  }
  static_assert(foo3() == 10);

  constexpr int foo4() {
    try {
      nested();
    } catch (Parent &P) {
      return P.getTen();
    }
    return 0;
  }
  static_assert(foo4() == 10);

  constexpr int foo5() {
    try {
      throw 13;
    } catch (const int &a) {
      return 25;
    }
    return -1;
  }
  static_assert(foo5() == 25);

  consteval bool reference_test(const int & ref) {
      try {
          throw ref;
      } catch (const int & exc_ref) {
          if (exc_ref != ref) {
              return 3;
          } else if (&exc_ref != &ref) {
              return 2;
          }
          return 1;
      }
  }
  static_assert(reference_test(10) == 1);

  consteval bool copy_test(const int & ref) {
      try {
          throw ref;
      } catch (const int exc_ref) {
          if (exc_ref != ref) {
              return 3;
          } else if (&exc_ref == &ref) {
              return 2;
          }
          return 1;
      }
  }
  static_assert(copy_test(10) == 1);

  consteval bool conversion_test(const int & ref) {
      try {
          throw ref;
      } catch (const long exc_ref) {
          if (exc_ref != ref) {
              return 3;
          } else if (static_cast<const void *>(&exc_ref) == static_cast<const void *>(&ref)) {
              return 2;
          }
          return 4;
      } catch (int) {
          return 1;
      }
  }
  static_assert(conversion_test(10) == 1);




}

namespace References2 {
  static constexpr auto via_const_catch = 2;
  static constexpr auto via_childs_get_value = 3;
  static constexpr auto via_special_child_catch = 5;

  struct parent {
    int value;
    explicit constexpr parent(int v) noexcept: value{v} { }
    constexpr virtual int get_value() const noexcept {
      return value;
    }
    constexpr virtual ~parent() = default;
  };

  struct modifying_child: parent {
    explicit constexpr modifying_child(int v) noexcept: parent{v} { }
    constexpr int get_value() const noexcept override {
      return value * via_childs_get_value;
    }
  };

  struct ordinary_child: parent {
    explicit constexpr ordinary_child(int v) noexcept: parent{v} { }
  };

  struct special_child: parent {
    explicit constexpr special_child(int v) noexcept: parent{v} { }
  };

  consteval int test(void (*fnc)()) {
    int result = 0;
    try {
      fnc();
    } catch (special_child & sch) {
      result = sch.get_value() * via_special_child_catch;
    } catch (const special_child & sch) {
      result = sch.get_value() * via_special_child_catch * via_const_catch;
    } catch (parent & exc) {
      result = exc.get_value();
    } catch (const parent & exc) {
      result = exc.get_value() * via_const_catch;
    }
    return result;
  }

  constexpr auto r1 = test([] { throw parent{1}; });
  static_assert(r1 == 1);

  constexpr auto r2 = test([] { throw modifying_child{3}; });
  static_assert(r2 == 3 * via_childs_get_value);

  constexpr auto r3 = test([] { throw ordinary_child{5}; });
  static_assert(r3 == 5);

  constexpr auto r4 = test([] { throw special_child{17}; });
  static_assert(r4 == 17 * via_special_child_catch); 
}

namespace Comma {
  constexpr int catch_it() {
    try {
      return (1, throw 2, 3); // expected-warning {{left operand of comma operator has no effect}}
    } catch (int v) {
      return v;
    }
  }
  static_assert(catch_it() == 2);
}

namespace Lifetime {
  [[noreturn]] constexpr auto create(int v) -> int {
    throw v; // expected-note {{uncaught exception of type 'int': '42'}} \
             // expected-note {{uncaught exception of type 'int': '56'}} \
             // expected-note {{uncaught exception of type 'int': '32'}} \
             // expected-note {{uncaught exception of type 'int': '4'}} \
             // expected-note {{uncaught exception of type 'int': '1'}}
  }

  constexpr int convert(int x) {
    return x;
  }

  constexpr auto value1 = convert(create(42)); // expected-error {{must be initialized by a constant expression}}


  struct wrapper {
    int value;
  };

  constexpr auto value2 = wrapper{create(56)}; // expected-error {{must be initialized by a constant expression}}
  constexpr auto value3 = wrapper(create(32)); // expected-error {{must be initialized by a constant expression}}
  constexpr auto value4 = (1,2,3,create(4)); // expected-error {{must be initialized by a constant expression}}

  constexpr int fnc() {
    const auto v = create(1);
    return v;
  }

  constexpr auto value5 = fnc(); // expected-error {{must be initialized by a constant expression}}
}

namespace TheWorst {
  constexpr int zomg() try {
    throw 12;
  } catch (int i) {
    return 13;
  }

  constexpr int c() {
    return zomg();
  }
  static_assert(c() == 13);


  struct hanaxception {
    int v;
  };

  struct checker {
    int value;
    constexpr checker(int v) try : value{v} {
      if (v > 10) {
        throw hanaxception{v};
      }
    } catch (const hanaxception h) {
    }
  };

  constexpr int test() {
    auto c = checker{11};
    return 42;
  }
  constexpr int constructor_test = test();
  static_assert(constructor_test == 42);
}

namespace Destructors {
  struct F {
    constexpr ~F() noexcept(false){
      throw 42; // expected-note {{uncaught exception of type 'int': '42'}}
    }
  };
  constexpr int test() {
    try {
      F f;
      return 1337;
    } catch (int i) {
      return i;
    }

    return 12;
  }
  static_assert(test() == 42);

  constexpr int test2() {
    F f;
    return 1337;
  }
  static_assert(test2() == 42); // expected-error {{not an integral constant expression}}

}

namespace Noexcept {
  constexpr void throw_exception_here() noexcept(false) {
    throw 42;
  }

  constexpr int test() noexcept {
    throw_exception_here(); // expected-note {{uncaught exception in noexcept function}}
    return 42;
  }
  constexpr int value = test(); // expected-error {{must be initialized by a constant expression}} \
                                // expected-note {{in call to}}
}

namespace Move {
  struct foo {
    int value;
    constexpr foo(int v): value{v} {}
    constexpr foo(foo && other): value{other.value + 1} {}
    constexpr int get() const {
        return value;
    }
  };

  consteval int testMove() {
      try {
          throw std::move(foo{1});
      } catch (const foo & f) {
          return f.get();
      }
      return 8;
  }
  static_assert(testMove() == 2);
}

#if 0
namespace UncaughtExceptions {
  constexpr int foo() {
    try {
      throw 42;
    } catch (...) {
      return __builtin_uncaught_exceptions();
    }
    return -1;
  }
  static_assert(foo() == 0);

  struct F{
    constexpr ~F() {
      if (__builtin_uncaught_exceptions() != 0) {
        __builtin_abort(); // expected-note {{subexpression not valid in a constant expression}}
      }
    }
  };
  constexpr int foo2() {

    try {
      F f; // expected-note {{in call to}}
      throw 42;
    } catch (...) {
      return 0;
    }

    return -1;
  }
  static_assert(foo2() == 0); // expected-error {{not an integral constant expression}} \
                              // expected-note {{in call to}}

  constexpr int foo3() {
    try {
      throw 42;
    } catch (int) {
      return __builtin_uncaught_exceptions();
    }
    return -1;
  }
  static_assert(foo3() == 0);
}
#endif

namespace Uncaught2 {
  class E : std::exception {
    constexpr const char *what() const noexcept {
      return "SOME EXCEPTION WHADDAYAKNOW";
    }
  };
  constexpr int foo() {
    // FIXME: Call what() instead(?)
    throw E(); // expected-note {{uncaught exception of type 'E': '&E()'}}
    return 1;
  }
  static_assert(foo() == 1); // expected-error {{static assertion expression is not an integral constant expression}}


}

namespace UnusualTypes {
  constexpr int foo1() {
    throw (int[]){1,2,3}; // expected-note {{uncaught exception of type 'int *': '&(int[3]){1, 2, 3}[0]'}}
    return 1;
  }
  static_assert(foo1() == 1); // expected-error {{not an integral constant expression}}


  constexpr int foo2() {
    throw 1i; // expected-note {{uncaught exception of type '_Complex int': '&1i'}}
    return 1;
  }
  static_assert(foo2() == 1); // expected-error {{not an integral constant expression}}
}

namespace VirtCallOnException {
  struct A {
    virtual constexpr int getNumber()const  { return 10; }
  };
  struct B : A {};
  struct C : B {
    constexpr int getNumber() const override {
      return 100;
    }
  };

  constexpr int foo() {
    try {
      throw C{};
    } catch (A& a) {
      return a.getNumber();
    }
    return 1;
  }
  static_assert(foo() == 100);

}

namespace VirtCall {
  struct A {
    virtual constexpr int getNumber()const  { return 10; }
  };
  struct B : A {};
  struct C : B {
    constexpr int getNumber() const override {
      return 100;
    }
  };

  constexpr void nested() {
    throw C{};
  }

  struct F {
    constexpr virtual int foo() {
      throw C{};
      return 0;
    }
  };

  constexpr int foo() {
    try {
      F f;
      f.foo();
    } catch (A& a) {
      return a.getNumber();
    }
    return 1;
  }
  static_assert(foo() == 100);
}

namespace Variadic {
  constexpr void variadic(...) {
    throw 123;
  }
  constexpr int foo() {
    try {
      variadic(1,2,3,4);
    } catch (int a) {
      return a;
    }
    return 1;
  }
  static_assert(foo() == 123);
}

namespace StmtExpr {
  constexpr int foo() {
    try {
    return  ({
       int a = 123;
       throw 100;
       12;
       })
      == 12;
    } catch (int) {
      return 200;
    }

    return -1;
  }
  static_assert(foo() == 200);
}

namespace TryBody {
  constexpr int t() {
    throw 100;
  };

  constexpr int test() try {
    t();
    return 1;
  } catch (...) {

    return 20;
  }
  static_assert(test() == 20);
}

namespace ReThrow {
  constexpr int t() {
    throw 100;
  };

  constexpr int test() {
    try {
      t();
      return 1;
    } catch (...) {
      throw;
      return -4;
    }
    return -10;
  }
  constexpr int test2() {
    try {
      test();
    } catch (int a) {
      return a;
    }
    return -1;
  }
  static_assert(test2() == 100);


  constexpr int test3() {
    throw; // expected-note {{throw with no caught exception active}}
  }
  static_assert(test3() == 100); // expected-error {{static assertion expression is not an integral constant expression}} \
                                 // expected-note {{in call to}}

  constexpr int test4() {
    try {
      throw 100; // expected-note {{uncaught exception of type 'int': '100'}}
    } catch(int) {
      throw;
    }
  }
  static_assert(test4() == 100); // expected-error {{static assertion expression is not an integral constant expression}}
}

namespace PointerInThrownValue {
  /// Used to crash because of lifetime issues between the Pointer
  /// saved in ThrownValue and the InterpState.
  struct P    {};
  struct C : P{
    constexpr C() {}
  };
  constexpr int foo() {

      try {
      } catch (int e) {
      }

    try {
      auto thrower = []() { throw C(); };
      thrower();
    } catch (const P&) {
      return 100;
    }

    return -1;
  }
  static_assert(foo() == 100);
}

namespace CatchAfterCallWithExceptionAlreadySet {
  struct P {};
  struct C : P{
    constexpr C() {}
  };
  constexpr int foo() {

      try {
        throw 42;
      } catch (int e) {
      }
    try {
      auto thrower = []() { throw C(); };
      thrower();
    } catch (const P&) {
      return 100;
    }

    return -1;
  }
}
