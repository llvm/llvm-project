// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -pedantic %s

// Test the c++0x-specific reference initialization rules, e.g., the
// rules for rvalue references.
template<typename T> T prvalue();
template<typename T> T&& xvalue();
template<typename T> T& lvalue();

struct Base { };
struct Derived : Base { };

struct HasArray {
  int array[5];
};

int f(int);

template<typename T>
struct ConvertsTo {
  operator T(); // expected-note 2{{candidate function}}
};

void test_rvalue_refs() {
  // If the initializer expression...
  //   - is an xvalue, class prvalue, array prvalue or function lvalue
  //     and "cv1 T1" is reference-compatible with "cv2 T2", or

  // xvalue case
  Base&& base0 = xvalue<Base>();
  Base&& base1 = xvalue<Derived>();
  int&& int0 = xvalue<int>();

  // class prvalue case
  Base&& base2 = prvalue<Base>();
  Base&& base3 = prvalue<Derived>();

  // array prvalue case
  int (&&array0)[5] = HasArray().array;

  // function lvalue case
  int (&&function0)(int) = f;

  //   - has a class type (i.e., T2 is a class type), where T1 is not
  //     reference-related to T2, and can be implicitly converted to
  //     an xvalue, class prvalue, or function lvalue of type "cv3
  //     T3", where "cv1 T1" is reference-compatible with "cv3 T3",

  // xvalue
  Base&& base4 = ConvertsTo<Base&&>();
  Base&& base5 = ConvertsTo<Derived&&>();
  int && int1 = ConvertsTo<int&&>();

  // class prvalue
  Base&& base6 = ConvertsTo<Base>();
  Base&& base7 = ConvertsTo<Derived>();

  // function lvalue
  int (&&function1)(int) = ConvertsTo<int(&)(int)>();

  // In the second case, if the reference is an rvalue reference and
  // the second standard conversion sequence of the user-defined
  // conversion sequence includes an lvalue-to-rvalue conversion, the
  // program is ill-formed.
  int &&int2 = ConvertsTo<int&>(); // expected-error{{no viable conversion from 'ConvertsTo<int &>' to 'int'}}
  int &&int3 = ConvertsTo<float&>(); // expected-error{{no viable conversion from 'ConvertsTo<float &>' to 'int'}}
}

class NonCopyable {
  NonCopyable(const NonCopyable&);
};

class NonCopyableDerived : public NonCopyable {
  NonCopyableDerived(const NonCopyableDerived&);
};

// Make sure we get direct bindings with no copies.
void test_direct_binding() {
  NonCopyable &&nc0 = prvalue<NonCopyable>();
  NonCopyable &&nc1 = prvalue<NonCopyableDerived>();
  NonCopyable &&nc2 = xvalue<NonCopyable>();
  NonCopyable &&nc3 = xvalue<NonCopyableDerived>();
  const NonCopyable &nc4 = prvalue<NonCopyable>();
  const NonCopyable &nc5 = prvalue<NonCopyableDerived>();
  const NonCopyable &nc6 = xvalue<NonCopyable>();
  const NonCopyable &nc7 = xvalue<NonCopyableDerived>();
  NonCopyable &&nc8 = ConvertsTo<NonCopyable&&>();
  NonCopyable &&nc9 = ConvertsTo<NonCopyableDerived&&>();
  const NonCopyable &nc10 = ConvertsTo<NonCopyable&&>();
  const NonCopyable &nc11 = ConvertsTo<NonCopyableDerived&&>();
}

namespace std_example_1 {
  double d = 2.0; 
  double& rd = d; 
  const double& rcd = d;
  struct A { }; 
  struct B : A { 
    operator int&();
  } b;
  A& ra = b; 
  const A& rca = b; 
  int& ir = B();
}

namespace std_example_2 {
  double& rd2 = 2.0; // expected-error{{non-const lvalue reference to type 'double' cannot bind to a temporary of type 'double'}}
  int i = 2; 
  double& rd3 = i; // expected-error{{non-const lvalue reference to type 'double' cannot bind to a value of unrelated type 'int'}}
  struct A { }; 
  struct B : A { } b; 
  extern B f(); 
  const A& rca = f(); 
  A&& rra = f();
  struct X { 
    operator B();  // expected-note{{candidate function}}
    operator int&(); // expected-note{{candidate function}}
  } x;
  const A& r = x;
  int&& rri = static_cast<int&&>(i);
  B&& rrb = x;
  int&& rri2 = X(); // expected-error{{no viable conversion from 'std_example_2::X' to 'int'}}

  const double& rcd2 = 2;
  double&& rrd = 2;
  const volatile int cvi = 1; 
  const int& r2 = cvi; // expected-error{{binding of reference to type 'const int' to a value of type 'const volatile int' drops qualifiers}}

  double d;
  double&& rrd2 = d; // expected-error{{rvalue reference to type 'double' cannot bind to lvalue of type 'double'}}
  double&& rrd3 = i;
}

namespace argument_passing {
  void base_rvalue_ref(Base&&);
  void int_rvalue_ref(int&&); // expected-note{{candidate function not viable: no known conversion from 'ConvertsTo<int &>' to 'int &&' for 1st argument}} \
  // expected-note{{candidate function not viable: no known conversion from 'ConvertsTo<float &>' to 'int &&' for 1st argument}}

  void array_rvalue_ref(int (&&)[5]);
  void function_rvalue_ref(int (&&)(int));

  void test() {
    base_rvalue_ref(xvalue<Base>());
    base_rvalue_ref(xvalue<Derived>());
    int_rvalue_ref(xvalue<int>());
    
    base_rvalue_ref(prvalue<Base>());
    base_rvalue_ref(prvalue<Derived>());
    
    array_rvalue_ref(HasArray().array);
    
    function_rvalue_ref(f);
    
    base_rvalue_ref(ConvertsTo<Base&&>());
    base_rvalue_ref(ConvertsTo<Derived&&>());
    int_rvalue_ref(ConvertsTo<int&&>());
    
    base_rvalue_ref(ConvertsTo<Base>());
    base_rvalue_ref(ConvertsTo<Derived>());

    function_rvalue_ref(ConvertsTo<int(&)(int)>());
    
    int_rvalue_ref(ConvertsTo<int&>()); // expected-error{{no matching function for call to 'int_rvalue_ref'}}
    int_rvalue_ref(ConvertsTo<float&>()); // expected-error{{no matching function for call to 'int_rvalue_ref'}}
  }

}

namespace pr10644 {
  struct string {
    string(const char* __s);
  };
  class map {
    int& operator[](const string& __k);
  public:
    int& operator[](const string&& __k);
  };
  void foo() {
    static map key_map;
    key_map["line"];
  }
}

namespace PR11003 {
  class Value {
  };
  struct MoveRef {
    operator Value &() const ;
  };
  MoveRef Move(int);
  void growTo() {
    Value x = Move(0);
    Value y(Move(0));
  }
}
