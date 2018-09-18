// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -analyzer-config c++-inlining=destructors -std=c++11 -verify -Wno-tautological-undefined-compare %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-multi-file -analyzer-config c++-inlining=destructors -std=c++11 %s -o %t.plist -Wno-tautological-undefined-compare
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/path-notes.cpp.plist

class Foo {
public:
  static void use(int *p) {
    *p = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
    // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  }

  Foo(int *p) {
    use(p);
    // expected-note@-1 {{Passing null pointer value via 1st parameter 'p'}}
    // expected-note@-2 {{Calling 'Foo::use'}}
  }
};

static int *globalPtr;

class Bar {
public:
  ~Bar() {
    Foo f(globalPtr);
    // expected-note@-1 {{Passing null pointer value via 1st parameter 'p'}}
    // expected-note@-2 {{Calling constructor for 'Foo'}}
  }
};

void test() {
  Bar b;
  globalPtr = 0;
  // expected-note@-1 {{Null pointer value stored to 'globalPtr'}}
} // expected-note {{Calling '~Bar'}}


void testAnonymous() {
  class {
  public:
    void method(int *p) {
      *p = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
      // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
    }
  } anonymous;

  anonymous.method(0);
  // expected-note@-1 {{Passing null pointer value via 1st parameter 'p'}}
  // expected-note@-2 {{Calling 'method'}}
}


// A simplified version of std::move.
template <typename T>
T &&move(T &obj) {
  return static_cast<T &&>(obj);
}


namespace defaulted {
  class Dereferencer {
  public:
    Dereferencer() {
      *globalPtr = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'globalPtr')}}
      // expected-note@-1 {{Dereference of null pointer (loaded from variable 'globalPtr')}}
    }

    Dereferencer(const Dereferencer &Other) {
      *globalPtr = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'globalPtr')}}
      // expected-note@-1 {{Dereference of null pointer (loaded from variable 'globalPtr')}}
    }

    Dereferencer(Dereferencer &&Other) {
      *globalPtr = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'globalPtr')}}
      // expected-note@-1 {{Dereference of null pointer (loaded from variable 'globalPtr')}}
    }

    void operator=(const Dereferencer &Other) {
      *globalPtr = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'globalPtr')}}
      // expected-note@-1 {{Dereference of null pointer (loaded from variable 'globalPtr')}}
    }

    void operator=(Dereferencer &&Other) {
      *globalPtr = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'globalPtr')}}
      // expected-note@-1 {{Dereference of null pointer (loaded from variable 'globalPtr')}}
    }

    ~Dereferencer() {
      *globalPtr = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'globalPtr')}}
      // expected-note@-1 {{Dereference of null pointer (loaded from variable 'globalPtr')}}
    }
  };

  class Wrapper {
    Dereferencer d;
  };

  class MovableWrapper {
    Dereferencer d;
  public:
    MovableWrapper() = default;

    MovableWrapper(MovableWrapper &&Other) = default;
    // expected-note@-1 {{Calling move constructor for 'Dereferencer'}}

    MovableWrapper &operator=(MovableWrapper &&Other) = default;
    // expected-note@-1 {{Calling move assignment operator for 'Dereferencer'}}
  };

  void testDefaultConstruction() {
    globalPtr = 0;
    // expected-note@-1 {{Null pointer value stored to 'globalPtr'}}
    Wrapper w;
    // expected-note@-1 {{Calling implicit default constructor for 'Wrapper'}}
    // expected-note@-2 {{Calling default constructor for 'Dereferencer'}}
  }

  void testCopyConstruction(const Wrapper &input) {
    globalPtr = 0;
    // expected-note@-1 {{Null pointer value stored to 'globalPtr'}}
    Wrapper w{input};
    // expected-note@-1 {{Calling implicit copy constructor for 'Wrapper'}}
    // expected-note@-2 {{Calling copy constructor for 'Dereferencer'}}
  }

  void testMoveConstruction(MovableWrapper &&input) {
    globalPtr = 0;
    // expected-note@-1 {{Null pointer value stored to 'globalPtr'}}
    MovableWrapper w{move(input)};
    // expected-note@-1 {{Calling defaulted move constructor for 'MovableWrapper'}}
  }

  void testCopyAssignment(const Wrapper &input) {
    Wrapper w;
    globalPtr = 0;
    // expected-note@-1 {{Null pointer value stored to 'globalPtr'}}
    w = input;
    // expected-note@-1 {{Calling implicit copy assignment operator for 'Wrapper'}}
    // expected-note@-2 {{Calling copy assignment operator for 'Dereferencer'}}
  }

  void testMoveAssignment(MovableWrapper &&input) {
    MovableWrapper w;
    globalPtr = 0;
    // expected-note@-1 {{Null pointer value stored to 'globalPtr'}}
    w = move(input);
    // expected-note@-1 {{Calling defaulted move assignment operator for 'MovableWrapper'}}
  }

  void testDestruction() {
    Wrapper w;
    globalPtr = 0;
    // expected-note@-1 {{Null pointer value stored to 'globalPtr'}}
  }
  // expected-note@-1 {{Calling implicit destructor for 'Wrapper'}}
  // expected-note@-2 {{Calling '~Dereferencer'}}
}

namespace ReturnZeroNote {
  int getZero() {
    return 0;
    // expected-note@-1 {{Returning zero}}
  }

  const int &getZeroByRef() {
    static int zeroVar;
    zeroVar = 0;
    // expected-note@-1 {{The value 0 is assigned to 'zeroVar'}}
    return zeroVar;
    // expected-note@-1 {{Returning zero (reference to 'zeroVar')}}
  }

  void test() {
    int problem = 1 / getZero(); // expected-warning {{Division by zero}}
    // expected-note@-1 {{Calling 'getZero'}}
    // expected-note@-2 {{Returning from 'getZero'}}
    // expected-note@-3 {{Division by zero}}
  }

  void testRef() {
    int problem = 1 / getZeroByRef(); // expected-warning {{Division by zero}}
    // expected-note@-1 {{Calling 'getZeroByRef'}}
    // expected-note@-2 {{Returning from 'getZeroByRef'}}
    // expected-note@-3 {{Division by zero}}
  }
}

int &returnNullReference() {
  int *x = 0;
  // expected-note@-1 {{'x' initialized to a null pointer value}}
  return *x; // expected-warning{{Returning null reference}}
  // expected-note@-1 {{Returning null reference}}
}

struct FooWithInitializer {
	int *ptr;
	FooWithInitializer(int *p) : ptr(p) { // expected-note {{Null pointer value stored to 'f.ptr'}}
		*ptr = 1; // expected-note {{Dereference of null pointer (loaded from field 'ptr')}}
    // expected-warning@-1 {{Dereference of null pointer (loaded from field 'ptr')}}
	}
};

void testPathNoteOnInitializer() {
	int *p = 0; // expected-note {{'p' initialized to a null pointer value}}

	FooWithInitializer f(p); // expected-note {{Passing null pointer value via 1st parameter 'p'}}
  // expected-note@-1 {{Calling constructor for 'FooWithInitializer'}}
}

int testNonPrintableAssignment(int **p) {
  int *&y = *p; // expected-note {{'y' initialized here}}
  y = 0;        // expected-note {{Storing null pointer value}}
  return *y; // expected-warning {{Dereference of null pointer (loaded from variable 'y')}}
             // expected-note@-1 {{Dereference of null pointer (loaded from variable 'y')}}
}

struct Base { int *x; };
struct Derived : public Base {};

void test(Derived d) {
  d.x = 0; //expected-note {{Null pointer value stored to 'd.x'}}
  *d.x = 1; // expected-warning {{Dereference of null pointer (loaded from field 'x')}}
            // expected-note@-1 {{Dereference of null pointer (loaded from field 'x')}}
}

struct Owner {
	struct Wrapper {
		int x;
	};
	Wrapper *arr;
	void testGetDerefExprOnMemberExprWithADot();
};

void Owner::testGetDerefExprOnMemberExprWithADot() {
	if (arr)  // expected-note {{Assuming pointer value is null}}
            // expected-note@-1 {{Taking false branch}}
	  ;
	arr[1].x = 1; //expected-warning {{Dereference of null pointer}}
                //expected-note@-1 {{Dereference of null pointer}}
}

void testGetDerefExprOnMemberExprWithADot() {
  Owner::Wrapper *arr; // expected-note {{'arr' declared without an initial value}}
	arr[2].x = 1; // expected-warning {{Dereference of undefined pointer value}}
                // expected-note@-1 {{Dereference of undefined pointer value}}
}



class A {
public:
  void bar() const {}
};
const A& testDeclRefExprToReferenceInGetDerefExpr(const A *ptr) {
  const A& val = *ptr; //expected-note {{'val' initialized here}}

  // This is not valid C++; if 'ptr' were null, creating 'ref' would be illegal.
  // However, this is not checked at runtime, so this branch is actually
  // possible.
  if (&val == 0) { //expected-note {{Assuming pointer value is null}}
                   // expected-note@-1 {{Taking true branch}}
    val.bar(); // expected-warning {{Called C++ object pointer is null}}
               // expected-note@-1 {{Called C++ object pointer is null}}
  }

  return val;
}

int generateNoteOnDefaultArgument(int one, int two = 0) {
  return one/two; // expected-warning {{Division by zero}}
                  // expected-note@-1 {{Division by zero}}
}
int callGenerateNoteOnDefaultArgument(int o) {
  return generateNoteOnDefaultArgument(o); //expected-note{{Calling 'generateNoteOnDefaultArgument'}}
                                           //expected-note@-1 {{Passing the value 0 via 2nd parameter 'two'}}
}

namespace PR17746 {
  class Inner {
  public:
    ~Inner() {
      *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
      // expected-note@-1 {{Dereference of null pointer}}
    }
  };

  class Outer {
  public:
    Inner *inner;
    ~Outer() {
      delete inner;
      // expected-note@-1 {{Calling '~Inner'}}
    }
  };

  void test(Outer *outer) {
    delete outer;
    // expected-note@-1 {{Calling '~Outer'}}
  }
}

