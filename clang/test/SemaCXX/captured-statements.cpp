// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -fblocks

void test_nest_lambda() {
  int x;
  int y;
  [&,y]() {
    int z;
    #pragma clang __debug captured
    {
      x = y; // OK
      y = z; // expected-error{{cannot assign to a variable captured by copy in a non-mutable lambda}}
      z = y; // OK
    }
  }();

  int a;
  #pragma clang __debug captured
  {
    int b;
    int c;
    [&,c]() {
      a = b; // OK
      b = c; // OK
      c = a; // expected-error{{cannot assign to a variable captured by copy in a non-mutable lambda}}
    }();
  }
}

class test_obj_capture {
  int a;
  void b();
  static void test() {
    test_obj_capture c;
    #pragma clang __debug captured
    { (void)c.a; }  // OK
    #pragma clang __debug captured
    { c.b(); }      // OK
  }
};

class test_this_capture {
  int a;
  void b();
  void test() {
    #pragma clang __debug captured
    { (void)this; } // OK
    #pragma clang __debug captured
    { (void)a; }    // OK
    #pragma clang __debug captured
    { b(); }        // OK
  }
};

template <typename T>
void template_capture_var() {
  T x; // expected-error{{declaration of reference variable 'x' requires an initializer}}
  #pragma clang _debug captured
  {
    (void)x;
  }
}

template <typename T>
class Val {
  T v;
public:
  void set(const T &v0) {
    #pragma clang __debug captured
    {
      v = v0;
    }
  }
};

void test_capture_var() {
  template_capture_var<int>(); // OK
  template_capture_var<int&>(); // expected-note{{in instantiation of function template specialization 'template_capture_var<int &>' requested here}}

  Val<float> Obj;
  Obj.set(0.0f); // OK
}

template <typename S, typename T>
S template_capture_var(S x, T y) {  // expected-note{{variable 'y' declared const here}}
  #pragma clang _debug captured
  {
    x++;
    y++;  // expected-error{{cannot assign to variable 'y' with const-qualified type 'const int'}}
  }

  return x;
}

// Check if can recover from a template error.
void test_capture_var_error() {
  template_capture_var<int, int>(0, 1); // OK
  template_capture_var<int, const int>(0, 1); // expected-note{{in instantiation of function template specialization 'template_capture_var<int, const int>' requested here}}
  template_capture_var<int, int>(0, 1); // OK
}

template <typename T>
void template_capture_in_lambda() {
  T x, y;
  [=, &y]() {
    #pragma clang __debug captured
    {
      y += x;
    }
  }();
}

void test_lambda() {
  template_capture_in_lambda<int>(); // OK
}

struct Foo {
  void foo() { }
  static void bar() { }
};

template <typename T>
void template_capture_func(T &t) {
  #pragma clang __debug captured
  {
    t.foo();
  }

  #pragma clang __debug captured
  {
    T::bar();
  }
}

void test_template_capture_func() {
  Foo Obj;
  template_capture_func(Obj);
}

template <typename T>
T captured_sum(const T &a, const T &b) {
  T result;

  #pragma clang __debug captured
  {
    result = a + b;
  }

  return result;
}

template <typename T, typename... Args>
T captured_sum(const T &a, const Args&... args) {
  T result;

  #pragma clang __debug captured
  {
    result = a + captured_sum(args...);
  }

  return result;
}

void test_capture_variadic() {
  (void)captured_sum(1, 2, 3); // OK
  (void)captured_sum(1, 2, 3, 4, 5); // OK
}

void test_capture_with_attributes() {
  [[]] // expected-error {{an attribute list cannot appear here}}
  #pragma clang __debug captured
  {
  }
}
