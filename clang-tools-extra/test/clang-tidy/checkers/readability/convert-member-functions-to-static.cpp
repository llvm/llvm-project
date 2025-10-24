// RUN: %check_clang_tidy %s readability-convert-member-functions-to-static %t

class DoNotMakeEmptyStatic {
  void emptyMethod() {}
  void empty_method_out_of_line();
};

void DoNotMakeEmptyStatic::empty_method_out_of_line() {}

class A {
  int field;
  const int const_field;
  static int static_field;

  void no_use() {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'no_use' can be made static
    // CHECK-FIXES: static void no_use() {
    int i = 1;
  }

  int read_field() {
    return field;
  }

  void write_field() {
    field = 1;
  }

  int call_non_const_member() { return read_field(); }

  int call_static_member() {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'call_static_member' can be made static
    // CHECK-FIXES: static int call_static_member() {
    already_static();
    return 0;
  }

  int read_static() {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'read_static' can be made static
    // CHECK-FIXES: static int read_static() {
    return static_field;
  }
  void write_static() {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'write_static' can be made static
    // CHECK-FIXES: static void write_static() {
    static_field = 1;
  }

  void static_nested() {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'static_nested' can be made static
    // CHECK-FIXES: static void static_nested() {
    struct Nested {
      int Foo;
      int getFoo() { return Foo; }
    };
  }

  void write_nested() {
    struct Nested {
      int Foo;
      int getFoo() { return Foo; }
    };
    // Ensure we still detect usages of `this` once we leave the nested class definition.
    field = 1;
  }

  static int already_static() { return static_field; }

  int already_const() const { return field; }

  int already_const_convert_to_static() const {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'already_const_convert_to_static' can be made static
    // CHECK-FIXES: static int already_const_convert_to_static() {
    return static_field;
  }

  static int out_of_line_already_static();

  void out_of_line_call_static();
  // CHECK-FIXES: static void out_of_line_call_static();
  int out_of_line_const_to_static() const;
  // CHECK-FIXES: static int out_of_line_const_to_static() ;
};

int A::out_of_line_already_static() { return 0; }

void A::out_of_line_call_static() {
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: method 'out_of_line_call_static' can be made static
  // CHECK-FIXES: void A::out_of_line_call_static() {
  already_static();
}

int A::out_of_line_const_to_static() const {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'out_of_line_const_to_static' can be made static
  // CHECK-FIXES: int A::out_of_line_const_to_static() {
  return 0;
}

struct KeepVirtual {
  virtual int f() { return 0; }
  virtual int h() const { return 0; }
};

struct KeepVirtualDerived : public KeepVirtual {
  int f() { return 0; }
  int h() const override { return 0; }
};

// Don't add 'static' to special member functions and operators.
struct KeepSpecial {
  KeepSpecial() { int L = 0; }
  ~KeepSpecial() { int L = 0; }
  int operator+() { return 0; }
  operator int() { return 0; }
};

void KeepLambdas() {
  using FT = int (*)();
  auto F = static_cast<FT>([]() { return 0; });
  auto F2 = []() { return 0; };
}

template <class Base>
struct KeepWithTemplateBase : public Base {
  int i;
  // We cannot make these methods static because they might need to override
  // a function from Base.
  int static_f() { return 0; }
};

template <class T>
struct KeepTemplateClass {
  int i;
  // We cannot make these methods static because a specialization
  // might use *this differently.
  int static_f() { return 0; }
};

struct KeepTemplateMethod {
  int i;
  // We cannot make these methods static because a specialization
  // might use *this differently.
  template <class T>
  static int static_f() { return 0; }
};

void instantiate() {
  struct S {};
  KeepWithTemplateBase<S> I1;
  I1.static_f();

  KeepTemplateClass<int> I2;
  I2.static_f();

  KeepTemplateMethod I3;
  I3.static_f<int>();
}

struct Trailing {
  auto g() const -> int {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'g' can be made static
    // CHECK-FIXES: static auto g() -> int {
    return 0;
  }

  void vol() volatile {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'vol' can be made static
    return;
  }

  void ref() const & {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'ref' can be made static
    return;
  }
  void refref() const && {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'refref' can be made static
    return;
  }

  void restr() __restrict {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'restr' can be made static
    return;
  }
};

struct UnevaluatedContext {
  void f() { sizeof(this); }

  void noex() noexcept(noexcept(this));
};

struct LambdaCapturesThis {
  int Field;

  int explicitCapture() {
    return [this]() { return Field; }();
  }

  int implicitCapture() {
    return [&]() { return Field; }();
  }
};

struct NoFixitInMacro {
#define CONST const
  int no_use_macro_const() CONST {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'no_use_macro_const' can be made static
    return 0;
  }

#define ADD_CONST(F) F const
  int ADD_CONST(no_use_macro2()) {
    return 0;
  }

#define FUN no_use_macro()
  int i;
  int FUN {
    return i;
  }

#define T(FunctionName, Keyword) \
  Keyword int FunctionName() { return 0; }
#define EMPTY
  T(A, EMPTY)
  T(B, static)

#define T2(FunctionName) \
  int FunctionName() { return 0; }
  T2(A2)

#define VOLATILE volatile
  void volatileMacro() VOLATILE {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'volatileMacro' can be made static
    return;
  }
};
