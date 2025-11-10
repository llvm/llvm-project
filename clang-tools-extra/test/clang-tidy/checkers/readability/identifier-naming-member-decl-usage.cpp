// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.MemberCase: CamelCase, \
// RUN:     readability-identifier-naming.ParameterCase: CamelCase, \
// RUN:     readability-identifier-naming.MethodCase: camelBack, \
// RUN:     readability-identifier-naming.AggressiveDependentMemberLookup: true \
// RUN:  }}' -- -fno-delayed-template-parsing

int set_up(int);
int clear(int);

class Foo {
public:
  const int bar_baz; // comment-0
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for member 'bar_baz'
  // CHECK-FIXES: const int BarBaz; // comment-0

  Foo(int Val) : bar_baz(Val) { // comment-1
    // CHECK-FIXES: Foo(int Val) : BarBaz(Val) { // comment-1
    set_up(bar_baz); // comment-2
    // CHECK-FIXES: set_up(BarBaz); // comment-2
  }

  Foo() : Foo(0) {}

  ~Foo() {
    clear(bar_baz); // comment-3
    // CHECK-FIXES: clear(BarBaz); // comment-3
  }

  int getBar() const { return bar_baz; } // comment-4
  // CHECK-FIXES: int getBar() const { return BarBaz; } // comment-4
};

class FooBar : public Foo {
public:
  int getFancyBar() const {
    return this->bar_baz; // comment-5
    // CHECK-FIXES: return this->BarBaz; // comment-5
  }
};

int getBar(const Foo &Foo) {
  return Foo.bar_baz; // comment-6
  // CHECK-FIXES: return Foo.BarBaz; // comment-6
}

int getBar(const FooBar &Foobar) {
  return Foobar.bar_baz; // comment-7
  // CHECK-FIXES: return Foobar.BarBaz; // comment-7
}

int getFancyBar(const FooBar &Foobar) {
  return Foobar.getFancyBar();
}

template <typename Dummy>
class TempTest : public Foo {
public:
  TempTest() = default;
  TempTest(int Val) : Foo(Val) {}
  int getBar() const { return Foo::bar_baz; } // comment-8
  // CHECK-FIXES: int getBar() const { return Foo::BarBaz; } // comment-8
  int getBar2() const { return this->bar_baz; } // comment-9
  // CHECK-FIXES: int getBar2() const { return this->BarBaz; } // comment-9
};

namespace Bug41122 {
namespace std {

// for this example we aren't bothered about how std::vector is treated
template <typename T>   // NOLINT
struct vector {         // NOLINT
  void push_back(bool); // NOLINT
  void pop_back();      // NOLINT
};                      // NOLINT
};                      // namespace std

class Foo {
  std::vector<bool> &stack;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: invalid case style for member 'stack' [readability-identifier-naming]
public:
  Foo(std::vector<bool> &stack)
      // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: invalid case style for parameter 'stack' [readability-identifier-naming]
      // CHECK-FIXES: Foo(std::vector<bool> &Stack)
      : stack(stack) {
    // CHECK-FIXES: : Stack(Stack) {
    stack.push_back(true);
    // CHECK-FIXES: Stack.push_back(true);
  }
  ~Foo() {
    stack.pop_back();
    // CHECK-FIXES: Stack.pop_back();
  }
};
}; // namespace Bug41122

namespace Bug29005 {
class Foo {
public:
  int a_member_of_foo = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for member 'a_member_of_foo'
  // CHECK-FIXES: int AMemberOfFoo = 0;
};

int main() {
  Foo foo;
  return foo.a_member_of_foo;
  // CHECK-FIXES: return foo.AMemberOfFoo;
}
}; // namespace Bug29005

namespace CtorInits {
template <typename T, unsigned N>
class Container {
  T storage[N];
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for member 'storage'
  // CHECK-FIXES: T Storage[N];
  T *pointer = &storage[0];
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for member 'pointer'
  // CHECK-FIXES: T *Pointer = &Storage[0];
public:
  Container() : pointer(&storage[0]) {}
  // CHECK-FIXES: Container() : Pointer(&Storage[0]) {}
};

void foo() {
  Container<int, 5> container;
}
} // namespace CtorInits

namespace resolved_dependance {
template <typename T>
struct A0 {
  int value;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for member 'value'
  A0 &operator=(const A0 &Other) {
    value = Other.value;       // A0
    this->value = Other.value; // A0
    // CHECK-FIXES:      Value = Other.Value;       // A0
    // CHECK-FIXES-NEXT: this->Value = Other.Value; // A0
    return *this;
  }
  void outOfLineReset();
};

template <typename T>
void A0<T>::outOfLineReset() {
  this->value -= value; // A0
  // CHECK-FIXES: this->Value -= Value; // A0
}

template <typename T>
struct A1 {
  int value; // A1
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for member 'value'
  // CHECK-FIXES: int Value; // A1
  int GetValue() const { return value; } // A1
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for method 'GetValue'
  // CHECK-FIXES {{^}}  int getValue() const { return Value; } // A1
  void SetValue(int Value) { this->value = Value; } // A1
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for method 'SetValue'
  // CHECK-FIXES {{^}}  void setValue(int Value) { this->Value = Value; } // A1
  A1 &operator=(const A1 &Other) {
    this->SetValue(Other.GetValue()); // A1
    this->value = Other.value;        // A1
    // CHECK-FIXES:      this->setValue(Other.getValue()); // A1
    // CHECK-FIXES-NEXT: this->Value = Other.Value;        // A1
    return *this;
  }
  void outOfLineReset();
};

template <typename T>
void A1<T>::outOfLineReset() {
  this->value -= value; // A1
  // CHECK-FIXES: this->Value -= Value; // A1
}

template <unsigned T>
struct A2 {
  int value; // A2
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for member 'value'
  // CHECK-FIXES: int Value; // A2
  A2 &operator=(const A2 &Other) {
    value = Other.value;       // A2
    this->value = Other.value; // A2
    // CHECK-FIXES:      Value = Other.Value;       // A2
    // CHECK-FIXES-NEXT: this->Value = Other.Value; // A2
    return *this;
  }
};

// create some instances to check it works when instantiated.
A1<int> AInt{};
A1<int> BInt = (AInt.outOfLineReset(), AInt);
A1<unsigned> AUnsigned{};
A1<unsigned> BUnsigned = AUnsigned;
} // namespace resolved_dependance

namespace unresolved_dependance {
template <typename T>
struct DependentBase {
  int depValue;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for member 'depValue'
  // CHECK-FIXES: int DepValue;
};

template <typename T>
struct Derived : DependentBase<T> {
  Derived &operator=(const Derived &Other) {
    this->depValue = Other.depValue;
    // CHECK-FIXES: this->DepValue = Other.DepValue;
    return *this;
  }
};

} // namespace unresolved_dependance
