// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: [ \
// RUN:     {key: readability-identifier-naming.MemberCase, value: CamelCase}, \
// RUN:     {key: readability-identifier-naming.ParameterCase, value: CamelCase} \
// RUN:  ]}' -- -fno-delayed-template-parsing

int set_up(int);
int clear(int);

class Foo {
public:
  const int bar_baz; // comment-0
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for member 'bar_baz'
  // CHECK-FIXES: {{^}}  const int BarBaz; // comment-0

  Foo(int Val) : bar_baz(Val) { // comment-1
    // CHECK-FIXES: {{^}}  Foo(int Val) : BarBaz(Val) { // comment-1
    set_up(bar_baz); // comment-2
    // CHECK-FIXES: {{^}}    set_up(BarBaz); // comment-2
  }

  Foo() : Foo(0) {}

  ~Foo() {
    clear(bar_baz); // comment-3
    // CHECK-FIXES: {{^}}    clear(BarBaz); // comment-3
  }

  int getBar() const { return bar_baz; } // comment-4
  // CHECK-FIXES: {{^}}  int getBar() const { return BarBaz; } // comment-4
};

class FooBar : public Foo {
public:
  int getFancyBar() const {
    return this->bar_baz; // comment-5
    // CHECK-FIXES: {{^}}    return this->BarBaz; // comment-5
  }
};

int getBar(const Foo &Foo) {
  return Foo.bar_baz; // comment-6
  // CHECK-FIXES: {{^}}  return Foo.BarBaz; // comment-6
}

int getBar(const FooBar &Foobar) {
  return Foobar.bar_baz; // comment-7
  // CHECK-FIXES: {{^}}  return Foobar.BarBaz; // comment-7
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
  // CHECK-FIXES: {{^}}  int getBar() const { return Foo::BarBaz; } // comment-8
  int getBar2() const { return this->bar_baz; } // comment-9
  // CHECK-FIXES: {{^}}  int getBar2() const { return this->BarBaz; } // comment-9
};

TempTest<int> x; //force an instantiation

int blah() {
  return x.getBar2(); // gotta have a reference to the getBar2 so that the
                      // compiler will generate the function and resolve
                      // the DependantScopeMemberExpr
}

namespace Bug41122 {
namespace std {

// for this example we aren't bothered about how std::vector is treated
template <typename T> //NOLINT
class vector { //NOLINT
public:
  void push_back(bool); //NOLINT
  void pop_back(); //NOLINT
}; //NOLINT
}; // namespace std

class Foo { 
  std::vector<bool> &stack;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: invalid case style for member 'stack' [readability-identifier-naming]
public:
  Foo(std::vector<bool> &stack)
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: invalid case style for parameter 'stack' [readability-identifier-naming]
      // CHECK-FIXES: {{^}}  Foo(std::vector<bool> &Stack)
      : stack(stack) {
    // CHECK-FIXES: {{^}}      : Stack(Stack) {
    stack.push_back(true);
    // CHECK-FIXES: {{^}}    Stack.push_back(true);
  }
  ~Foo() {
    stack.pop_back();
    // CHECK-FIXES: {{^}}    Stack.pop_back();
  }
};
}; // namespace Bug41122

namespace Bug29005 {
class Foo {
public:
  int a_member_of_foo = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for member 'a_member_of_foo'
  // CHECK-FIXES: {{^}}  int AMemberOfFoo = 0;
};

int main() {
  Foo foo;
  return foo.a_member_of_foo;
  // CHECK-FIXES: {{^}}  return foo.AMemberOfFoo;
}
}; // namespace Bug29005

namespace CtorInits {
template <typename T, unsigned N>
class Container {
  T storage[N];
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for member 'storage'
  // CHECK-FIXES: {{^}}  T Storage[N];
  T *pointer = &storage[0];
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for member 'pointer'
  // CHECK-FIXES: {{^}}  T *Pointer = &Storage[0];
public:
  Container() : pointer(&storage[0]) {}
  // CHECK-FIXES: {{^}}  Container() : Pointer(&Storage[0]) {}
};

void foo() {
  Container<int, 5> container;
}
}; // namespace CtorInits
