// RUN: rm -f %t.14 %t.2a
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c++14 -DCXX2A=0 -fblocks -Wall -Wno-unused -Werror %s > %t.14 2>&1
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG -std=c++2a -DCXX2A=1 -fblocks -Wall -Wno-unused -Werror %s > %t.2a 2>&1
// RUN: FileCheck --input-file=%t.14 -check-prefixes=CHECK,CXX14 -implicit-check-not=destructor %s
// RUN: FileCheck --input-file=%t.2a -check-prefixes=CHECK,CXX2A -implicit-check-not=destructor %s

int puts(const char *);

struct Foo {
  Foo() = delete;
#if CXX2A
  // Guarantee that the elided examples are actually elided by deleting the
  // copy constructor.
  Foo(const Foo &) = delete;
#else
  // No elision support, so we need a copy constructor.
  Foo(const Foo &);
#endif
  ~Foo();
};

struct TwoFoos {
  Foo foo1, foo2;
  ~TwoFoos();
};

Foo get_foo();

struct Bar {
  Bar();
  Bar(const Bar &);
  ~Bar();
  Bar &operator=(const Bar &);
};

Bar get_bar();

struct TwoBars {
  Bar foo1, foo2;
  ~TwoBars();
};

// Start of tests:

void elided_assign() {
  Foo x = get_foo();
}
// CHECK: void elided_assign()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~Foo() (Implicit destructor)

void nonelided_assign() {
  Bar x = (const Bar &)get_bar();
}
// CHECK: void nonelided_assign()
// CHECK: (CXXConstructExpr{{.*}}, Bar)
// CHECK: ~Bar() (Temporary object destructor)
// CHECK: ~Bar() (Implicit destructor)

void elided_paren_init() {
  Foo x(get_foo());
}
// CHECK: void elided_paren_init()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~Foo() (Implicit destructor)

void nonelided_paren_init() {
  Bar x((const Bar &)get_bar());
}
// CHECK: void nonelided_paren_init()
// CHECK: (CXXConstructExpr{{.*}}, Bar)
// CHECK: ~Bar() (Temporary object destructor)
// CHECK: ~Bar() (Implicit destructor)

void elided_brace_init() {
  Foo x{get_foo()};
}
// CHECK: void elided_brace_init()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~Foo() (Implicit destructor)

void nonelided_brace_init() {
  Bar x{(const Bar &)get_bar()};
}
// CHECK: void nonelided_brace_init()
// CHECK: (CXXConstructExpr{{.*}}, Bar)
// CHECK: ~Bar() (Temporary object destructor)
// CHECK: ~Bar() (Implicit destructor)

void elided_lambda_capture_init() {
  // The copy from get_foo() into the lambda should be elided.  Should call
  // the lambda's destructor, but not ~Foo() separately.
  // (This syntax is C++14 'generalized lambda captures'.)
  auto z = [x=get_foo()]() {};
}
// CHECK: void elided_lambda_capture_init()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~(lambda at {{.*}})() (Temporary object destructor)
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~(lambda at {{.*}})() (Implicit destructor)

void nonelided_lambda_capture_init() {
  // Should call the lambda's destructor as well as ~Bar() for the temporary.
  auto z = [x((const Bar &)get_bar())]() {};
}
// CHECK: void nonelided_lambda_capture_init()
// CHECK: (CXXConstructExpr{{.*}}, Bar)
// CXX14: ~(lambda at {{.*}})() (Temporary object destructor)
// CHECK: ~Bar() (Temporary object destructor)
// CHECK: ~(lambda at {{.*}})() (Implicit destructor)

Foo elided_return_stmt_expr() {
  // Two copies, both elided in C++17.
  return ({ get_foo(); });
}
// CHECK: Foo elided_return_stmt_expr()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~Foo() (Temporary object destructor)
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~Foo() (Temporary object destructor)

void elided_stmt_expr() {
  // One copy, elided in C++17.
  ({ get_foo(); });
}
// CHECK: void elided_stmt_expr()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~Foo() (Temporary object destructor)


void elided_stmt_expr_multiple_stmts() {
  // Make sure that only the value returned out of a statement expression is
  // elided.
  ({ get_bar(); get_foo(); });
}
// CHECK: void elided_stmt_expr_multiple_stmts()
// CHECK: ~Bar() (Temporary object destructor)
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~Foo() (Temporary object destructor)


void unelided_stmt_expr() {
  ({ (const Bar &)get_bar(); });
}
// CHECK: void unelided_stmt_expr()
// CHECK: (CXXConstructExpr{{.*}}, Bar)
// CHECK: ~Bar() (Temporary object destructor)
// CHECK: ~Bar() (Temporary object destructor)

void elided_aggregate_init() {
  TwoFoos x{get_foo(), get_foo()};
}
// CHECK: void elided_aggregate_init()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~Foo() (Temporary object destructor)
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~TwoFoos() (Implicit destructor)

void nonelided_aggregate_init() {
  TwoBars x{(const Bar &)get_bar(), (const Bar &)get_bar()};
}
// CHECK: void nonelided_aggregate_init()
// CHECK: (CXXConstructExpr{{.*}}, Bar)
// CHECK: (CXXConstructExpr{{.*}}, Bar)
// CHECK: ~Bar() (Temporary object destructor)
// CHECK: ~Bar() (Temporary object destructor)
// CHECK: ~TwoBars() (Implicit destructor)

TwoFoos return_aggregate_init() {
  return TwoFoos{get_foo(), get_foo()};
}
// CHECK: TwoFoos return_aggregate_init()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~TwoFoos() (Temporary object destructor)
// CXX14: ~Foo() (Temporary object destructor)
// CXX14: ~Foo() (Temporary object destructor)

void lifetime_extended() {
  const Foo &x = (get_foo(), get_foo());
  puts("one destroyed before, one destroyed after");
}
// CHECK: void lifetime_extended()
// CHECK: ~Foo() (Temporary object destructor)
// CHECK: one destroyed before, one destroyed after
// CHECK: ~Foo() (Implicit destructor)

void not_lifetime_extended() {
  Foo x = (get_foo(), get_foo());
  puts("one destroyed before, one destroyed after");
}
// CHECK: void not_lifetime_extended()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CHECK: ~Foo() (Temporary object destructor)
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: one destroyed before, one destroyed after
// CHECK: ~Foo() (Implicit destructor)

void compound_literal() {
  (void)(Bar[]){{}, {}};
}
// CHECK: void compound_literal()
// CHECK: (CXXConstructExpr, Bar)
// CHECK: (CXXConstructExpr, Bar)
// CHECK: ~Bar[2]() (Temporary object destructor)

Foo elided_return() {
  return get_foo();
}
// CHECK: Foo elided_return()
// CXX14: (CXXConstructExpr{{.*}}, Foo)
// CXX14: ~Foo() (Temporary object destructor)

auto elided_return_lambda() {
  return [x=get_foo()]() {};
}
// CHECK: (lambda at {{.*}}) elided_return_lambda()
// CXX14: (CXXConstructExpr{{.*}}, class (lambda at {{.*}}))
// CXX14: ~(lambda at {{.*}})() (Temporary object destructor)
// CXX14: ~Foo() (Temporary object destructor)

void const_auto_obj() {
  const Bar bar;
}
// CHECK: void const_auto_obj()
// CHECK: .~Bar() (Implicit destructor)

void has_default_arg(Foo foo = get_foo());
void test_default_arg() {
  // FIXME: This emits a destructor but no constructor.  Search CFG.cpp for
  // 'PR13385' for details.
  has_default_arg();
}
// CHECK: void test_default_arg()
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~Foo() (Temporary object destructor)

struct DefaultArgInCtor {
    DefaultArgInCtor(Foo foo = get_foo());
    ~DefaultArgInCtor();
};

void default_ctor_with_default_arg() {
  // FIXME: Default arguments are mishandled in two ways:
  // - The constructor is not emitted at all (not specific to arrays; see fixme
  //   in CFG.cpp that mentions PR13385).
  // - The destructor is emitted once, even though the default argument will be
  //   constructed and destructed once per array element.
  // Ideally, the CFG would expand array constructions into a loop that
  // constructs each array element, allowing default argument
  // constructor/destructor calls to be correctly placed inside the loop.
  DefaultArgInCtor qux[3];
}
// CHECK: void default_ctor_with_default_arg()
// CHECK: CXXConstructExpr, {{.*}}, DefaultArgInCtor[3]
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~Foo() (Temporary object destructor)
// CHECK: .~DefaultArgInCtor[3]() (Implicit destructor)

void new_default_ctor_with_default_arg(long count) {
  // Same problems as above.
  new DefaultArgInCtor[count];
}
// CHECK: void new_default_ctor_with_default_arg(long count)
// CHECK: CXXConstructExpr, {{.*}}, DefaultArgInCtor[]
// CXX14: ~Foo() (Temporary object destructor)
// CHECK: ~Foo() (Temporary object destructor)

#if CXX2A
// Boilerplate needed to test co_return:

namespace std {
template <typename Promise>
struct coroutine_handle {
  static coroutine_handle from_address(void *) noexcept;
};
} // namespace std

struct TestPromise {
  TestPromise initial_suspend();
  TestPromise final_suspend() noexcept;
  bool await_ready() noexcept;
  void await_suspend(const std::coroutine_handle<TestPromise> &) noexcept;
  void await_resume() noexcept;
  Foo return_value(const Bar &);
  Bar get_return_object();
  void unhandled_exception();
};

namespace std {
template <typename Ret, typename... Args>
struct coroutine_traits;
template <>
struct coroutine_traits<Bar> {
  using promise_type = TestPromise;
};
} // namespace std

Bar coreturn() {
  co_return get_bar();
  // This expands to something like:
  //     promise.return_value(get_bar());
  // get_bar() is passed by reference to return_value() and is then destroyed;
  // there is no equivalent of RVO.  TestPromise::return_value also returns a
  // Foo, which should be immediately destroyed.
  // FIXME: The generated CFG completely ignores get_return_object().
}
// CXX2A: Bar coreturn()
// CXX2A: ~Foo() (Temporary object destructor)
// CXX2A: ~Bar() (Temporary object destructor)
#endif
