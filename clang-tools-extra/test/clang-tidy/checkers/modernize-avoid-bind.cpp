// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-avoid-bind %t

namespace std {
inline namespace impl {
template <class Fp, class... Arguments>
class bind_rt {};

template <class Fp, class... Arguments>
bind_rt<Fp, Arguments...> bind(Fp &&, Arguments &&...);
}

template <typename T>
T ref(T &t);
}

namespace boost {
template <class Fp, class... Arguments>
class bind_rt {};

template <class Fp, class... Arguments>
bind_rt<Fp, Arguments...> bind(const Fp &, Arguments...);

template <class T>
struct reference_wrapper {
  explicit reference_wrapper(T &t) {}
};

template <class T>
reference_wrapper<T> const ref(T &t) {
  return reference_wrapper<T>(t);
}

} // namespace boost

namespace C {
int add(int x, int y) { return x + y; }
} // namespace C

struct Foo {
  static int add(int x, int y) { return x + y; }
};

struct D {
  D() = default;
  void operator()(int x, int y) const {}

  void MemberFunction(int x) {}

  static D *create();
};

struct F {
  F(int x) {}
  ~F() {}

  int get() { return 42; }
};

void UseF(F);

struct placeholder {};
placeholder _1;
placeholder _2;

int add(int x, int y) { return x + y; }
int addThree(int x, int y, int z) { return x + y + z; }

// Let's fake a minimal std::function-like facility.
namespace std {
template <typename _Tp>
_Tp declval();

template <typename _Functor, typename... _ArgTypes>
struct __res {
  template <typename... _Args>
  static decltype(declval<_Functor>()(_Args()...)) _S_test(int);

  template <typename...>
  static void _S_test(...);

  using type = decltype(_S_test<_ArgTypes...>(0));
};

template <typename>
struct function;

template <typename... _ArgTypes>
struct function<void(_ArgTypes...)> {
  template <typename _Functor,
            typename = typename __res<_Functor, _ArgTypes...>::type>
  function(_Functor) {}
};
} // namespace std

struct Thing {};
void UseThing(Thing *);

struct Callback {
  Callback();
  Callback(std::function<void()>);
  void Reset(std::function<void()>);
};

int GlobalVariable = 42;

struct TestCaptureByValueStruct {
  int MemberVariable;
  static int StaticMemberVariable;
  F MemberStruct;

  void testCaptureByValue(int Param, F f) {
    int x = 3;
    int y = 4;
    auto AAA = std::bind(add, x, y);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to std::bind [modernize-avoid-bind]
    // CHECK-FIXES: auto AAA = [x, y] { return add(x, y); };

    // When the captured variable is repeated, it should only appear in the capture list once.
    auto BBB = std::bind(add, x, x);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to std::bind [modernize-avoid-bind]
    // CHECK-FIXES: auto BBB = [x] { return add(x, x); };

    int LocalVariable;
    // Global variables shouldn't be captured at all, and members should be captured through this.
    auto CCC = std::bind(add, MemberVariable, GlobalVariable);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to std::bind [modernize-avoid-bind]
    // CHECK-FIXES: auto CCC = [this] { return add(MemberVariable, GlobalVariable); };

    // Static member variables shouldn't be captured, but locals should
    auto DDD = std::bind(add, TestCaptureByValueStruct::StaticMemberVariable, LocalVariable);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to std::bind [modernize-avoid-bind]
    // CHECK-FIXES: auto DDD = [LocalVariable] { return add(TestCaptureByValueStruct::StaticMemberVariable, LocalVariable); };

    auto EEE = std::bind(add, Param, Param);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to std::bind [modernize-avoid-bind]
    // CHECK-FIXES: auto EEE = [Param] { return add(Param, Param); };

    // The signature of boost::bind() is different, and causes
    // CXXBindTemporaryExprs to be created in certain cases.  So let's test
    // those here.
    auto FFF = boost::bind(UseF, f);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to boost::bind [modernize-avoid-bind]
    // CHECK-FIXES: auto FFF = [f] { return UseF(f); };

    auto GGG = boost::bind(UseF, MemberStruct);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to boost::bind [modernize-avoid-bind]
    // CHECK-FIXES: auto GGG = [this] { return UseF(MemberStruct); };
  }
};

void testLiteralParameters() {
  auto AAA = std::bind(add, 2, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind [modernize-avoid-bind]
  // CHECK-FIXES: auto AAA = [] { return add(2, 2); };

  auto BBB = std::bind(addThree, 2, 3, 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind [modernize-avoid-bind]
  // CHECK-FIXES: auto BBB = [] { return addThree(2, 3, 4); };
}

void testCaptureByReference() {
  int x = 2;
  int y = 2;
  auto AAA = std::bind(add, std::ref(x), std::ref(y));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto AAA = [&x, &y] { return add(x, y); };

  auto BBB = std::bind(add, std::ref(x), y);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto BBB = [&x, y] { return add(x, y); };

  auto CCC = std::bind(add, y, std::ref(x));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto CCC = [y, &x] { return add(y, x); };

  // Make sure it works with boost::ref() too which has slightly different
  // semantics.
  auto DDD = boost::bind(add, boost::ref(x), boost::ref(y));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to boost::bind
  // CHECK-FIXES: auto DDD = [&x, &y] { return add(x, y); };

  auto EEE = boost::bind(add, boost::ref(x), y);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to boost::bind
  // CHECK-FIXES: auto EEE = [&x, y] { return add(x, y); };

  auto FFF = boost::bind(add, y, boost::ref(x));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to boost::bind
  // CHECK-FIXES: auto FFF = [y, &x] { return add(y, x); };
}

void testCaptureByInitExpression() {
  int x = 42;
  auto AAA = std::bind(add, x, F(x).get());
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto AAA = [x, capture0 = F(x).get()] { return add(x, capture0); };
}

void testFunctionObjects() {
  D d;
  D *e = nullptr;
  auto AAA = std::bind(d, 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto AAA = [d] { return d(1, 2); }

  auto BBB = std::bind(*e, 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto BBB = [e] { return (*e)(1, 2); }

  auto CCC = std::bind(D{}, 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto CCC = [] { return D{}(1, 2); }

  auto DDD = std::bind(D(), 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto DDD = [] { return D()(1, 2); }

  auto EEE = std::bind(*D::create(), 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto EEE = [Func = *D::create()] { return Func(1, 2); };
}

void testPlaceholders() {
  int x = 2;
  auto AAA = std::bind(add, x, _1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto AAA = [x](auto && PH1) { return add(x, PH1); };

  auto BBB = std::bind(add, _2, _1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto BBB = [](auto && PH1, auto && PH2) { return add(PH2, PH1); };

  // No fix is applied for reused placeholders.
  auto CCC = std::bind(add, _1, _1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto CCC = std::bind(add, _1, _1);

  // When a placeholder is skipped, we always add skipped ones to the lambda as
  // unnamed parameters.
  auto DDD = std::bind(add, _2, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto DDD = [](auto &&, auto && PH2) { return add(PH2, 1); };
}

void testGlobalFunctions() {
  auto AAA = std::bind(C::add, 1, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto AAA = [] { return C::add(1, 1); };

  auto BBB = std::bind(Foo::add, 1, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto BBB = [] { return Foo::add(1, 1); };

  // The & should get removed inside of the lambda body.
  auto CCC = std::bind(&C::add, 1, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto CCC = [] { return C::add(1, 1); };

  auto DDD = std::bind(&Foo::add, 1, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto DDD = [] { return Foo::add(1, 1); };

  auto EEE = std::bind(&add, 1, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto EEE = [] { return add(1, 1); };
}

void testCapturedSubexpressions() {
  int x = 3;
  int y = 3;

  auto AAA = std::bind(add, 1, add(2, 5));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // Results of nested calls are captured by value.
  // CHECK-FIXES: auto AAA = [capture0 = add(2, 5)] { return add(1, capture0); };

  auto BBB = std::bind(add, x, add(y, 5));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // Results of nested calls are captured by value.
  // CHECK-FIXES: auto BBB = [x, capture0 = add(y, 5)] { return add(x, capture0); };
}

struct E {
  void MemberFunction(int x) {}

  void testMemberFunctions() {
    D *d;
    D dd;
    auto AAA = std::bind(&D::MemberFunction, d, 1);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to std::bind
    // CHECK-FIXES: auto AAA = [d] { d->MemberFunction(1); };

    auto BBB = std::bind(&D::MemberFunction, &dd, 1);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to std::bind
    // CHECK-FIXES: auto BBB = [ObjectPtr = &dd] { ObjectPtr->MemberFunction(1); };

    auto CCC = std::bind(&E::MemberFunction, this, 1);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to std::bind
    // CHECK-FIXES: auto CCC = [this] { MemberFunction(1); };

    // Test what happens when the object pointer is itself a placeholder.
    auto DDD = std::bind(&D::MemberFunction, _1, 1);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer a lambda to std::bind
    // CHECK-FIXES: auto DDD = [](auto && PH1) { PH1->MemberFunction(1); };
  }
};

void testStdFunction(Thing *t) {
  Callback cb;
  if (t)
    cb.Reset(std::bind(UseThing, t));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: cb.Reset([t] { return UseThing(t); });
}
