// RUN: %check_clang_tidy %s misc-const-correctness %t -- -- -std=c++17 -fno-delayed-template-parsing

template <typename L, typename R>
struct MyPair {
  L left;
  R right;
  MyPair(const L &ll, const R &rr) : left{ll}, right{rr} {}
};

void f() {
  // FIXME: Decomposition Decls need special treatment, because they require to use 'auto'
  // and the 'const' should only be added if all elements can be const.
  // The issue is similar to multiple declarations in one statement.
  // Simply bail for now.
  auto [np_local0, np_local1] = MyPair<int, int>(42, 42);
  np_local0++;
  np_local1++;
  // CHECK-FIXES-NOT: auto const [np_local0, np_local1]

  auto [np_local2, p_local0] = MyPair<double, double>(42., 42.);
  np_local2++;
  // CHECK-FIXES-NOT: auto const [np_local2, p_local0]

  auto [p_local1, np_local3] = MyPair<double, double>(42., 42.);
  np_local3++;
  // CHECK-FIXES-NOT: auto const [p_local1, np_local3]

  auto [p_local2, p_local3] = MyPair<double, double>(42., 42.);
  // CHECK-FIXES-NOT: auto const [p_local2, p_local3]
}

void g() {
  int p_local0 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const p_local0 = 42;
}

template <typename SomeValue>
struct DoGooder {
  DoGooder(void *accessor, SomeValue value) {
  }
};
struct Bingus {
  static constexpr auto someRandomConstant = 99;
};
template <typename Foo>
struct HardWorker {
  HardWorker() {
    const DoGooder<int> anInstanceOf(nullptr, Foo::someRandomConstant);
  }
};
struct TheContainer {
  HardWorker<Bingus> m_theOtherInstance;
  // CHECK-FIXES-NOT: HardWorker<Bingus> const m_theOtherInstance
};
