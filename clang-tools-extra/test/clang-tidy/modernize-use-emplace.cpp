// RUN: %check_clang_tidy %s modernize-use-emplace %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: modernize-use-emplace.ContainersWithPushBack, \
// RUN:               value: '::std::vector; ::std::list; ::std::deque; llvm::LikeASmallVector'}, \
// RUN:              {key: modernize-use-emplace.TupleTypes, \
// RUN:               value: '::std::pair; std::tuple; ::test::Single'}, \
// RUN:              {key: modernize-use-emplace.TupleMakeFunctions, \
// RUN:               value: '::std::make_pair; ::std::make_tuple; ::test::MakeSingle'}] \
// RUN:             }" -- -std=c++11

namespace std {
template <typename>
class initializer_list
{
public:
  initializer_list() noexcept {}
};

template <typename T>
class vector {
public:
  vector() = default;
  vector(initializer_list<T>) {}

  void push_back(const T &) {}
  void push_back(T &&) {}

  template <typename... Args>
  void emplace_back(Args &&... args){};
  ~vector();
};
template <typename T>
class list {
public:
  void push_back(const T &) {}
  void push_back(T &&) {}

  template <typename... Args>
  void emplace_back(Args &&... args){};
  ~list();
};

template <typename T>
class deque {
public:
  void push_back(const T &) {}
  void push_back(T &&) {}

  template <typename... Args>
  void emplace_back(Args &&... args){};
  ~deque();
};

template <typename T> struct remove_reference { using type = T; };
template <typename T> struct remove_reference<T &> { using type = T; };
template <typename T> struct remove_reference<T &&> { using type = T; };

template <typename T1, typename T2> class pair {
public:
  pair() = default;
  pair(const pair &) = default;
  pair(pair &&) = default;

  pair(const T1 &, const T2 &) {}
  pair(T1 &&, T2 &&) {}

  template <typename U1, typename U2> pair(const pair<U1, U2> &){};
  template <typename U1, typename U2> pair(pair<U1, U2> &&){};
};

template <typename T1, typename T2>
pair<typename remove_reference<T1>::type, typename remove_reference<T2>::type>
make_pair(T1 &&, T2 &&) {
  return {};
};

template <typename... Ts> class tuple {
public:
  tuple() = default;
  tuple(const tuple &) = default;
  tuple(tuple &&) = default;

  tuple(const Ts &...) {}
  tuple(Ts &&...) {}

  template <typename... Us> tuple(const tuple<Us...> &){};
  template <typename... Us> tuple(tuple<Us...> &&) {}

  template <typename U1, typename U2> tuple(const pair<U1, U2> &) {
    static_assert(sizeof...(Ts) == 2, "Wrong tuple size");
  };
  template <typename U1, typename U2> tuple(pair<U1, U2> &&) {
    static_assert(sizeof...(Ts) == 2, "Wrong tuple size");
  };
};

template <typename... Ts>
tuple<typename remove_reference<Ts>::type...> make_tuple(Ts &&...) {
  return {};
}

template <typename T>
class unique_ptr {
public:
  explicit unique_ptr(T *) {}
  ~unique_ptr();
};
} // namespace std

namespace llvm {
template <typename T>
class LikeASmallVector {
public:
  void push_back(const T &) {}
  void push_back(T &&) {}

  template <typename... Args>
  void emplace_back(Args &&... args){};
};

} // llvm

void testInts() {
  std::vector<int> v;
  v.push_back(42);
  v.push_back(int(42));
  v.push_back(int{42});
  v.push_back(42.0);
  int z;
  v.push_back(z);
}

struct Something {
  Something(int a, int b = 41) {}
  Something() {}
  void push_back(Something);
  int getInt() { return 42; }
};

struct Convertable {
  operator Something() { return Something{}; }
};

struct Zoz {
  Zoz(Something, int = 42) {}
};

Zoz getZoz(Something s) { return Zoz(s); }

void test_Something() {
  std::vector<Something> v;

  v.push_back(Something(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back instead of push_back [modernize-use-emplace]
  // CHECK-FIXES: v.emplace_back(1, 2);

  v.push_back(Something{1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(1, 2);

  v.push_back(Something());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back();

  v.push_back(Something{});
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back();

  Something Different;
  v.push_back(Something(Different.getInt(), 42));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(Different.getInt(), 42);

  v.push_back(Different.getInt());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(Different.getInt());

  v.push_back(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(42);

  Something temporary(42, 42);
  temporary.push_back(temporary);
  v.push_back(temporary);

  v.push_back(Convertable());
  v.push_back(Convertable{});
  Convertable s;
  v.push_back(s);
}

template <typename ElemType>
void dependOnElem() {
  std::vector<ElemType> v;
  v.push_back(ElemType(42));
}

template <typename ContainerType>
void dependOnContainer() {
  ContainerType v;
  v.push_back(Something(42));
}

void callDependent() {
  dependOnElem<Something>();
  dependOnContainer<std::vector<Something>>();
}

void test2() {
  std::vector<Zoz> v;
  v.push_back(Zoz(Something(21, 37)));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(Something(21, 37));

  v.push_back(Zoz(Something(21, 37), 42));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(Something(21, 37), 42);

  v.push_back(getZoz(Something(1, 2)));
}

struct GetPair {
  std::pair<int, long> getPair();
};
void testPair() {
  std::vector<std::pair<int, int>> v;
  v.push_back(std::pair<int, int>(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(1, 2);

  GetPair g;
  v.push_back(g.getPair());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(g.getPair());

  std::vector<std::pair<Something, Zoz>> v2;
  v2.push_back(std::pair<Something, Zoz>(Something(42, 42), Zoz(Something(21, 37))));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use emplace_back
  // CHECK-FIXES: v2.emplace_back(Something(42, 42), Zoz(Something(21, 37)));
}

void testTuple() {
  std::vector<std::tuple<bool, char, int>> v;
  v.push_back(std::tuple<bool, char, int>(false, 'x', 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(false, 'x', 1);

  v.push_back(std::tuple<bool, char, int>{false, 'y', 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(false, 'y', 2);

  v.push_back({true, 'z', 3});
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(true, 'z', 3);

  std::vector<std::tuple<int, bool>> x;
  x.push_back(std::make_pair(1, false));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: x.emplace_back(1, false);

  x.push_back(std::make_pair(2LL, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: x.emplace_back(2LL, 1);
}

struct Base {
  Base(int, int *, int = 42);
};

struct Derived : Base {
  Derived(int *, Something) : Base(42, nullptr) {}
};

void testDerived() {
  std::vector<Base> v;
  v.push_back(Derived(nullptr, Something{}));
}

void testNewExpr() {
  std::vector<Derived> v;
  v.push_back(Derived(new int, Something{}));
}

void testSpaces() {
  std::vector<Something> v;

  // clang-format off

  v.push_back(Something(1, //arg1
                2 // arg2
               ) // Something
              );
  // CHECK-MESSAGES: :[[@LINE-4]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(1, //arg1
  // CHECK-FIXES:                2 // arg2
  // CHECK-FIXES:                  // Something
  // CHECK-FIXES:                );

  v.push_back(    Something   (1, 2)    );
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(1, 2   );

  v.push_back(    Something   {1, 2}    );
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(1, 2   );

  v.push_back(  Something {}    );
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(   );

  v.push_back(
             Something(1, 2)    );
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(1, 2   );

  std::vector<Base> v2;
  v2.push_back(
    Base(42, nullptr));
  // CHECK-MESSAGES: :[[@LINE-2]]:6: warning: use emplace_back
  // CHECK-FIXES: v2.emplace_back(42, nullptr);

  // clang-format on
}

void testPointers() {
  std::vector<int *> v;
  v.push_back(new int(5));

  std::vector<std::unique_ptr<int>> v2;
  v2.push_back(std::unique_ptr<int>(new int(42)));
  // This call can't be replaced with emplace_back.
  // If emplacement will fail (not enough memory to add to vector)
  // we will have leak of int because unique_ptr won't be constructed
  // (and destructed) as in push_back case.

  auto *ptr = new int;
  v2.push_back(std::unique_ptr<int>(ptr));
  // Same here
}

void testMakePair() {
  std::vector<std::pair<int, int>> v;
  v.push_back(std::make_pair(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(1, 2);

  v.push_back(std::make_pair(42LL, 13));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(42LL, 13);

  v.push_back(std::make_pair<char, char>(0, 3));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(std::make_pair<char, char>(0, 3));
  //
  // Even though the call above could be turned into v.emplace_back(0, 3),
  // we don't eliminate the make_pair call here, because of the explicit
  // template parameters provided. make_pair's arguments can be convertible
  // to its explicitly provided template parameter, but not to the pair's
  // element type. The examples below illustrate the problem.
  struct D {
    D(...) {}
    operator char() const { return 0; }
  };
  v.push_back(std::make_pair<D, int>(Something(), 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(std::make_pair<D, int>(Something(), 2));

  struct X {
    X(std::pair<int, int>) {}
  };
  std::vector<X> x;
  x.push_back(std::make_pair(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: x.emplace_back(std::make_pair(1, 2));
  // make_pair cannot be removed here, as X is not constructible with two ints.

  struct Y {
    Y(std::pair<int, int>&&) {}
  };
  std::vector<Y> y;
  y.push_back(std::make_pair(2, 3));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: y.emplace_back(std::make_pair(2, 3));
  // make_pair cannot be removed here, as Y is not constructible with two ints.
}

void testMakeTuple() {
  std::vector<std::tuple<int, bool, char>> v;
  v.push_back(std::make_tuple(1, true, 'v'));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(1, true, 'v');

  v.push_back(std::make_tuple(2ULL, 1, 0));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(2ULL, 1, 0);

  v.push_back(std::make_tuple<long long, int, int>(3LL, 1, 0));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(std::make_tuple<long long, int, int>(3LL, 1, 0));
  // make_tuple is not removed when there are explicit template
  // arguments provided.
}

namespace test {
template <typename T> struct Single {
  Single() = default;
  Single(const Single &) = default;
  Single(Single &&) = default;

  Single(const T &) {}
  Single(T &&) {}

  template <typename U> Single(const Single<U> &) {}
  template <typename U> Single(Single<U> &&) {}

  template <typename U> Single(const std::tuple<U> &) {}
  template <typename U> Single(std::tuple<U> &&) {}
};

template <typename T>
Single<typename std::remove_reference<T>::type> MakeSingle(T &&) {
  return {};
}
} // namespace test

void testOtherTuples() {
  std::vector<test::Single<int>> v;
  v.push_back(test::Single<int>(1));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(1);

  v.push_back({2});
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(2);

  v.push_back(test::MakeSingle(3));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(3);

  v.push_back(test::MakeSingle<long long>(4));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(test::MakeSingle<long long>(4));
  // We don't remove make functions with explicit template parameters.

  v.push_back(test::MakeSingle(5LL));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(5LL);

  v.push_back(std::make_tuple(6));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(6);

  v.push_back(std::make_tuple(7LL));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(7LL);
}

void testOtherContainers() {
  std::list<Something> l;
  l.push_back(Something(42, 41));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: l.emplace_back(42, 41);

  std::deque<Something> d;
  d.push_back(Something(42));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: d.emplace_back(42);

  llvm::LikeASmallVector<Something> ls;
  ls.push_back(Something(42));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use emplace_back
  // CHECK-FIXES: ls.emplace_back(42);
}

class IntWrapper {
public:
  IntWrapper(int x) : value(x) {}
  IntWrapper operator+(const IntWrapper other) const {
    return IntWrapper(value + other.value);
  }

private:
  int value;
};

void testMultipleOpsInPushBack() {
  std::vector<IntWrapper> v;
  v.push_back(IntWrapper(42) + IntWrapper(27));
}

// Macro tests.
#define PUSH_BACK_WHOLE(c, x) c.push_back(x)
#define PUSH_BACK_NAME push_back
#define PUSH_BACK_ARG(x) (x)
#define SOME_OBJ Something(10)
#define MILLION 3
#define SOME_WEIRD_PUSH(v) v.push_back(Something(
#define OPEN (
#define CLOSE )
void macroTest() {
  std::vector<Something> v;
  Something s;

  PUSH_BACK_WHOLE(v, Something(5, 6));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use emplace_back

  v.PUSH_BACK_NAME(Something(5));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back

  v.push_back PUSH_BACK_ARG(Something(5, 6));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back

  v.push_back(SOME_OBJ);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back

  v.push_back(Something(MILLION));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(MILLION);

  // clang-format off
  v.push_back(  Something OPEN 3 CLOSE  );
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // clang-format on
  PUSH_BACK_WHOLE(s, Something(1));
}

struct A {
  int value1, value2;
};

struct B {
  B(A) {}
};

struct C {
  int value1, value2, value3;
};

void testAggregation() {
  // This should not be noticed or fixed; after the correction, the code won't
  // compile.

  std::vector<A> v;
  v.push_back(A({1, 2}));

  std::vector<B> vb;
  vb.push_back(B({10, 42}));
}

struct Bitfield {
  unsigned bitfield : 1;
  unsigned notBitfield;
};

void testBitfields() {
  std::vector<Something> v;
  Bitfield b;
  v.push_back(Something(42, b.bitfield));
  v.push_back(Something(b.bitfield));

  v.push_back(Something(42, b.notBitfield));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(42, b.notBitfield);
  int var;
  v.push_back(Something(42, var));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(42, var);
}

class PrivateCtor {
  PrivateCtor(int z);

public:
  void doStuff() {
    std::vector<PrivateCtor> v;
    // This should not change it because emplace back doesn't have permission.
    // Check currently doesn't support friend declarations because pretty much
    // nobody would want to be friend with std::vector :(.
    v.push_back(PrivateCtor(42));
  }
};

struct WithDtor {
  WithDtor(int) {}
  ~WithDtor();
};

void testWithDtor() {
  std::vector<WithDtor> v;

  v.push_back(WithDtor(42));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use emplace_back
  // CHECK-FIXES: v.emplace_back(42);
}

void testInitializerList() {
  std::vector<std::vector<int>> v;
  v.push_back(std::vector<int>({1}));
  // Test against the bug reported in PR32896.

  v.push_back({{2}});

  using PairIntVector = std::pair<int, std::vector<int>>;
  std::vector<PairIntVector> x;
  x.push_back(PairIntVector(3, {4}));
  x.push_back({5, {6}});
}
