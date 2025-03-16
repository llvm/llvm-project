// RUN: %check_clang_tidy %s -std=c++20 modernize-use-emplace %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {modernize-use-emplace.ContainersWithPushBack: \
// RUN:                '::std::vector; ::std::list; ::std::deque; llvm::LikeASmallVector', \
// RUN:              modernize-use-emplace.TupleTypes: \
// RUN:                '::std::pair; std::tuple; ::test::Single', \
// RUN:              modernize-use-emplace.TupleMakeFunctions: \
// RUN:                '::std::make_pair; ::std::make_tuple; ::test::MakeSingle'}}"

namespace std {
template <typename E>
class initializer_list {
public:
  const E *a, *b;
  initializer_list() noexcept {}
};

template <typename T>
class vector {
public:
  using value_type = T;

  class iterator {};
  class const_iterator {};
  const_iterator begin() { return const_iterator{}; }

  vector() = default;
  vector(initializer_list<T>) {}

  void push_back(const T &) {}
  void push_back(T &&) {}

  template <typename... Args>
  void emplace_back(Args &&... args){};
  template <typename... Args>
  iterator emplace(const_iterator pos, Args &&...args);
  ~vector();
};

} // namespace std

struct InnerType {
  InnerType() {}
  InnerType(char const*) {}
};

//Not aggregate but we should still be able to directly initialize it with emplace_back
struct NonTrivialNoCtor {
  InnerType it;
};

struct Aggregate {
  int a;
  int b;
};

void testCXX20Cases() {
  std::vector<Aggregate> v1;

  v1.push_back(Aggregate{1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use emplace_back instead of push_back
  // CHECK-FIXES: v1.emplace_back(1, 2);

  v1.push_back({1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use emplace_back instead of push_back
  // CHECK-FIXES: v1.emplace_back(1, 2);

  std::vector<NonTrivialNoCtor> v2;

  v2.push_back(NonTrivialNoCtor{""});
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use emplace_back instead of push_back
  // CHECK-FIXES: v2.emplace_back("");

  v2.push_back({""});
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use emplace_back instead of push_back
  // CHECK-FIXES: v2.emplace_back("");

  v2.push_back(NonTrivialNoCtor{InnerType{""}});
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use emplace_back instead of push_back
  // CHECK-FIXES: v2.emplace_back(InnerType{""});

  v2.push_back({InnerType{""}});
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use emplace_back instead of push_back
  // CHECK-FIXES: v2.emplace_back(InnerType{""});

}
