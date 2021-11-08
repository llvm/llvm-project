// RUN: %check_clang_tidy %s abseil-cleanup-ctad -std=c++17 %t

namespace std {

template <typename, typename>
struct is_same {
  static const bool value = false;
};

template <typename T>
struct is_same<T, T> { static const bool value = true; };

template <typename>
class function {
public:
  template <typename T>
  function(T) {}
  function(const function &) {}
};

} // namespace std

namespace absl {

namespace cleanup_internal {

struct Tag {};

template <typename Callback>
class Storage {
public:
  Storage() = delete;

  explicit Storage(Callback callback) {}

  Storage(Storage &&other) {}

  Storage(const Storage &other) = delete;

  Storage &operator=(Storage &&other) = delete;

  Storage &operator=(const Storage &other) = delete;

private:
  bool is_callback_engaged_;
  alignas(Callback) char callback_buffer_[sizeof(Callback)];
};

} // namespace cleanup_internal

template <typename Arg, typename Callback = void()>
class Cleanup final {
public:
  Cleanup(Callback callback) // NOLINT
      : storage_(static_cast<Callback &&>(callback)) {}

  Cleanup(Cleanup &&other) = default;

  void Cancel() &&;

  void Invoke() &&;

  ~Cleanup();

private:
  cleanup_internal::Storage<Callback> storage_;
};

template <typename Callback>
Cleanup(Callback callback) -> Cleanup<cleanup_internal::Tag, Callback>;

template <typename... Args, typename Callback>
absl::Cleanup<cleanup_internal::Tag, Callback> MakeCleanup(Callback callback) {
  return {static_cast<Callback &&>(callback)};
}

} // namespace absl

void test() {
  auto a = absl::MakeCleanup([] {});
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer absl::Cleanup's class template argument deduction pattern in C++17 and higher
  // CHECK-FIXES: {{^}}  absl::Cleanup a = [] {};{{$}}

  // Removes extra parens
  auto b = absl::MakeCleanup(([] {}));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer absl::Cleanup{{.*}}C++17 and higher
  // CHECK-FIXES: {{^}}  absl::Cleanup b = [] {};{{$}}

  auto c = absl::MakeCleanup(std::function<void()>([] {}));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer absl::Cleanup{{.*}}C++17 and higher
  // CHECK-FIXES: {{^}}  absl::Cleanup c = std::function<void()>([] {});{{$}}

  // Removes extra parens
  auto d = absl::MakeCleanup((std::function<void()>([] {})));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer absl::Cleanup{{.*}}C++17 and higher
  // CHECK-FIXES: {{^}}  absl::Cleanup d = std::function<void()>([] {});{{$}}

  const auto e = absl::MakeCleanup([] {});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: prefer absl::Cleanup{{.*}}C++17 and higher
  // CHECK-FIXES: {{^}}  const absl::Cleanup e = [] {};{{$}}

  // Removes extra parens
  const auto f = absl::MakeCleanup(([] {}));
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: prefer absl::Cleanup{{.*}}C++17 and higher
  // CHECK-FIXES: {{^}}  const absl::Cleanup f = [] {};{{$}}

  const auto g = absl::MakeCleanup(std::function<void()>([] {}));
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: prefer absl::Cleanup{{.*}}C++17 and higher
  // CHECK-FIXES: {{^}}  const absl::Cleanup g = std::function<void()>([] {});{{$}}

  // Removes extra parens
  const auto h = absl::MakeCleanup((std::function<void()>([] {})));
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: prefer absl::Cleanup{{.*}}C++17 and higher
  // CHECK-FIXES: {{^}}  const absl::Cleanup h = std::function<void()>([] {});{{$}}
}
