// RUN: %check_clang_tidy %s bugprone-unused-return-value %t -- \
// RUN:   --config="{CheckOptions: {bugprone-unused-return-value.AllowCastToVoid: true}}" -- -fexceptions

namespace std {

struct future {};

enum class launch {
  async,
  deferred
};

template <typename Function, typename... Args>
future async(Function &&, Args &&...);

template <typename Function, typename... Args>
future async(launch, Function &&, Args &&...);

template <typename ForwardIt, typename T>
ForwardIt remove(ForwardIt, ForwardIt, const T &);

template <typename ForwardIt, typename UnaryPredicate>
ForwardIt remove_if(ForwardIt, ForwardIt, UnaryPredicate);

template <typename ForwardIt>
ForwardIt unique(ForwardIt, ForwardIt);

template <typename T>
struct default_delete;

template <typename T, typename Deleter = std::default_delete<T>>
struct unique_ptr {
  T *release() noexcept;
};

template <typename T>
struct char_traits;

template <typename T>
struct allocator;

template <typename CharT,
          typename Traits = char_traits<CharT>,
          typename Allocator = allocator<CharT>>
struct basic_string {
  bool empty() const;
};

typedef basic_string<char> string;

template <typename T, typename Allocator = std::allocator<T>>
struct vector {
  bool empty() const noexcept;
};

class error_code {
};

// the check should be able to match std lib calls even if the functions are
// declared inside inline namespaces
inline namespace v1 {

template <typename T>
T *launder(T *);

} // namespace v1
} // namespace std

struct Foo {
  void f();
};

int increment(int i) {
  return i + 1;
}

void useFuture(const std::future &fut);

std::error_code errorFunc() {
    return std::error_code();
}

void warning() {
  std::async(increment, 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  std::async(std::launch::async, increment, 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  Foo F;
  std::launder(&F);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  std::remove_if(nullptr, nullptr, nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  std::unique(nullptr, nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  std::unique_ptr<Foo> UPtr;
  UPtr.release();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  std::string Str;
  Str.empty();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  (int)Str.empty();
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:8: note: cast the expression to void to silence this warning

  std::vector<Foo> Vec;
  Vec.empty();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  // test discarding return values inside different kinds of statements

  auto Lambda = [] { std::remove(nullptr, nullptr, 1); };
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:22: note: cast the expression to void to silence this warning

  if (true)
    std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning
  else if (true)
    std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning
  else
    std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning

  while (true)
    std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning

  do
    std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning
  while (true);

  for (;;)
    std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning

  for (std::remove(nullptr, nullptr, 1);;)
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
    // CHECK-MESSAGES: [[@LINE-2]]:8: note: cast the expression to void to silence this warning
    ;

  for (;; std::remove(nullptr, nullptr, 1))
    // CHECK-MESSAGES: [[@LINE-1]]:11: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
    // CHECK-MESSAGES: [[@LINE-2]]:11: note: cast the expression to void to silence this warning
    ;

  for (auto C : "foo")
    std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning

  switch (1) {
  case 1:
    std::remove(nullptr, nullptr, 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
    // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning
    break;
  default:
    std::remove(nullptr, nullptr, 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
    // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning
    break;
  }

  try {
    std::remove(nullptr, nullptr, 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
    // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning
  } catch (...) {
    std::remove(nullptr, nullptr, 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
    // CHECK-MESSAGES: [[@LINE-2]]:5: note: cast the expression to void to silence this warning
  }

  errorFunc();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning
}

void noWarning() {
  auto AsyncRetval1 = std::async(increment, 42);
  auto AsyncRetval2 = std::async(std::launch::async, increment, 42);

  Foo FNoWarning;
  auto LaunderRetval = std::launder(&FNoWarning);

  auto RemoveRetval = std::remove(nullptr, nullptr, 1);

  auto RemoveIfRetval = std::remove_if(nullptr, nullptr, nullptr);

  auto UniqueRetval = std::unique(nullptr, nullptr);

  std::unique_ptr<Foo> UPtrNoWarning;
  auto ReleaseRetval = UPtrNoWarning.release();

  std::string StrNoWarning;
  auto StrEmptyRetval = StrNoWarning.empty();

  std::vector<Foo> VecNoWarning;
  auto VecEmptyRetval = VecNoWarning.empty();

  (void) errorFunc();

  // test using the return value in different kinds of expressions
  useFuture(std::async(increment, 42));
  std::launder(&FNoWarning)->f();
  delete std::launder(&FNoWarning);

  if (std::launder(&FNoWarning))
    ;
  for (; std::launder(&FNoWarning);)
    ;
  while (std::launder(&FNoWarning))
    ;
  do
    ;
  while (std::launder(&FNoWarning));
  switch (std::unique(1, 1))
    ;

  // cast to void should allow ignoring the return value
  (void)std::async(increment, 42);

  // test discarding return value of functions that are not configured to be checked
  increment(1);

  // test that the check is disabled inside GNU statement expressions
  ({ std::async(increment, 42); });
  auto StmtExprRetval = ({ std::async(increment, 42); });
}
