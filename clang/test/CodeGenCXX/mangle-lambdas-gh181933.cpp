// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o /dev/null %s

namespace GH181933 {
template <typename Predicate>
void foo(Predicate pred) {
  pred(42);
}

template <typename Predicate>
auto bar(Predicate pred) {
  foo(pred);
}

template <typename T>
concept Baz = requires(const T& x) {
  {
    bar([](const auto&) { return true; })
  };
};

static_assert(Baz<int>);
}

namespace PR {
template <typename Predicate>
void foo(Predicate pred) {
  pred(42);
}

template <typename Predicate>
auto bar(Predicate pred) {
  foo(pred);
}

extern "C++" {
static_assert(requires(const int& x) {
  {
    bar([](const auto&) { return true; })
  };
});
}
}
