// RUN: %check_clang_tidy %s readability-redundant-smartptr-get %t -- \
// RUN:   -config="{CheckOptions: [{key: readability-redundant-smartptr-get.IgnoreMacros, value: 0}]}" \
// RUN:   -- -std=c++11

namespace std {

template <typename T>
struct shared_ptr {
  T &operator*() const;
  T *operator->() const;
  T *get() const;
  explicit operator bool() const noexcept;
};

} // namespace std

#define MACRO(p) p.get()

void Positive() {
  std::shared_ptr<int> x;
  if (MACRO(x) == nullptr)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:13: warning: redundant get() call on smart pointer
};
