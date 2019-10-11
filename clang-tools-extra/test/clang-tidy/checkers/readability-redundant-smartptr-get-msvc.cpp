// RUN: %check_clang_tidy %s readability-redundant-smartptr-get %t

#define NULL __null

namespace std {

// MSVC headers define operator templates instead of plain operators.

template <typename T>
struct unique_ptr {
  template <typename T2 = T>
  T2& operator*() const;
  template <typename T2 = T>
  T2* operator->() const;
  T* get() const;
  explicit operator bool() const noexcept;
};

template <typename T>
struct shared_ptr {
  template <typename T2 = T>
  T2& operator*() const;
  template <typename T2 = T>
  T2* operator->() const;
  T* get() const;
  explicit operator bool() const noexcept;
};

}  // namespace std

struct Bar {
  void Do();
  void ConstDo() const;
};

void Positive() {
  std::unique_ptr<Bar>* up;
  (*up->get()).Do();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant get() call
  // CHECK-MESSAGES: (*up->get()).Do();
  // CHECK-FIXES: (**up).Do();

  std::unique_ptr<int> uu;
  std::shared_ptr<double> *ss;
  bool bb = uu.get() == nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant get() call
  // CHECK-MESSAGES: uu.get() == nullptr;
  // CHECK-FIXES: bool bb = uu == nullptr;

  if (up->get());
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant get() call
  // CHECK-MESSAGES: if (up->get());
  // CHECK-FIXES: if (*up);
  if ((uu.get()));
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant get() call
  // CHECK-MESSAGES: if ((uu.get()));
  // CHECK-FIXES: if ((uu));
  bb = !ss->get();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant get() call
  // CHECK-MESSAGES: bb = !ss->get();
  // CHECK-FIXES: bb = !*ss;

  bb = nullptr != ss->get();
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: redundant get() call
  // CHECK-MESSAGES: nullptr != ss->get();
  // CHECK-FIXES: bb = nullptr != *ss;

  bb = std::unique_ptr<int>().get() == NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant get() call
  // CHECK-MESSAGES: bb = std::unique_ptr<int>().get() == NULL;
  // CHECK-FIXES: bb = std::unique_ptr<int>() == NULL;
  bb = ss->get() == NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant get() call
  // CHECK-MESSAGES: bb = ss->get() == NULL;
  // CHECK-FIXES: bb = *ss == NULL;

  std::unique_ptr<int> x, y;
  if (x.get() == nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant get() call
  // CHECK-MESSAGES: if (x.get() == nullptr);
  // CHECK-FIXES: if (x == nullptr);
  if (nullptr == y.get());
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: redundant get() call
  // CHECK-MESSAGES: if (nullptr == y.get());
  // CHECK-FIXES: if (nullptr == y);
  if (x.get() == NULL);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant get() call
  // CHECK-MESSAGES: if (x.get() == NULL);
  // CHECK-FIXES: if (x == NULL);
  if (NULL == x.get());
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: redundant get() call
  // CHECK-MESSAGES: if (NULL == x.get());
  // CHECK-FIXES: if (NULL == x);
}
