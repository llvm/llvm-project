// RUN: %check_clang_tidy %s readability-redundant-smartptr-get %t

#define NULL __null

namespace std {

template <typename T>
struct unique_ptr {
  T& operator*() const;
  T* operator->() const;
  T* get() const;
};

template <typename T>
struct shared_ptr {
  T& operator*() const;
  T* operator->() const;
  T* get() const;
};

}  // namespace std

struct Bar {
  void Do();
  void ConstDo() const;
};
struct BarPtr {
  Bar* operator->();
  Bar& operator*();
  Bar* get();
};
struct int_ptr {
  int* get();
  int* operator->();
  int& operator*();
};

struct Fail1 {
  Bar* get();
};
struct Fail2 {
  Bar* get();
  int* operator->();
  int& operator*();
};

struct PointerWithOverloadedGet {
  int* get();
  template <typename T>
  T* get();
  int* operator->();
  int& operator*();
};

void Positive() {
  BarPtr u;
  // CHECK-FIXES: BarPtr u;
  BarPtr().get()->Do();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant get() call on smart pointer [readability-redundant-smartptr-get]
  // CHECK-MESSAGES: BarPtr().get()->Do();
  // CHECK-FIXES: BarPtr()->Do();

  u.get()->ConstDo();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant get() call
  // CHECK-MESSAGES: u.get()->ConstDo();
  // CHECK-FIXES: u->ConstDo();

  Bar& b = *BarPtr().get();
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant get() call
  // CHECK-MESSAGES: Bar& b = *BarPtr().get();
  // CHECK-FIXES: Bar& b = *BarPtr();

  Bar& b2 = *std::unique_ptr<Bar>().get();
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant get() call
  // CHECK-MESSAGES: Bar& b2 = *std::unique_ptr<Bar>().get();
  // CHECK-FIXES: Bar& b2 = *std::unique_ptr<Bar>();

  (*BarPtr().get()).ConstDo();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant get() call
  // CHECK-MESSAGES: (*BarPtr().get()).ConstDo();
  // CHECK-FIXES: (*BarPtr()).ConstDo();

  (*std::unique_ptr<Bar>().get()).ConstDo();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant get() call
  // CHECK-MESSAGES: (*std::unique_ptr<Bar>().get()).ConstDo();
  // CHECK-FIXES: (*std::unique_ptr<Bar>()).ConstDo();

  std::unique_ptr<Bar>* up;
  (*up->get()).Do();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant get() call
  // CHECK-MESSAGES: (*up->get()).Do();
  // CHECK-FIXES: (**up).Do();

  int_ptr ip;
  int i = *ip.get();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant get() call
  // CHECK-MESSAGES: int i = *ip.get();
  // CHECK-FIXES: int i = *ip;

  auto ip2 = ip;
  i = *ip2.get();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant get() call
  // CHECK-MESSAGES: i = *ip2.get();
  // CHECK-FIXES: i = *ip2;

  std::unique_ptr<int> uu;
  std::shared_ptr<double> *ss;
  bool bb = uu.get() == nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant get() call
  // CHECK-MESSAGES: uu.get() == nullptr;
  // CHECK-FIXES: bool bb = uu == nullptr;

  bb = nullptr != ss->get();
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: redundant get() call
  // CHECK-MESSAGES: nullptr != ss->get();
  // CHECK-FIXES: bb = nullptr != *ss;

  i = *PointerWithOverloadedGet().get();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant get() call
  // CHECK-MESSAGES: i = *PointerWithOverloadedGet().get();
  // CHECK-FIXES: i = *PointerWithOverloadedGet();

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
  if (x.get());
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant get() call
  // CHECK-MESSAGES: if (x.get());
  // CHECK-FIXES: if (x);
}

void Negative() {
  struct NegPtr {
    int* get();
    int* operator->() {
      return &*this->get();
    }
    int& operator*() {
      return *get();
    }
  };

  long l = *PointerWithOverloadedGet().get<long>();

  std::unique_ptr<Bar>* u;
  u->get()->Do();

  Fail1().get()->Do();
  Fail2().get()->Do();
  const Bar& b = *Fail1().get();
  (*Fail2().get()).Do();

  int_ptr ip;
  bool bb = ip.get() == nullptr;
}
