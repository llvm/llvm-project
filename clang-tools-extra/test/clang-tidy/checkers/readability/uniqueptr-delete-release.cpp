// RUN: %check_clang_tidy %s readability-uniqueptr-delete-release %t -check-suffix=NULLPTR
// RUN: %check_clang_tidy %s readability-uniqueptr-delete-release %t -check-suffix=RESET -config='{ \
// RUN: CheckOptions: [{key: readability-uniqueptr-delete-release.PreferResetCall, value: true}]}'
namespace std {
template <typename T>
struct default_delete {};

template <typename T, typename D = default_delete<T>>
class unique_ptr {
 public:
  unique_ptr();
  ~unique_ptr();
  explicit unique_ptr(T*);
  template <typename U, typename E>
  unique_ptr(unique_ptr<U, E>&&);
  T* release();
  void reset(T *P = nullptr);
  T &operator*() const;
  T *operator->() const;
};
}  // namespace std

std::unique_ptr<int>& ReturnsAUnique();

void Positives() {
  std::unique_ptr<int> P;
  delete P.release();
  // CHECK-MESSAGES-NULLPTR: :[[@LINE-1]]:3: warning: prefer '= nullptr' to reset 'unique_ptr<>' objects
  // CHECK-MESSAGES-RESET: :[[@LINE-2]]:3: warning: prefer 'reset()' to reset 'unique_ptr<>' objects
  // CHECK-FIXES-NULLPTR: {{^}}  P = nullptr;
  // CHECK-FIXES-RESET: {{^}}  P.reset();

  auto P2 = P;
  delete P2.release();
  // CHECK-MESSAGES-NULLPTR: :[[@LINE-1]]:3: warning: prefer '= nullptr' to reset 'unique_ptr<>' objects
  // CHECK-MESSAGES-RESET: :[[@LINE-2]]:3: warning: prefer 'reset()' to reset 'unique_ptr<>' objects
  // CHECK-FIXES-NULLPTR: {{^}}  P2 = nullptr;
  // CHECK-FIXES-RESET: {{^}}  P2.reset();

  delete (P2.release());
  // CHECK-MESSAGES-NULLPTR: :[[@LINE-1]]:3: warning: prefer '= nullptr'
  // CHECK-MESSAGES-RESET: :[[@LINE-2]]:3: warning: prefer 'reset()'
  // CHECK-FIXES-NULLPTR: {{^}}  (P2 = nullptr);
  // CHECK-FIXES-RESET: {{^}}  (P2.reset());

  std::unique_ptr<int> Array[20];
  delete Array[4].release();
  // CHECK-MESSAGES-NULLPTR: :[[@LINE-1]]:3: warning: prefer '= nullptr'
  // CHECK-MESSAGES-RESET: :[[@LINE-2]]:3: warning: prefer 'reset()'
  // CHECK-FIXES-NULLPTR: {{^}}  Array[4] = nullptr;
  // CHECK-FIXES-RESET: {{^}}  Array[4].reset();

  delete ReturnsAUnique().release();
  // CHECK-MESSAGES-NULLPTR: :[[@LINE-1]]:3: warning: prefer '= nullptr'
  // CHECK-MESSAGES-RESET: :[[@LINE-2]]:3: warning: prefer 'reset()'
  // CHECK-FIXES-NULLPTR: {{^}}  ReturnsAUnique() = nullptr;
  // CHECK-FIXES-RESET: {{^}}  ReturnsAUnique().reset();

  std::unique_ptr<int> *P3(&P);
  delete P3->release();
  // CHECK-MESSAGES-NULLPTR: :[[@LINE-1]]:3: warning: prefer '= nullptr'
  // CHECK-MESSAGES-RESET: :[[@LINE-2]]:3: warning: prefer 'reset()'
  // CHECK-FIXES-NULLPTR: {{^}}  *P3 = nullptr;
  // CHECK-FIXES-RESET: {{^}}  P3->reset();

  std::unique_ptr<std::unique_ptr<int>> P4;
  delete (*P4).release();
  // CHECK-MESSAGES-NULLPTR: :[[@LINE-1]]:3: warning: prefer '= nullptr'
  // CHECK-MESSAGES-RESET: :[[@LINE-2]]:3: warning: prefer 'reset()'
  // CHECK-FIXES-NULLPTR: {{^}}  (*P4) = nullptr;
  // CHECK-FIXES-RESET: {{^}}  (*P4).reset();

  delete P4->release();
  // CHECK-MESSAGES-NULLPTR: :[[@LINE-1]]:3: warning: prefer '= nullptr'
  // CHECK-MESSAGES-RESET: :[[@LINE-2]]:3: warning: prefer 'reset()'
  // CHECK-FIXES-NULLPTR: {{^}}  *P4 = nullptr;
  // CHECK-FIXES-RESET: {{^}}  P4->reset();

  delete (P4)->release();
  // CHECK-MESSAGES-NULLPTR: :[[@LINE-1]]:3: warning: prefer '= nullptr'
  // CHECK-MESSAGES-RESET: :[[@LINE-2]]:3: warning: prefer 'reset()'
  // CHECK-FIXES-NULLPTR: {{^}}  *(P4) = nullptr;
  // CHECK-FIXES-RESET: {{^}}  (P4)->reset();
}

struct NotDefaultDeleter {};

struct NotUniquePtr {
  int* release();
};

void Negatives() {
  std::unique_ptr<int, NotDefaultDeleter> P;
  delete P.release();

  NotUniquePtr P2;
  delete P2.release();

  // We don't trigger on bound member function calls.
  delete (P2.release)();
}

template <typename T, typename D>
void NegativeDeleterT() {
  // Ideally this would trigger a warning, but we have all dependent types
  // disabled for now.
  std::unique_ptr<T> P;
  delete P.release();

  // We ignore this one because the deleter is a template argument.
  // Not all instantiations will use the default deleter.
  std::unique_ptr<int, D> P2;
  delete P2.release();
}
template void NegativeDeleterT<int, std::default_delete<int>>();

// Test some macros

#define DELETE_RELEASE(x) delete (x).release()
void NegativesWithTemplate() {
  std::unique_ptr<int> P;
  DELETE_RELEASE(P);
}
