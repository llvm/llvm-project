// RUN: %check_clang_tidy %s readability-uniqueptr-delete-release %t

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
};
}  // namespace std

std::unique_ptr<int>& ReturnsAUnique();

void Positives() {
  std::unique_ptr<int> P;
  delete P.release();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer '= nullptr' to 'delete x.release()' to reset unique_ptr<> objects [readability-uniqueptr-delete-release]
  // CHECK-FIXES: {{^}}  P = nullptr;

  auto P2 = P;
  delete P2.release();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer '= nullptr' to 'delete x.release()' to reset unique_ptr<> objects [readability-uniqueptr-delete-release]
  // CHECK-FIXES: {{^}}  P2 = nullptr;

  std::unique_ptr<int> Array[20];
  delete Array[4].release();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer '= nullptr' to 'delete
  // CHECK-FIXES: {{^}}  Array[4] = nullptr;

  delete ReturnsAUnique().release();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer '= nullptr' to 'delete
  // CHECK-FIXES: {{^}}  ReturnsAUnique() = nullptr;
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
