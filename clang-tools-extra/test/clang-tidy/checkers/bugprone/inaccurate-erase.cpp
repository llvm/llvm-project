// RUN: %check_clang_tidy %s bugprone-inaccurate-erase %t

namespace std {
template <typename T> struct vec_iterator {
  T ptr;
  vec_iterator operator++(int);

  template <typename X>
  vec_iterator(const vec_iterator<X> &); // Omit enable_if<...>.
};

template <typename T> struct vector {
  typedef vec_iterator<T*> iterator;

  iterator begin();
  iterator end();

  void erase(iterator);
  void erase(iterator, iterator);
};

template <typename T> struct vector_with_const_iterator {
  typedef vec_iterator<T*> iterator;
  typedef vec_iterator<const T*> const_iterator;

  iterator begin();
  iterator end();

  void erase(const_iterator);
  void erase(const_iterator, const_iterator);
};

template <typename FwIt, typename T>
FwIt remove(FwIt begin, FwIt end, const T &val);

template <typename FwIt, typename Func>
FwIt remove_if(FwIt begin, FwIt end, Func f);

template <typename FwIt> FwIt unique(FwIt begin, FwIt end);

template <typename T> struct unique_ptr {};
} // namespace std

struct custom_iter {};
struct custom_container {
  void erase(...);
  custom_iter begin();
  custom_iter end();
};

template <typename T> void g() {
  T t;
  t.erase(std::remove(t.begin(), t.end(), 10));
  // CHECK-FIXES: t.erase(std::remove(t.begin(), t.end(), 10));

  std::vector<int> v;
  v.erase(remove(v.begin(), v.end(), 10));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this call will remove at most one
  // CHECK-FIXES: v.erase(remove(v.begin(), v.end(), 10), v.end());
}

#define ERASE(x, y) x.erase(remove(x.begin(), x.end(), y))
// CHECK-FIXES: #define ERASE(x, y) x.erase(remove(x.begin(), x.end(), y))

int main() {
  std::vector<int> v;

  v.erase(remove(v.begin(), v.end(), 10));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this call will remove at most one item even when multiple items should be removed [bugprone-inaccurate-erase]
  // CHECK-FIXES: v.erase(remove(v.begin(), v.end(), 10), v.end());
  v.erase(remove(v.begin(), v.end(), 20), v.end());

  auto *p = &v;
  p->erase(remove(p->begin(), p->end(), 11));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this call will remove at most one
  // CHECK-FIXES: p->erase(remove(p->begin(), p->end(), 11), p->end());

  std::vector_with_const_iterator<int> v2;
  v2.erase(remove(v2.begin(), v2.end(), 12));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this call will remove at most one
  // CHECK-FIXES: v2.erase(remove(v2.begin(), v2.end(), 12), v2.end());

  // Fix is not trivial.
  auto it = v.end();
  v.erase(remove(v.begin(), it, 10));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this call will remove at most one
  // CHECK-FIXES: v.erase(remove(v.begin(), it, 10));

  g<std::vector<int>>();
  g<custom_container>();

  ERASE(v, 15);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: this call will remove at most one
  // CHECK-FIXES: ERASE(v, 15);

  std::vector<std::unique_ptr<int>> vupi;
  auto iter = vupi.begin();
  vupi.erase(iter++);
  // CHECK-FIXES: vupi.erase(iter++);
}
