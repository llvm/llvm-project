// RUN: %check_clang_tidy %s modernize-make-unique %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-make-unique.MakeSmartPtrType, value: '::base::unique_ptr'}, \
// RUN:                           {key: modernize-make-unique.MakeSmartPtrFunction, value: 'base::make_unique'}, \
// RUN:                           {key: modernize-make-unique.MakeSmartPtrFunctionHeader, value: ''}]}" \
// RUN:   -- -std=c++14 -nostdinc++

namespace std {

template <typename T>
struct default_delete {
  void operator()(T *ptr) const { delete ptr; }
};

} // namespace std

namespace base {

template <typename T, typename Deleter = std::default_delete<T>>
class unique_ptr {
public:
  unique_ptr() : ptr_(nullptr) {}
  explicit unique_ptr(T *p) : ptr_(p) {}
  unique_ptr(unique_ptr &&r) : ptr_(r.ptr_) { r.ptr_ = nullptr; }
  unique_ptr(const unique_ptr &) = delete;
  unique_ptr &operator=(const unique_ptr &) = delete;
  unique_ptr &operator=(unique_ptr &&r) {
    delete ptr_;
    ptr_ = r.ptr_;
    r.ptr_ = nullptr;
    return *this;
  }

  void reset(T *p = nullptr) {
    delete ptr_;
    ptr_ = p;
  }

  T *get() const { return ptr_; }
  T &operator*() const { return *ptr_; }
  T *operator->() const { return ptr_; }
  ~unique_ptr() { delete ptr_; }

private:
  T *ptr_;
};

template <typename T, typename... Args>
unique_ptr<T> make_unique(Args &&...args) {
  return unique_ptr<T>(new T(args...));
}

} // namespace base

struct Base {
  Base();
  Base(int, int);
};

void test() {
  base::unique_ptr<Base> P1 = base::unique_ptr<Base>(new Base());
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use base::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: base::unique_ptr<Base> P1 = base::make_unique<Base>();

  P1.reset(new Base(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use base::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P1 = base::make_unique<Base>(1, 2);

  P1 = base::unique_ptr<Base>(new Base(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use base::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P1 = base::make_unique<Base>(1, 2);
}

base::unique_ptr<Base> factory() {
  return base::unique_ptr<Base>(new Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use base::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: return base::make_unique<Base>();
}
