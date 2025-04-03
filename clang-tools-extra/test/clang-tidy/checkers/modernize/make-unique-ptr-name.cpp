// RUN: %check_clang_tidy %s modernize-make-unique %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-make-unique.MakeSmartPtrType, value: '::base::unique_ptr'}, \
// RUN:                           {key: modernize-make-unique.MakeSmartPtrFunction, value: 'base::make_unique'}]}" \
// RUN:   -- -std=c++11 -nostdinc++

namespace std {
  template<typename T>
  struct default_delete {
    void operator()(T* ptr) const { delete ptr; }
  };
}

namespace base {

using nullptr_t = decltype(nullptr);

template <typename T, typename Deleter = std::default_delete<T>>
class unique_ptr {
public:
  typedef T element_type;
  typedef T* pointer;

  constexpr unique_ptr() noexcept : ptr_(nullptr) {}
  constexpr unique_ptr(nullptr_t) noexcept : ptr_(nullptr) {}
  explicit unique_ptr(T* p) noexcept : ptr_(p) {}
  unique_ptr(unique_ptr&& r) noexcept : ptr_(r.ptr_) { r.ptr_ = nullptr; }
  unique_ptr(const unique_ptr&) = delete;
  unique_ptr& operator=(const unique_ptr&) = delete;
  unique_ptr& operator=(unique_ptr&& r) noexcept {
    T* old = ptr_;
    ptr_ = r.ptr_;
    r.ptr_ = nullptr;
    delete old;
    return *this;
  }
  void reset(T* p = nullptr) noexcept {
    T* old = ptr_;
    ptr_ = p;
    delete old;
  }
  T* get() const noexcept { return ptr_; }
  T& operator*() const noexcept { return *ptr_; }
  T* operator->() const noexcept { return ptr_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }
  ~unique_ptr() { delete ptr_; }
private:
  T* ptr_;
};

template <typename T>
unique_ptr<T> make_unique() {
  return unique_ptr<T>(new T());
}

template <typename T, typename Arg1>
unique_ptr<T> make_unique(const Arg1& arg1) {
  return unique_ptr<T>(new T(arg1));
}

template <typename T, typename Arg1, typename Arg2>
unique_ptr<T> make_unique(const Arg1& arg1, const Arg2& arg2) {
  return unique_ptr<T>(new T(arg1, arg2));
}

}  // namespace base

struct Base {
  Base() {}
  Base(int, int) {}
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
