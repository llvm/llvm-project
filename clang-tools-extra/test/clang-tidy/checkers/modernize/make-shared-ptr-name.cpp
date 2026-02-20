// RUN: %check_clang_tidy %s modernize-make-shared %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-make-shared.MakeSmartPtrType, value: '::base::scoped_refptr'}, \
// RUN:                           {key: modernize-make-shared.MakeSmartPtrFunction, value: 'base::MakeRefCounted'}, \
// RUN:                           {key: modernize-make-shared.MakeSmartPtrFunctionHeader, value: ''}]}" \
// RUN:   -- -std=c++11 -nostdinc++

namespace base {

template <typename T>
class scoped_refptr {
public:
  scoped_refptr() : ptr_(nullptr) {}
  explicit scoped_refptr(T *p) : ptr_(p) {}
  scoped_refptr(const scoped_refptr &r) : ptr_(r.ptr_) {}
  scoped_refptr(scoped_refptr &&r) : ptr_(r.ptr_) { r.ptr_ = nullptr; }

  void reset(T *p = nullptr) { ptr_ = p; }
  T *get() const { return ptr_; }
  T &operator*() const { return *ptr_; }
  T *operator->() const { return ptr_; }

  ~scoped_refptr() {}

private:
  T *ptr_;
};

template <typename T, typename... Args>
scoped_refptr<T> MakeRefCounted(Args &&...args) {
  return scoped_refptr<T>(new T(args...));
}

} // namespace base

struct Base {
  Base();
  Base(int, int);
};

struct Derived : public Base {
  Derived();
  Derived(int, int);
};

void testReset() {
  base::scoped_refptr<Base> P1;
  P1.reset(new Base());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use base::MakeRefCounted instead
  // CHECK-FIXES: P1 = base::MakeRefCounted<Base>();

  P1.reset(new Derived());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use base::MakeRefCounted instead
  // CHECK-FIXES: P1 = base::MakeRefCounted<Derived>();
}

base::scoped_refptr<Base> factory() {
  return base::scoped_refptr<Base>(new Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use base::MakeRefCounted instead
  // CHECK-FIXES: return base::MakeRefCounted<Base>();
}
