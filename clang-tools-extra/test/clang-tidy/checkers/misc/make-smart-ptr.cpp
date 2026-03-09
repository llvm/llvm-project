// RUN: %check_clang_tidy %s misc-make-smart-ptr %t -- \
// RUN:   -config="{CheckOptions: [{key: misc-make-smart-ptr.MakeSmartPtrType, value: '::base::scoped_refptr'}, \
// RUN:                           {key: misc-make-smart-ptr.MakeSmartPtrFunction, value: 'base::MakeRefCounted'}, \
// RUN:                           {key: misc-make-smart-ptr.MakeSmartPtrFunctionHeader, value: ''}]}" \
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

void testReset() {
  base::scoped_refptr<Base> P1;
  P1.reset(new Base());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use base::MakeRefCounted instead
  // CHECK-FIXES: P1 = base::MakeRefCounted<Base>();

  P1.reset(new Base(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use base::MakeRefCounted instead
  // CHECK-FIXES: P1 = base::MakeRefCounted<Base>(1, 2);
}

base::scoped_refptr<Base> factory() {
  return base::scoped_refptr<Base>(new Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use base::MakeRefCounted instead
  // CHECK-FIXES: return base::MakeRefCounted<Base>();
}
