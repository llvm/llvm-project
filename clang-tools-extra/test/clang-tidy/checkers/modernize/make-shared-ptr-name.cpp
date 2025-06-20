// RUN: %check_clang_tidy %s modernize-make-shared %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-make-shared.MakeSmartPtrType, value: '::base::scoped_refptr'}, \
// RUN:                           {key: modernize-make-shared.MakeSmartPtrFunction, value: 'base::MakeRefCounted'}]}" \
// RUN:   -- -std=c++11 -nostdinc++

namespace base {

using nullptr_t = decltype(nullptr);

template <typename T>
struct remove_extent { typedef T type; };

template <typename T>
struct remove_reference { typedef T type; };

template <typename T>
class scoped_refptr {
public:
  typedef T element_type;
  typedef T* pointer;
  typedef element_type& reference;

  // Default constructors
  constexpr scoped_refptr() noexcept : ptr_(nullptr) {}
  constexpr scoped_refptr(nullptr_t) noexcept : ptr_(nullptr) {}
  
  // Constructors from raw pointer
  explicit scoped_refptr(T* p) noexcept : ptr_(p) {}
  scoped_refptr(T* p, bool) noexcept : ptr_(p) {} // Special constructor for temporaries

  // Copy constructors
  scoped_refptr(const scoped_refptr& r) noexcept : ptr_(r.ptr_) {}
  template<typename U>
  scoped_refptr(const scoped_refptr<U>& r) noexcept : ptr_(r.get()) {}

  // Move constructors  
  scoped_refptr(scoped_refptr&& r) noexcept : ptr_(r.ptr_) {
    r.ptr_ = nullptr;
  }
  template<typename U>
  scoped_refptr(scoped_refptr<U>&& r) noexcept : ptr_(r.get()) {
    r.reset();
  }

  // Assignment operators
  scoped_refptr& operator=(const scoped_refptr& r) noexcept {
    ptr_ = r.ptr_;
    return *this;
  }
  template<typename U>
  scoped_refptr& operator=(const scoped_refptr<U>& r) noexcept {
    ptr_ = r.get();
    return *this;
  }

  scoped_refptr& operator=(scoped_refptr&& r) noexcept {
    ptr_ = r.ptr_;
    r.ptr_ = nullptr;
    return *this;
  }
  template<typename U>
  scoped_refptr& operator=(scoped_refptr<U>&& r) noexcept {
    ptr_ = r.get();
    r.reset();
    return *this;
  }

  void reset(T* p = nullptr) noexcept {
    ptr_ = p;
  }

  void swap(scoped_refptr& r) noexcept {
    T* tmp = ptr_;
    ptr_ = r.ptr_;
    r.ptr_ = tmp;
  }

  // Observers
  T* get() const noexcept { return ptr_; }
  T& operator*() const noexcept { return *ptr_; }
  T* operator->() const noexcept { return ptr_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  ~scoped_refptr() { }

private:
  element_type* ptr_;
};

// Non-member functions
template<class T>
void swap(scoped_refptr<T>& a, scoped_refptr<T>& b) noexcept {
    a.swap(b);
}

template<class T, class U>
bool operator==(const scoped_refptr<T>& a, const scoped_refptr<U>& b) noexcept {
    return a.get() == b.get();
}

template<class T>
bool operator==(const scoped_refptr<T>& a, nullptr_t) noexcept {
    return !a;
}

template <typename T, typename... Args>
scoped_refptr<T> MakeRefCounted(Args&&... args) {
  return scoped_refptr<T>(new T(args...));
}

}  // namespace base

struct Base {
  Base() {}
  Base(int, int) {}
};

struct Derived : public Base {
  Derived() {}
  Derived(int, int) : Base(0, 0) {}
};

void basic() {
  // Direct constructor calls - not covered by this check
  base::scoped_refptr<int> P1(new int());
  base::scoped_refptr<Base> basePtr(new Base());

  // Reset calls
  P1.reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use base::MakeRefCounted instead
  // CHECK-FIXES: P1 = base::MakeRefCounted<int>();

  basePtr.reset(new Derived());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use base::MakeRefCounted instead
  // CHECK-FIXES: basePtr = base::MakeRefCounted<Derived>();
}

base::scoped_refptr<Base> factory() {
  return base::scoped_refptr<Base>(new Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use base::MakeRefCounted instead
  // CHECK-FIXES: return base::MakeRefCounted<Base>();
}
