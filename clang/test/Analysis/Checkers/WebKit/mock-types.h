#ifndef mock_types_1103988513531
#define mock_types_1103988513531

template<typename T>
struct RawPtrTraits {
  using StorageType = T*;

  template<typename U>
  static T* exchange(StorageType& ptr, U&& newValue)
  {
    StorageType oldValue = static_cast<StorageType&&>(ptr);
    ptr = static_cast<U&&>(newValue);
    return oldValue;
  }

  static void swap(StorageType& a, StorageType& b)
  {
    StorageType temp = static_cast<StorageType&&>(a);
    a = static_cast<StorageType&&>(b);
    b = static_cast<StorageType&&>(temp);
  }
  static T* unwrap(const StorageType& ptr) { return ptr; }
};

template<typename T> struct DefaultRefDerefTraits {
  static T* refIfNotNull(T* ptr)
  {
    if (ptr)
      ptr->ref();
    return ptr;
  }

  static T& ref(T& ref)
  {
    ref.ref();
    return ref;
  }

  static void derefIfNotNull(T* ptr)
  {
    if (ptr)
      ptr->deref();
  }
};

template <typename T, typename PtrTraits = RawPtrTraits<T>, typename RefDerefTraits = DefaultRefDerefTraits<T>> struct Ref {
  typename PtrTraits::StorageType t;

  Ref() : t{} {};
  Ref(T &t) : t(&RefDerefTraits::ref(t)) { }
  Ref(const Ref& o) : t(RefDerefTraits::refIfNotNull(PtrTraits::unwrap(o.t))) { }
  ~Ref() { RefDerefTraits::derefIfNotNull(PtrTraits::exchange(t, nullptr)); }
  T &get() { return *PtrTraits::unwrap(t); }
  T *ptr() { return PtrTraits::unwrap(t); }
  T *operator->() { return PtrTraits::unwrap(t); }
  operator const T &() const { return *PtrTraits::unwrap(t); }
  operator T &() { return *PtrTraits::unwrap(t); }
  T* leakRef() { return PtrTraits::exchange(t, nullptr); }
};

template <typename T> struct RefPtr {
  T *t;

  RefPtr() : t(new T) {}
  RefPtr(T *t)
    : t(t) {
    if (t)
      t->ref();
  }
  RefPtr(Ref<T>&& o)
    : t(o.leakRef())
  { }
  ~RefPtr() {
    if (t)
      t->deref();
  }
  T *get() { return t; }
  T *operator->() { return t; }
  const T *operator->() const { return t; }
  T &operator*() { return *t; }
  RefPtr &operator=(T *) { return *this; }
  operator bool() const { return t; }
};

template <typename T> bool operator==(const RefPtr<T> &, const RefPtr<T> &) {
  return false;
}

template <typename T> bool operator==(const RefPtr<T> &, T *) { return false; }

template <typename T> bool operator==(const RefPtr<T> &, T &) { return false; }

template <typename T> bool operator!=(const RefPtr<T> &, const RefPtr<T> &) {
  return false;
}

template <typename T> bool operator!=(const RefPtr<T> &, T *) { return false; }

template <typename T> bool operator!=(const RefPtr<T> &, T &) { return false; }

struct RefCountable {
  static Ref<RefCountable> create();
  void ref() {}
  void deref() {}
  void method();
  int trivial() { return 123; }
};

template <typename T> T *downcast(T *t) { return t; }

template <typename T> struct CheckedRef {
private:
  T *t;

public:
  CheckedRef() : t{} {};
  CheckedRef(T &t) : t(t) { t->incrementPtrCount(); }
  CheckedRef(const CheckedRef& o) : t(o.t) { if (t) t->incrementPtrCount(); }
  ~CheckedRef() { if (t) t->decrementPtrCount(); }
  T &get() { return *t; }
  T *ptr() { return t; }
  T *operator->() { return t; }
  operator const T &() const { return *t; }
  operator T &() { return *t; }
};

template <typename T> struct CheckedPtr {
private:
  T *t;

public:
  CheckedPtr() : t(nullptr) {}
  CheckedPtr(T *t)
    : t(t) {
    if (t)
      t->incrementPtrCount();
  }
  CheckedPtr(Ref<T>&& o)
    : t(o.leakRef())
  { }
  ~CheckedPtr() {
    if (t)
      t->decrementPtrCount();
  }
  T *get() { return t; }
  T *operator->() { return t; }
  const T *operator->() const { return t; }
  T &operator*() { return *t; }
  CheckedPtr &operator=(T *) { return *this; }
  operator bool() const { return t; }
};

class CheckedObj {
public:
  void incrementPtrCount();
  void decrementPtrCount();
};

#endif
