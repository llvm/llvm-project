#ifndef mock_types_1103988513531
#define mock_types_1103988513531

template <typename T> struct Ref {
  T *t;

  Ref() : t{} {};
  Ref(T *) {}
  T *get() { return t; }
  T *ptr() { return t; }
  operator const T &() const { return *t; }
  operator T &() { return *t; }
};

template <typename T> struct RefPtr {
  T *t;

  RefPtr() : t(new T) {}
  RefPtr(T *t) : t(t) {}
  T *get() { return t; }
  T *operator->() { return t; }
  T &operator*() { return *t; }
  RefPtr &operator=(T *) { return *this; }
  operator bool() { return t; }
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
};

template <typename T> T *downcast(T *t) { return t; }

#endif
