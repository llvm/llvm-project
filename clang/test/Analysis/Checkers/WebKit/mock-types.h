#ifndef mock_types_1103988513531
#define mock_types_1103988513531

template <typename T> struct Ref {
  T *t;

  Ref() : t{} {};
  Ref(T &t)
    : t(t) {
    if (t)
      t->ref();
  }
  ~Ref() {
    if (t)
      t->deref();
  }
  T *get() { return t; }
  T *ptr() { return t; }
  operator const T &() const { return *t; }
  operator T &() { return *t; }
};

template <typename T> struct RefPtr {
  T *t;

  RefPtr() : t(new T) {}
  RefPtr(T *t)
    : t(t) {
    if (t)
      t->ref();
  }
  ~RefPtr() {
    if (t)
      t->deref();
  }
  T *get() { return t; }
  T *operator->() { return t; }
  const T *operator->() const { return t; }
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

template <typename T, typename S>
struct TypeCastTraits {
  static bool isOfType(S&);
};

// Type checking function, to use before casting with downcast<>().
template <typename T, typename S>
inline bool is(const S &source) {
    return TypeCastTraits<const T, const S>::isOfType(source);
}

template <typename T, typename S>
inline bool is(S *source) {
    return source && TypeCastTraits<const T, const S>::isOfType(*source);
}

template <typename T, typename S> T *downcast(S *t) { return static_cast<T*>(t); }
template <typename T, typename S> T *dynamicDowncast(S &t) {
  return is<T>(t) ? &static_cast<T&>(t) : nullptr;
}
template <typename T, typename S> T *dynamicDowncast(S *t) {
  return is<T>(t) ? static_cast<T*>(t) : nullptr;
}

#endif
