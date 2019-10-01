// RUN: %check_clang_tidy %s performance-for-range-copy %t -- \
// RUN:     -config="{CheckOptions: [{key: performance-for-range-copy.AllowedTypes, value: '[Pp]ointer$;[Pp]tr$;[Rr]ef(erence)?$'}]}" \
// RUN:     -- -fno-delayed-template-parsing

template <typename T>
struct Iterator {
  void operator++() {}
  const T& operator*() {
    static T* TT = new T();
    return *TT;
  }
  bool operator!=(const Iterator &) { return false; }
  typedef const T& const_reference;
};
template <typename T>
struct View {
  T begin() { return T(); }
  T begin() const { return T(); }
  T end() { return T(); }
  T end() const { return T(); }
  typedef typename T::const_reference const_reference;
};

struct SmartPointer {
  ~SmartPointer();
};

struct smart_pointer {
  ~smart_pointer();
};

struct SmartPtr {
  ~SmartPtr();
};

struct smart_ptr {
  ~smart_ptr();
};

struct SmartReference {
  ~SmartReference();
};

struct smart_reference {
  ~smart_reference();
};

struct SmartRef {
  ~SmartRef();
};

struct smart_ref {
  ~smart_ref();
};

struct OtherType {
  ~OtherType();
};

template <typename T> struct SomeComplexTemplate {
  ~SomeComplexTemplate();
};

typedef SomeComplexTemplate<int> NotTooComplexRef;

void negativeSmartPointer() {
  for (auto P : View<Iterator<SmartPointer>>()) {
    auto P2 = P;
  }
}

void negative_smart_pointer() {
  for (auto p : View<Iterator<smart_pointer>>()) {
    auto p2 = p;
  }
}

void negativeSmartPtr() {
  for (auto P : View<Iterator<SmartPtr>>()) {
    auto P2 = P;
  }
}

void negative_smart_ptr() {
  for (auto p : View<Iterator<smart_ptr>>()) {
    auto p2 = p;
  }
}

void negativeSmartReference() {
  for (auto R : View<Iterator<SmartReference>>()) {
    auto R2 = R;
  }
}

void negative_smart_reference() {
  for (auto r : View<Iterator<smart_reference>>()) {
    auto r2 = r;
  }
}

void negativeSmartRef() {
  for (auto R : View<Iterator<SmartRef>>()) {
    auto R2 = R;
  }
}

void negative_smart_ref() {
  for (auto r : View<Iterator<smart_ref>>()) {
    auto r2 = r;
  }
}

void positiveOtherType() {
  for (auto O : View<Iterator<OtherType>>()) {
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable is copied but only used as const reference; consider making it a const reference [performance-for-range-copy]
  // CHECK-FIXES: for (const auto& O : View<Iterator<OtherType>>()) {
    auto O2 = O;
  }
}

void negativeNotTooComplexRef() {
  for (NotTooComplexRef R : View<Iterator<NotTooComplexRef>>()) {
    auto R2 = R;
  }
}
