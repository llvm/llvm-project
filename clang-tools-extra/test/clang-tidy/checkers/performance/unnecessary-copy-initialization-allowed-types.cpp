// RUN: %check_clang_tidy %s performance-unnecessary-copy-initialization %t -- -config="{CheckOptions: [{key: performance-unnecessary-copy-initialization.AllowedTypes, value: '[Pp]ointer$;[Pp]tr$;[Rr]ef(erence)?$'}]}" --

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
  void constMethod() const;
};

template <typename T> struct SomeComplexTemplate {
  ~SomeComplexTemplate();
};

typedef SomeComplexTemplate<int> NotTooComplexRef;

const SmartPointer &getSmartPointer();
const smart_pointer &get_smart_pointer();
const SmartPtr &getSmartPtr();
const smart_ptr &get_smart_ptr();
const SmartReference &getSmartReference();
const smart_reference &get_smart_reference();
const SmartRef &getSmartRef();
const smart_ref &get_smart_ref();
const OtherType &getOtherType();
const NotTooComplexRef &getNotTooComplexRef();

void negativeSmartPointer() {
  const auto P = getSmartPointer();
}

void negative_smart_pointer() {
  const auto p = get_smart_pointer();
}

void negativeSmartPtr() {
  const auto P = getSmartPtr();
}

void negative_smart_ptr() {
  const auto p = get_smart_ptr();
}

void negativeSmartReference() {
  const auto R = getSmartReference();
}

void negative_smart_reference() {
  const auto r = get_smart_reference();
}

void negativeSmartRef() {
  const auto R = getSmartRef();
}

void negative_smart_ref() {
  const auto r = get_smart_ref();
}

void positiveOtherType() {
  const auto O = getOtherType();
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: the const qualified variable 'O' is copy-constructed from a const reference; consider making it a const reference [performance-unnecessary-copy-initialization]
  // CHECK-FIXES: const auto& O = getOtherType();
  O.constMethod();
}

void negativeNotTooComplexRef() {
  const NotTooComplexRef R = getNotTooComplexRef();
  // Using `auto` here would result in the "canonical" type which does not match
  // the pattern.
}
