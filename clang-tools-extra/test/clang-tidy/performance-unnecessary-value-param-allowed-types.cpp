// RUN: %check_clang_tidy %s performance-unnecessary-value-param %t -- -config="{CheckOptions: [{key: performance-unnecessary-value-param.AllowedTypes, value: '[Pp]ointer$;[Pp]tr$;[Rr]ef(erence)?$'}]}" --

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

void negativeSmartPointer(SmartPointer P) {
}

void negative_smart_pointer(smart_pointer p) {
}

void negativeSmartPtr(SmartPtr P) {
}

void negative_smart_ptr(smart_ptr p) {
}

void negativeSmartReference(SmartReference R) {
}

void negative_smart_reference(smart_reference r) {
}

void negativeSmartRef(SmartRef R) {
}

void negative_smart_ref(smart_ref r) {
}

void positiveOtherType(OtherType O) {
  // CHECK-MESSAGES: [[@LINE-1]]:34: warning: the parameter 'O' is copied for each invocation but only used as a const reference; consider making it a const reference [performance-unnecessary-value-param]
  // CHECK-FIXES: void positiveOtherType(const OtherType& O) {
}

void negativeNotTooComplexRef(NotTooComplexRef R) {
}
