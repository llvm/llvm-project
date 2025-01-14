// RUN: %check_clang_tidy %s misc-const-correctness %t -- \
// RUN:   -config="{CheckOptions: {\
// RUN:     misc-const-correctness.AllowedTypes: '[Pp]ointer$;[Pp]tr$;[Rr]ef(erence)?$;qualified::Type;::fully::QualifiedType;ConstTemplate', \
// RUN:     misc-const-correctness.TransformPointersAsValues: true, \
// RUN:     misc-const-correctness.TransformReferences: true, \
// RUN:     misc-const-correctness.WarnPointersAsValues: true } \
// RUN:   }" -- -fno-delayed-template-parsing

struct SmartPointer {
};

struct smart_pointer {
};

struct SmartPtr {
};

struct smart_ptr {
};

struct SmartReference {
};

struct smart_reference {
};

struct SmartRef {
};

struct smart_ref {
};

struct OtherType {
};

template <typename T> struct ConstTemplate {
};

namespace qualified {
struct Type {
};
} // namespace qualified

namespace fully {
struct QualifiedType {
};
} // namespace fully

void negativeSmartPointer() {
  SmartPointer p1 = {};
  SmartPointer* p2 = {};
  SmartPointer& p3 = p1;
}

void negative_smart_pointer() {
  smart_pointer p1 = {};
  smart_pointer* p2 = {};
  smart_pointer& p3 = p1;
}

void negativeSmartPtr() {
  SmartPtr p1 = {};
  SmartPtr* p2 = {};
  SmartPtr& p3 = p1;
}

void negative_smart_ptr() {
  smart_ptr p1 = {};
  smart_ptr* p2 = {};
  smart_ptr& p3 = p1;
}

void negativeSmartReference() {
  SmartReference p1 = {};
  SmartReference* p2 = {};
  SmartReference& p3 = p1;
}

void negative_smart_reference() {
  smart_reference p1 = {};
  smart_reference* p2 = {};
  smart_reference& p3 = p1;
}

void negativeSmartRef() {
  SmartRef p1 = {};
  SmartRef* p2 = {};
  SmartRef& p3 = p1;
}

void negative_smart_ref() {
  smart_ref p1 = {};
  smart_ref* p2 = {};
  smart_ref& p3 = p1;
}

void positiveOtherType() {
  OtherType t = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 't' of type 'OtherType' can be declared 'const'
  // CHECK-FIXES: OtherType const t = {};
}

void negativeSomeComplex() {
   ConstTemplate<int> t1 = {};
   ConstTemplate<int>* t2 = {};
   ConstTemplate<int>& t3 = t1;
}

void negativeQualified() {
  qualified::Type t1 = {};
  qualified::Type* t2 = {};
  qualified::Type& t3 = t1;

  using qualified::Type;
  Type t4 = {};
  Type* t5 = {};
  Type& t6 = t4;
}

void negativeFullyQualified() {
  fully::QualifiedType t1 = {};
  fully::QualifiedType* t2 = {};
  fully::QualifiedType& t3 = t1;

  using fully::QualifiedType;
  QualifiedType t4 = {};
  QualifiedType* t5 = {};
  QualifiedType& t6 = t4;
}

using MySP = SmartPointer;
using MyTemplate = ConstTemplate<int>;
template <typename T> using MyTemplate2 = ConstTemplate<T>;

void positiveTypedefs() {
  MySP p1 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p1' of type 'MySP' (aka 'SmartPointer') can be declared 'const'
  // CHECK-FIXES: MySP const p1 = {};
  
  MySP* p2 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p2' of type 'MySP *' (aka 'SmartPointer *') can be declared 'const'
  // CHECK-FIXES: MySP* const p2 = {};

  MySP& p3 = p1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p3' of type 'MySP &' (aka 'SmartPointer &') can be declared 'const'
  // CHECK-FIXES: MySP const& p3 = p1;

  MyTemplate t1 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 't1' of type 'MyTemplate' (aka 'ConstTemplate<int>') can be declared 'const'
  // CHECK-FIXES: MyTemplate const t1 = {};

  MyTemplate* t2 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 't2' of type 'MyTemplate *' (aka 'ConstTemplate<int> *') can be declared 'const'
  // CHECK-FIXES: MyTemplate* const t2 = {};

  MyTemplate& t3 = t1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 't3' of type 'MyTemplate &' (aka 'ConstTemplate<int> &') can be declared 'const'
  // CHECK-FIXES: MyTemplate const& t3 = t1;

  MyTemplate2<int> t4 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 't4' of type 'MyTemplate2<int>' (aka 'ConstTemplate<int>') can be declared 'const'
  // CHECK-FIXES: MyTemplate2<int> const t4 = {};

  MyTemplate2<int>* t5 = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 't5' of type 'MyTemplate2<int> *' (aka 'ConstTemplate<int> *') can be declared 'const'
  // CHECK-FIXES: MyTemplate2<int>* const t5 = {};

  MyTemplate2<int>& t6 = t4;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 't6' of type 'MyTemplate2<int> &' (aka 'ConstTemplate<int> &') can be declared 'const'
  // CHECK-FIXES: MyTemplate2<int> const& t6 = t4;
}

template <typename T>
class Vector {};

void positiveSmartPtrWrapped() {
  Vector<SmartPtr> vec = {};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'vec' of type 'Vector<SmartPtr>' can be declared 'const'
  // CHECK-FIXES: Vector<SmartPtr> const vec = {};
}
