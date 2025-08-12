// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.MemoryUnsafeCastChecker -verify %s

class Base { };
class Derived : public Base { };

template<typename Target, typename Source>
Target& downcast_ref(Source& source){
  [[clang::suppress]]
  return static_cast<Target&>(source);
}

template<typename Target, typename Source>
Target* downcast_ptr(Source* source){
  [[clang::suppress]]
  return static_cast<Target*>(source);
}

void test_pointers(Base *base) {
  Derived *derived_static = static_cast<Derived*>(base);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Derived *derived_reinterpret = reinterpret_cast<Derived*>(base);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Derived *derived_c = (Derived*)base;
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Derived *derived_d = downcast_ptr<Derived, Base>(base);  // no warning
}

void test_non_pointers(Derived derived) {
  Base base_static = static_cast<Base>(derived);  // no warning
}

void test_refs(Base &base) {
  Derived &derived_static = static_cast<Derived&>(base);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Derived &derived_reinterpret = reinterpret_cast<Derived&>(base);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Derived &derived_c = (Derived&)base;
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Derived &derived_d = downcast_ref<Derived, Base>(base);  // no warning
}

class BaseVirtual {
  virtual void virtual_base_function();
};

class DerivedVirtual : public BaseVirtual {
  void virtual_base_function() override { }
};

void test_dynamic_casts(BaseVirtual *base_ptr, BaseVirtual &base_ref) {
  DerivedVirtual *derived_dynamic_ptr = dynamic_cast<DerivedVirtual*>(base_ptr);
  // expected-warning@-1{{Unsafe cast from base type 'BaseVirtual' to derived type 'DerivedVirtual'}}
  DerivedVirtual &derived_dynamic_ref = dynamic_cast<DerivedVirtual&>(base_ref);
  // expected-warning@-1{{Unsafe cast from base type 'BaseVirtual' to derived type 'DerivedVirtual'}}
}

struct BaseStruct { };
struct DerivedStruct : BaseStruct { };

void test_struct_pointers(struct BaseStruct *base_struct) {
  struct DerivedStruct *derived_static = static_cast<struct DerivedStruct*>(base_struct);
  // expected-warning@-1{{Unsafe cast from base type 'BaseStruct' to derived type 'DerivedStruct'}}
  struct DerivedStruct *derived_reinterpret = reinterpret_cast<struct DerivedStruct*>(base_struct);
  // expected-warning@-1{{Unsafe cast from base type 'BaseStruct' to derived type 'DerivedStruct'}}
  struct DerivedStruct *derived_c = (struct DerivedStruct*)base_struct;
  // expected-warning@-1{{Unsafe cast from base type 'BaseStruct' to derived type 'DerivedStruct'}}
}

typedef struct BaseStruct BStruct;
typedef struct DerivedStruct DStruct;

void test_struct_refs(BStruct &base_struct) {
  DStruct &derived_static = static_cast<DStruct&>(base_struct);
  // expected-warning@-1{{Unsafe cast from base type 'BaseStruct' to derived type 'DerivedStruct'}}
  DStruct &derived_reinterpret = reinterpret_cast<DStruct&>(base_struct);
  // expected-warning@-1{{Unsafe cast from base type 'BaseStruct' to derived type 'DerivedStruct'}}
  DStruct &derived_c = (DStruct&)base_struct;
  // expected-warning@-1{{Unsafe cast from base type 'BaseStruct' to derived type 'DerivedStruct'}}
}

int counter = 0;
void test_recursive(BStruct &base_struct) {
  if (counter == 5)
    return;
  counter++;
  DStruct &derived_static = static_cast<DStruct&>(base_struct);
  // expected-warning@-1{{Unsafe cast from base type 'BaseStruct' to derived type 'DerivedStruct'}}
}

template<typename T>
class BaseTemplate { };

template<typename T>
class DerivedTemplate : public BaseTemplate<T> { };

void test_templates(BaseTemplate<int> *base, BaseTemplate<int> &base_ref) {
  DerivedTemplate<int> *derived_static = static_cast<DerivedTemplate<int>*>(base);
  // expected-warning@-1{{Unsafe cast from base type 'BaseTemplate' to derived type 'DerivedTemplate'}}
  DerivedTemplate<int> *derived_reinterpret = reinterpret_cast<DerivedTemplate<int>*>(base);
  // expected-warning@-1{{Unsafe cast from base type 'BaseTemplate' to derived type 'DerivedTemplate'}}
  DerivedTemplate<int> *derived_c = (DerivedTemplate<int>*)base;
  // expected-warning@-1{{Unsafe cast from base type 'BaseTemplate' to derived type 'DerivedTemplate'}}
  DerivedTemplate<int> &derived_static_ref = static_cast<DerivedTemplate<int>&>(base_ref);
  // expected-warning@-1{{Unsafe cast from base type 'BaseTemplate' to derived type 'DerivedTemplate'}}
  DerivedTemplate<int> &derived_reinterpret_ref = reinterpret_cast<DerivedTemplate<int>&>(base_ref);
  // expected-warning@-1{{Unsafe cast from base type 'BaseTemplate' to derived type 'DerivedTemplate'}}
  DerivedTemplate<int> &derived_c_ref = (DerivedTemplate<int>&)base_ref;
  // expected-warning@-1{{Unsafe cast from base type 'BaseTemplate' to derived type 'DerivedTemplate'}}
}

#define CAST_MACRO_STATIC(X,Y) (static_cast<Y>(X))
#define CAST_MACRO_REINTERPRET(X,Y) (reinterpret_cast<Y>(X))
#define CAST_MACRO_C(X,Y) ((Y)X)

void test_macro_static(Base *base, Derived *derived, Base &base_ref) {
  Derived *derived_static = CAST_MACRO_STATIC(base, Derived*);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Derived &derived_static_ref = CAST_MACRO_STATIC(base_ref, Derived&);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Base *base_static_same = CAST_MACRO_STATIC(base, Base*);  // no warning
  Base *base_static_upcast = CAST_MACRO_STATIC(derived, Base*);  // no warning
}

void test_macro_reinterpret(Base *base, Derived *derived, Base &base_ref) {
  Derived *derived_reinterpret = CAST_MACRO_REINTERPRET(base, Derived*);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Derived &derived_reinterpret_ref = CAST_MACRO_REINTERPRET(base_ref, Derived&);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Base *base_reinterpret_same = CAST_MACRO_REINTERPRET(base, Base*);  // no warning
  Base *base_reinterpret_upcast = CAST_MACRO_REINTERPRET(derived, Base*);  // no warning
}

void test_macro_c(Base *base, Derived *derived, Base &base_ref) {
  Derived *derived_c = CAST_MACRO_C(base, Derived*);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Derived &derived_c_ref = CAST_MACRO_C(base_ref, Derived&);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  Base *base_c_same = CAST_MACRO_C(base, Base*);  // no warning
  Base *base_c_upcast = CAST_MACRO_C(derived, Base*);  // no warning
}

struct BaseStructCpp {
  int t;
  void increment() { t++; }
};
struct DerivedStructCpp : BaseStructCpp {
  void increment_t() {increment();}
};

void test_struct_cpp_pointers(struct BaseStructCpp *base_struct) {
  struct DerivedStructCpp *derived_static = static_cast<struct DerivedStructCpp*>(base_struct);
  // expected-warning@-1{{Unsafe cast from base type 'BaseStructCpp' to derived type 'DerivedStructCpp'}}
  struct DerivedStructCpp *derived_reinterpret = reinterpret_cast<struct DerivedStructCpp*>(base_struct);
  // expected-warning@-1{{Unsafe cast from base type 'BaseStructCpp' to derived type 'DerivedStructCpp'}}
  struct DerivedStructCpp *derived_c = (struct DerivedStructCpp*)base_struct;
  // expected-warning@-1{{Unsafe cast from base type 'BaseStructCpp' to derived type 'DerivedStructCpp'}}
}

typedef struct BaseStructCpp BStructCpp;
typedef struct DerivedStructCpp DStructCpp;

void test_struct_cpp_refs(BStructCpp &base_struct, DStructCpp &derived_struct) {
  DStructCpp &derived_static = static_cast<DStructCpp&>(base_struct);
  // expected-warning@-1{{Unsafe cast from base type 'BaseStructCpp' to derived type 'DerivedStructCpp'}}
  DStructCpp &derived_reinterpret = reinterpret_cast<DStructCpp&>(base_struct);
  // expected-warning@-1{{Unsafe cast from base type 'BaseStructCpp' to derived type 'DerivedStructCpp'}}
  DStructCpp &derived_c = (DStructCpp&)base_struct;
  // expected-warning@-1{{Unsafe cast from base type 'BaseStructCpp' to derived type 'DerivedStructCpp'}}
  BStructCpp &base = (BStructCpp&)derived_struct; // no warning
  BStructCpp &base_static = static_cast<BStructCpp&>(derived_struct); // no warning
  BStructCpp &base_reinterpret = reinterpret_cast<BStructCpp&>(derived_struct); // no warning
}

struct stack_st { };

#define STACK_OF(type) struct stack_st_##type

void test_stack(stack_st *base) {
  STACK_OF(void) *derived = (STACK_OF(void)*)base;
  // expected-warning@-1{{Unsafe cast from type 'stack_st' to an unrelated type 'stack_st_void'}}
}

class Parent { };
class Child1 : public Parent { };
class Child2 : public Parent { };

void test_common_parent(Child1 *c1, Child2 *c2) {
  Child2 *c2_cstyle = (Child2 *)c1;
  // expected-warning@-1{{Unsafe cast from type 'Child1' to an unrelated type 'Child2'}}
  Child2 *c2_reinterpret = reinterpret_cast<Child2 *>(c1);
  // expected-warning@-1{{Unsafe cast from type 'Child1' to an unrelated type 'Child2'}}
}

class Type1 { };
class Type2 { };

void test_unrelated_ref(Type1 &t1, Type2 &t2) {
  Type2 &t2_cstyle = (Type2 &)t1;
  // expected-warning@-1{{Unsafe cast from type 'Type1' to an unrelated type 'Type2'}}
  Type2 &t2_reinterpret = reinterpret_cast<Type2 &>(t1);
  // expected-warning@-1{{Unsafe cast from type 'Type1' to an unrelated type 'Type2'}}
  Type2 &t2_same = reinterpret_cast<Type2 &>(t2); // no warning
}


class VirtualClass1 {
  virtual void virtual_base_function();
};

class VirtualClass2 {
  void virtual_base_function();
};

void test_unrelated_virtual(VirtualClass1 &v1) {
  VirtualClass2 &v2 = dynamic_cast<VirtualClass2 &>(v1);
  // expected-warning@-1{{Unsafe cast from type 'VirtualClass1' to an unrelated type 'VirtualClass2'}}
}

struct StructA { };
struct StructB { };

typedef struct StructA StA;
typedef struct StructB StB;

void test_struct_unrelated_refs(StA &a, StB &b) {
  StB &b_reinterpret = reinterpret_cast<StB&>(a);
  // expected-warning@-1{{Unsafe cast from type 'StructA' to an unrelated type 'StructB'}}
  StB &b_c = (StB&)a;
  // expected-warning@-1{{Unsafe cast from type 'StructA' to an unrelated type 'StructB'}}
  StA &a_local = (StA&)b;
  // expected-warning@-1{{Unsafe cast from type 'StructB' to an unrelated type 'StructA'}}
  StA &a_reinterpret = reinterpret_cast<StA&>(b);
  // expected-warning@-1{{Unsafe cast from type 'StructB' to an unrelated type 'StructA'}}
  StA &a_same = (StA&)a; // no warning
}

template<typename T>
class DeferrableRefCounted {
public:
  void deref() const {
    auto this_to_T =  static_cast<const T*>(this); // no warning
  }
};

class SomeArrayClass : public DeferrableRefCounted<SomeArrayClass> { };

void test_this_to_template(SomeArrayClass *ptr) {
  ptr->deref();
};

template<typename WeakPtrFactoryType>
class CanMakeWeakPtrBase {
public:
  void initializeWeakPtrFactory() const {
    auto &this_to_T = static_cast<const WeakPtrFactoryType&>(*this);
  }
};

template<typename T>
using CanMakeWeakPtr = CanMakeWeakPtrBase<T>;

class EventLoop : public CanMakeWeakPtr<EventLoop> { };

void test_this_to_template_ref(EventLoop *ptr) {
  ptr->initializeWeakPtrFactory();
};
