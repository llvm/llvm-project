// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

class Base {
public:
    void ref();
    void deref();
    void doWork();
};

class Derived : public Base {
public:
  virtual ~Derived();

  void ref() const;
  void deref() const;
};

class SubDerived final : public Derived {
};

class OtherObject {
public:
    Derived* obj();
    Base* base();
};

class String {
};

template<typename Target, typename Source>
inline Target* dynamicDowncast(Source* source)
{
    return static_cast<Target*>(source);
}

template<typename Target, typename Source>
inline Target* checkedDowncast(Source* source)
{
    return static_cast<Target*>(source);
}

template<typename Target, typename Source>
inline Target* uncheckedDowncast(Source* source)
{
    return static_cast<Target*>(source);
}

template<typename Target, typename Source>
Target* [[clang::annotate_type("webkit.pointerconversion")]] newCastFunction(Source*);

template<typename Target, typename Source>
Target* [[clang::annotate_type("unrelated-annotation")]] badCastFunction(Source*);

template<typename... Types>
String toString(const Types&... values);

void foo(OtherObject* other)
{
    dynamicDowncast<SubDerived>(other->obj());
    checkedDowncast<SubDerived>(other->obj());
    uncheckedDowncast<SubDerived>(other->obj());
    newCastFunction<SubDerived>(other->obj());
    badCastFunction<SubDerived>(other->obj());
    // expected-warning@-1{{Call argument is uncounted and unsafe}}
    toString(other->obj());
}

struct SomeStruct {
  Derived* [[clang::annotate_type("webkit.pointerconversion")]] ptrConversion(Base*);

  void foo(OtherObject& otherObj) {
    RefPtr ptr = otherObj.base();
    ptrConversion(ptr.get())->doWork();
  }
};
