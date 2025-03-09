// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

class Base {
public:
    inline void ref();
    inline void deref();
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
};

class String {
};

template<typename Target, typename Source>
inline Target* dynamicDowncast(Source* source)
{
    return static_cast<Target*>(source);
    // expected-warning@-1 {{static downcast from 'Derived' to 'SubDerived'}}
}

template<typename Target, typename Source>
inline Target* checkedDowncast(Source* source)
{
    return static_cast<Target*>(source);
    // expected-warning@-1 {{static downcast from 'Derived' to 'SubDerived'}}
}

template<typename Target, typename Source>
inline Target* uncheckedDowncast(Source* source)
{
    return static_cast<Target*>(source);
    // expected-warning@-1 {{static downcast from 'Derived' to 'SubDerived'}}
}

template<typename... Types>
String toString(const Types&... values);

void foo(OtherObject* other)
{
    dynamicDowncast<SubDerived>(other->obj()); // expected-note {{in instantiation}}
    checkedDowncast<SubDerived>(other->obj()); // expected-note {{in instantiation}}
    uncheckedDowncast<SubDerived>(other->obj()); // expected-note {{in instantiation}}
    toString(other->obj());
}
