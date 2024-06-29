// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

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

template<typename... Types>
String toString(const Types&... values);

void foo(OtherObject* other)
{
    dynamicDowncast<SubDerived>(other->obj());
    checkedDowncast<SubDerived>(other->obj());
    uncheckedDowncast<SubDerived>(other->obj());
    toString(other->obj());
}
