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

template<typename Target, typename Source>
inline Target* dynamicDowncast(Source* source)
{
    return static_cast<Target*>(source);
}

void foo(OtherObject* other)
{
    dynamicDowncast<SubDerived>(other->obj());
}
