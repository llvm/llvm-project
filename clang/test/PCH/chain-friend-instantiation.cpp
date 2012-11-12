// RUN: %clang_cc1 %s -ast-print -o - -chain-include %s -chain-include %s

#if !defined(PASS1)
#define PASS1

template <class T> class TClass;

namespace NS {
    template <class X, class Y> TClass<X> problematic(X * ptr, const TClass<Y> &src);

    template <class T>
    class TBaseClass
    {
    protected:
        template <class X, class Y> friend TClass<X> problematic(X * ptr, const TClass<Y> &src);
    };
}

template <class T>
class TClass: public NS::TBaseClass<T>
{
public:
    inline TClass() { }
};


namespace NS {
    template <class X, class T>
    TClass<X> problematic(X *ptr, const TClass<T> &src);
}

template <class X, class T>
TClass<X> unconst(const TClass<T> &src);

#elif !defined(PASS2)
#define PASS2

namespace std {
class s {};
}


typedef TClass<std::s> TStr;

struct crash {
  TStr str;

  crash(const TClass<std::s> p)
  {
    unconst<TStr>(p);
  }
};

#else

void f() {
    const TStr p;
    crash c(p);
}

#endif
