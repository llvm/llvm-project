// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoDeleteChecker -verify %s

#include "mock-types.h"

void crash();

template<typename T, typename U>
[[clang::suppress]] inline void [[clang::annotate_type("webkit.nodelete")]] lazyInitialize(const RefPtr<T>& ptr, Ref<U>&& obj)
{
    if (ptr)
        crash();
    const_cast<RefPtr<T>&>(ptr) = WTF::move(obj);
}

struct RefObj {
  static Ref<RefObj> [[clang::annotate_type("webkit.nodelete")]] create(int = 0);
  void ref() const;
  void deref() const;
  int value() const;
};

struct Container {

  void [[clang::annotate_type("webkit.nodelete")]] foo() {
    if (!m_bar)
      lazyInitialize(m_bar, RefObj::create());
      // expected-warning@-1{{A function 'foo' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
      // The 'RefObj::create()' temporary is passed as an argument, so its
      // lifetime ends at this full-expression in 'foo' (the caller destroys
      // arguments) and its destructor may run delete. Only returned prvalues
      // are elided, so this is correctly flagged.
  }

  void [[clang::annotate_type("webkit.nodelete")]] bar() {
    if (!m_bar)
      lazyInitialize(m_bar, RefObj::create(RefObj::create()->value()));
      // expected-warning@-1{{A function 'bar' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  const RefPtr<RefObj> m_bar;
};
