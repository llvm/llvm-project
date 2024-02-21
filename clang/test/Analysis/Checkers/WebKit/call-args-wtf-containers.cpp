// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

namespace WTF {

  template <typename T>
  class HashSet {
  public:
    template <typename U> T* find(U&) const;
    template <typename U> bool contains(U&) const;
    unsigned size() { return m_size; }
    template <typename U> void add(U&) const;
    template <typename U> void remove(U&) const;

  private:
    T* m_table { nullptr };
    unsigned m_size { 0 };
  };

  template <typename T, typename S>
  class HashMap {
  public:
    struct Item {
      T key;
      S value;
    };

    template <typename U> Item* find(U&) const;
    template <typename U> bool contains(U&) const;
    template <typename U> S* get(U&) const;
    template <typename U> S* inlineGet(U&) const;
    template <typename U> void add(U&) const;
    template <typename U> void remove(U&) const;

  private:
    Item* m_table { nullptr };
  };

  template <typename T>
  class WeakHashSet {
  public:
    template <typename U> T* find(U&) const;
    template <typename U> bool contains(U&) const;
    template <typename U> void add(U&) const;
    template <typename U> void remove(U&) const;
  };

  template <typename T>
  class Vector {
  public:
    unsigned size() { return m_size; }
    T& at(unsigned i) { return m_buffer[i]; }
    T& operator[](unsigned i) { return m_buffer[i]; }
    template <typename U> unsigned find(U&);
    template <typename U> unsigned reverseFind(U&);
    template <typename U> bool contains(U&);
    template <typename MatchFunction> unsigned findIf(const MatchFunction& match)
    {
      for (unsigned i = 0; i < m_size; ++i) {
        if (match(at(i)))
          return i;
      }
      return static_cast<unsigned>(-1);
    }
    template <typename MatchFunction> unsigned reverseFindIf(const MatchFunction& match)
    {
      for (unsigned i = 0; i < m_size; ++i) {
        if (match(at(m_size - i)))
          return i;
      }
      return static_cast<unsigned>(-1);
    }
    template <typename MatchFunction> bool containsIf(const MatchFunction& match)
    {
      for (unsigned i = 0; i < m_size; ++i) {
        if (match(at(m_size - i)))
          return true;
      }
      return false;
    }
    template <typename U> void append(U&) const;
    template <typename U> void remove(U&) const;

  private:
    T* m_buffer { nullptr };
    unsigned m_size { 0 };
  };

}

using WTF::HashSet;
using WTF::HashMap;
using WTF::WeakHashSet;
using WTF::Vector;

class RefCounted {
public:
  void ref() const;
  void deref() const;
};

RefCounted* object();

void test() {
  HashSet<RefPtr<RefCounted>> set;
  set.find(*object());
  set.contains(*object());
  set.add(*object());
  // expected-warning@-1{{Call argument is uncounted and unsafe}}
  set.remove(*object());
  // expected-warning@-1{{Call argument is uncounted and unsafe}}

  HashMap<Ref<RefCounted>, unsigned> map;
  map.find(*object());
  map.contains(*object());
  map.inlineGet(*object());
  map.add(*object());
  // expected-warning@-1{{Call argument is uncounted and unsafe}}
  map.remove(*object());
  // expected-warning@-1{{Call argument is uncounted and unsafe}}

  WeakHashSet<Ref<RefCounted>> weakSet;
  weakSet.find(*object());
  weakSet.contains(*object());
  weakSet.add(*object());
  // expected-warning@-1{{Call argument is uncounted and unsafe}}
  weakSet.remove(*object());
  // expected-warning@-1{{Call argument is uncounted and unsafe}}

  Vector<Ref<RefCounted>> vector;
  vector.at(0);
  vector[0];
  vector.find(*object());
  vector.reverseFind(*object());
  vector.contains(*object());
  vector.append(*object());
  // expected-warning@-1{{Call argument is uncounted and unsafe}}
  vector.remove(*object());
  // expected-warning@-1{{Call argument is uncounted and unsafe}}

  auto* obj = object();
  vector.findIf([&](Ref<RefCounted> key) { return key.ptr() == obj; });
  vector.reverseFindIf([&](Ref<RefCounted> key) { return key.ptr() == obj; });
  vector.containsIf([&](Ref<RefCounted> key) { return key.ptr() == obj; });
}