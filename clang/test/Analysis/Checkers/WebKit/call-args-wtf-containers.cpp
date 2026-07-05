// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

namespace WTF {

  constexpr unsigned long notFound = static_cast<unsigned long>(-1);

  class String;
  class StringImpl;

  class StringView {
  public:
    StringView(const String&);
  private:
    RefPtr<StringImpl> m_impl;
  };

  class StringImpl {
  public:
    void ref() const { ++m_refCount; }
    void deref() const {
      if (!--m_refCount)
        delete this;
    }

    static constexpr unsigned s_flagIs8Bit = 1u << 0;
    bool is8Bit() const { return m_hashAndFlags & s_flagIs8Bit; }
    const char* characters8() const { return m_char8; }
    const short* characters16() const { return m_char16; }
    unsigned length() const { return m_length; }
    Ref<StringImpl> substring(unsigned position, unsigned length) const;

    unsigned long find(char) const;
    unsigned long find(StringView) const;
    unsigned long contains(StringView) const;
    unsigned long findIgnoringASCIICase(StringView) const;

    bool startsWith(StringView) const;
    bool startsWithIgnoringASCIICase(StringView) const;
    bool endsWith(StringView) const;
    bool endsWithIgnoringASCIICase(StringView) const;

  private:
    mutable unsigned m_refCount { 0 };
    unsigned m_length { 0 };
    union {
      const char* m_char8;
      const short* m_char16;
    };
    unsigned m_hashAndFlags { 0 };
  };

  class String {
  public:
    String() = default;
    String(StringImpl& impl) : m_impl(&impl) { }
    String(StringImpl* impl) : m_impl(impl) { }
    String(Ref<StringImpl>&& impl) : m_impl(impl.get()) { }
    StringImpl* impl() { return m_impl.get(); }
    unsigned length() const { return m_impl ? m_impl->length() : 0; }
    const char* characters8() const { return m_impl ? m_impl->characters8() : nullptr; }
    const short* characters16() const { return m_impl ? m_impl->characters16() : nullptr; }

    bool is8Bit() const { return !m_impl || m_impl->is8Bit(); }

    unsigned long find(char character) const { return m_impl ? m_impl->find(character) : notFound; }
    unsigned long find(StringView str) const { return m_impl ? m_impl->find(str) : notFound; }
    unsigned long findIgnoringASCIICase(StringView) const;

    bool contains(char character) const { return find(character) != notFound; }
    bool contains(StringView) const;
    bool containsIgnoringASCIICase(StringView) const;

    bool startsWith(StringView) const;
    bool startsWithIgnoringASCIICase(StringView) const;
    bool endsWith(StringView) const;
    bool endsWithIgnoringASCIICase(StringView) const;

    String substring(unsigned position, unsigned length) const
    {
      if (!m_impl)
        return { };
      if (!position && length >= m_impl->length())
        return *this;
      return m_impl->substring(position, length);
    }

  private:
    RefPtr<StringImpl> m_impl;
  };

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

using WTF::StringView;
using WTF::StringImpl;
using WTF::String;
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
StringImpl* strImpl();
String* str();
StringView strView();

void test() {
  strImpl()->is8Bit();
  strImpl()->characters8();
  strImpl()->characters16();
  strImpl()->length();
  strImpl()->substring(2, 4);
  strImpl()->find(strView());
  strImpl()->contains(strView());
  strImpl()->findIgnoringASCIICase(strView());
  strImpl()->startsWith(strView());
  strImpl()->startsWithIgnoringASCIICase(strView());
  strImpl()->endsWith(strView());
  strImpl()->endsWithIgnoringASCIICase(strView());

  str()->is8Bit();
  str()->characters8();
  str()->characters16();
  str()->length();
  str()->substring(2, 4);
  str()->find(strView());
  str()->contains(strView());
  str()->findIgnoringASCIICase(strView());
  str()->startsWith(strView());
  str()->startsWithIgnoringASCIICase(strView());
  str()->endsWith(strView());
  str()->endsWithIgnoringASCIICase(strView());

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