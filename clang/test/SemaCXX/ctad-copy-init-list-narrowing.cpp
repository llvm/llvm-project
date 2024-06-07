// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value -std=c++20 %s
namespace std
{
  typedef long unsigned int size_t;
}

namespace std
{
  template<class _E>
    class initializer_list
    {
    public:
      typedef _E value_type;
      typedef const _E& reference;
      typedef const _E& const_reference;
      typedef size_t size_type;
      typedef const _E* iterator;
      typedef const _E* const_iterator;

    private:
      iterator _M_array;
      size_type _M_len;


      constexpr initializer_list(const_iterator __a, size_type __l)
      : _M_array(__a), _M_len(__l) { }

    public:
      constexpr initializer_list() noexcept
      : _M_array(0), _M_len(0) { }


      constexpr size_type
      size() const noexcept { return _M_len; }


      constexpr const_iterator
      begin() const noexcept { return _M_array; }


      constexpr const_iterator
      end() const noexcept { return begin() + size(); }
    };

  template<class _Tp>
    constexpr const _Tp*
    begin(initializer_list<_Tp> __ils) noexcept
    { return __ils.begin(); }

  template<class _Tp>
    constexpr const _Tp*
    end(initializer_list<_Tp> __ils) noexcept
    { return __ils.end(); }
}

template<class T, class Y>
class pair{
    private:
    T fst;
    Y snd;
    public:
    pair(T f, Y s) : fst(f), snd(s) {}
};

template<class T, class Y>
class map {
    public:
    map(std::initializer_list<pair<T, Y>>, int a = 4, int b = 5) {}
};

template<class T, class Y>
class Contained {
  public:
  Contained(T, Y) {}
};

template<class T, class Y>
class A {
  public:
  A(std::initializer_list<Contained<T, Y> >, int) {}
};

int main() {
    map mOk ={pair{5, 'a'}, {6, 'b'}, {7, 'c'}};
    map mNarrow ={pair{5, 'a'}, {6.0f, 'b'}, {7, 'c'}}; // expected-error {{type 'float' cannot be narrowed to 'int' in initializer list}} // expected-note {{insert an explicit cast to silence this issue}}

    A aOk = {{Contained{5, 'c'}, {5, 'c'}}, 5};
    A aNarrowNested = {{Contained{5, 'c'}, {5.0f, 'c'}}, 5}; // expected-error {{type 'float' cannot be narrowed to 'int' in initializer list}} // expected-note {{insert an explicit cast to silence this issue}}
    A aNarrow = {{Contained{5, 'c'}, {5, 'c'}}, 5.0f}; // expected-error {{type 'float' cannot be narrowed to 'int' in initializer list}} // expected-note {{insert an explicit cast to silence this issue}}
}
