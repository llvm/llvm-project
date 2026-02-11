// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace t1 {
template <class T> struct VSX {
  ~VSX() { static_assert(sizeof(T) != 4, ""); } // expected-error {{static assertion failed due to requirement 'sizeof(int) != 4':}} \
                                                // expected-note {{expression evaluates to '4 != 4'}}
};
struct VS {
  union {
    VSX<int> _Tail;
  };
  ~VS() { }
  VS(short);
  VS();
};
VS::VS() : VS(0) { } // delegating constructors should not produce errors
VS::VS(short) : _Tail() { } // expected-note {{in instantiation of member function 't1::VSX<int>::~VSX' requested here}}
}


namespace t2 {
template <class T> struct VSX {
  ~VSX() { static_assert(sizeof(T) != 4, ""); } // expected-error {{static assertion failed due to requirement 'sizeof(int) != 4':}} \
                                                // expected-note {{expression evaluates to '4 != 4'}}
};
struct VS {
  union {
    struct {
      VSX<int> _Tail;
    };
  };
  ~VS() { }
  VS(short);
};
VS::VS(short) : _Tail() { } // expected-note {{in instantiation of member function 't2::VSX<int>::~VSX' requested here}}
}


namespace t3 {
template <class T> struct VSX {
  ~VSX() { static_assert(sizeof(T) != 4, ""); } // expected-error {{static assertion failed due to requirement 'sizeof(int) != 4':}} \
                                                // expected-note {{expression evaluates to '4 != 4'}}
};
union VS {
  VSX<int> _Tail;
  ~VS() { }
  VS(short);
};
VS::VS(short) : _Tail() { } // expected-note {{in instantiation of member function 't3::VSX<int>::~VSX' requested here}}
}
