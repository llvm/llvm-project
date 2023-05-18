// RUN: %clang_cc1 -verify -fsyntax-only %s
// The diagnostics we produce for this code tickled a bug in raw_ostream.
template <typename _Alloc> class allocator;
template <class _CharT> struct char_traits;
template <typename _CharT, typename _Traits = char_traits<_CharT>,
          typename _Alloc = allocator<_CharT> >
class basic_string;
typedef basic_string<wchar_t> wstring;
class Closure {
};
template <class A1> class Callback1 {
};
template <class A1, class A2> class Callback2 {
};
template <class R, class A2> class ResultCallback1 {
};
template <bool del, class R, class T, class P1, class P2, class A1>
class AAAAAAAResultCallback_2_1 : public ResultCallback1<R, A1> {
};
template <bool del, class T, class P1, class P2, class A1>
class AAAAAAAResultCallback_2_1< del, void, T, P1, P2, A1> :
    public Callback1<A1> {
 public:
  typedef Callback1<A1> base;
};
template <class T1, class T2, class R, class P1, class P2, class A1>
inline typename AAAAAAAResultCallback_2_1<true, R, T1, P1, P2, A1>::base*
NewCallback(T1* obj, R(T2::* member)(P1, P2, A1), const P1& p1, const P2& p2) {}
namespace util { class Status {}; }
class xxxxxxxxxxxxxxxxx {
  void Bar(wstring* s, util::Status* status,
           Callback2<util::Status, wstring>* done);
  void Foo();
};
void xxxxxxxxxxxxxxxxx::Foo() {
  wstring* s = __null;
  util::Status* status = __null;
  Closure* cb = NewCallback(this, &xxxxxxxxxxxxxxxxx::Bar, s, status);  // expected-error{{cannot initialize}}
}
