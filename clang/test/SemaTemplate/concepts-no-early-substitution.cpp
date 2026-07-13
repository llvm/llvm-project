// RUN: %clang_cc1 -std=c++20 -x c++ %s -verify -fsyntax-only
// expected-no-diagnostics

template <typename T0>
concept HasMemberBegin = requires(T0 t) { t.begin(); };

struct GetBegin {
  template <HasMemberBegin T1>
  void operator()(T1);
};

GetBegin begin;

template <typename T2>
concept Concept = requires(T2 t) { begin(t); };

struct Subrange;

template <typename T3>
struct View {
  Subrange &getSubrange();

  operator bool()
    requires true;

  operator bool()
    requires requires { begin(getSubrange()); };

  void begin();
};

struct Subrange : View<void> {};
static_assert(Concept<Subrange>);

namespace GH197597 {

template <class> struct iterator_traits;
template <class _Tp> struct iterator_traits<_Tp *> {
  typedef _Tp value_type;
};
template <class _Ip>
concept contiguous_iterator =
    requires { typename iterator_traits<_Ip>::value_type; };
template <class _CharT> class basic_format_context;
template <class _Tp, class _Context,
          class = _Context::template formatter_type<_Tp>>
concept __formattable_with = requires(_Context __fc) { __fc; };
template <class _Tp, class _CharT>
concept formattable = __formattable_with<_Tp, basic_format_context<_CharT>>;
template <class _CharT> struct __retarget_buffer {
  struct __iterator {
    __retarget_buffer *__buffer_;
  };
  template <contiguous_iterator _Iterator> void __transform(_Iterator);
};
template <class> struct formatter;
template <class _Context> struct __basic_format_arg_value {
  struct __handle {
    template <class _Tp>
    __handle(_Tp)
        : __format_([](_Context &__ctx) {
            typename _Context::template formatter_type<_Tp>{}.format(__ctx);
          }) {}
    void (*__format_)(_Context &);
  } __handle;
};
template <class _Args> struct __format_arg_store {
  __format_arg_store(_Args __args) {
    (void)__basic_format_arg_value<basic_format_context<char>>{__args};
  }
};
template <class _CharT> struct basic_format_context {
  template <class _Tp> using formatter_type = formatter<_Tp>;
  __retarget_buffer<_CharT>::__iterator out();
};
void __transform(auto __out_it) {
  char __transform___first;
  __out_it.__buffer_->__transform(&__transform___first);
}
struct __formatter_integer {
  template <class _FormatContext> void format(_FormatContext __ctx) {
    __transform(__ctx.out());
  }
};
template <> struct formatter<int> : __formatter_integer {};
template <formattable<char> T> void fmt(T);
void test() {
  fmt(0);
  __format_arg_store<int>{0};
}
} // namespace GH197597
