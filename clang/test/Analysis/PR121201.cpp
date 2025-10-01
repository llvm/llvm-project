// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s \
// RUN:    -analyzer-config unroll-loops=true

// expected-no-diagnostics

template <bool, typename T, typename> using conditional_t = T;
class basic_format_arg;
template <typename> struct formatter;

template <typename Context> struct value {
  template <typename T> value(T) {
    using value_type = T;
    (void)format_custom_arg<value_type,
                      typename Context::template formatter_type<value_type>>;
  }

  template <typename, typename Formatter> static void format_custom_arg() {
    Context ctx;
    auto f = Formatter();
    f.format(0, ctx);
  }
};

struct context {
  template <typename T> using formatter_type = formatter<T>;
};

enum { max_packed_args };

template <typename Context, long>
using arg_t = conditional_t<max_packed_args, value<Context>, basic_format_arg>;

template <int NUM_ARGS> struct format_arg_store {
  arg_t<context, NUM_ARGS> args;
};

template <typename... T, long NUM_ARGS = sizeof...(T)>
auto make_format_args(T... args) -> format_arg_store<NUM_ARGS> {
  return {args...};
}

template <typename F> void write_padded(F write) { write(0); }

template <typename... T> void format(T... args) { make_format_args(args...); }

template <int> struct bitset {
  bitset(long);
};

template <long N> struct formatter<bitset<N>> {
  struct writer {
    bitset<N> bs;

    template <typename OutputIt> void operator()(OutputIt) {
      for (auto pos = N; pos > 0; --pos) // no-crash
        ;
    }
  };

  template <typename FormatContext> void format(bitset<N> bs, FormatContext) {
    write_padded(writer{bs});
  }
};

bitset<6> TestBody_bs(2);

void TestBody() { format(TestBody_bs); }
