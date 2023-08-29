
// This test case came up in the review of
// https://reviews.llvm.org/D159126
// when transforming `this` within a
// requires expression, we need to make sure
// the type of this (and its qualifiers) is respected.
namespace D159126 {

template <class _Tp>
concept __member_begin = requires(_Tp __t) {
  __t.begin();
};

struct {
  template <class _Tp>
  requires __member_begin<_Tp>
  auto operator()(_Tp &&) {}
} inline begin;

template <class>
concept range = requires {
  begin;
};

template <class _Tp>
concept __can_compare_begin = requires(_Tp __t) {
  begin(__t);
};

struct {
  template <__can_compare_begin _Tp> void operator()(_Tp &&);
} empty;

template <range _Rp> struct owning_view {
  _Rp __r_;
public:
  void empty() const requires requires { empty(__r_); };
};

template <class T>
concept HasEmpty = requires(T t) {
  t.empty();
};

struct ComparableIters {
    void begin();
};

static_assert(HasEmpty<owning_view<ComparableIters&>>);
static_assert(HasEmpty<owning_view<ComparableIters&&>>);
static_assert(!HasEmpty<owning_view<const ComparableIters&>>);
static_assert(!HasEmpty<owning_view<const ComparableIters&&>>);

}
