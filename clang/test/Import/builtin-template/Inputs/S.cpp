template <class T, T... I>
struct Seq {
  static constexpr T PackSize = sizeof...(I);
};

template <typename T, T N>
using MakeSeq = __make_integer_seq<Seq, T, N>;


using SizeT = decltype(sizeof(int));

template <SizeT i, typename ...T>
using TypePackElement = __type_pack_element<i, T...>;

template <int i>
struct X;

using X0 = X<0>;
template <int I>
using SameAsX = X<I>;

template <template <class...> class Templ, class...Types>
using TypePackDedup = Templ<__builtin_dedup_pack<Types...>...>;

template <class ...Ts>
struct TypeList {};
