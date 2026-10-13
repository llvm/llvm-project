using info = decltype(^^int);

struct Record {};
using Alias = Record;
template <info R> struct Reflected {};
template <class T> constexpr info reflection = ^^T;
template <class T> consteval auto pointer() { return ^^T *; }
template <class... Ts> consteval bool pack() {
  return ((^^Ts == reflection<Ts>) && ...);
}
constexpr auto record = ^^Record;
constexpr auto array = ^^int[3];
constexpr auto function = ^^void(int);
constexpr auto alias = ^^Alias;
constexpr auto vector = ^^int __attribute__((ext_vector_type(4)));
constexpr auto block = ^^void (^)(int);
constexpr auto bitint = ^^_BitInt(17);

struct WithType { using type = int; };
template <class T> consteval info written_type() {
  return ^^typename T::type;
}
constexpr auto typename_type = ^^typename WithType::type;

template<unsigned I, class... Ts> consteval info selected_type() {
  return ^^Ts...[I];
}
template<class T>struct Injected {
  static constexpr info value = ^^Injected;
};

namespace AliasTemplates {
template <class T = int, class U = T> using Defaults = int;
template <int N, int M = N + 1> using Numbers = int;
template <class... Ts> using Pack = int;
constexpr info defaulted = ^^Defaults<>;
constexpr info number = ^^Numbers<1 + 1>;
constexpr info packed = ^^Pack<int, Record>;
template <info R> using Reflection = int;
constexpr info nested = ^^Reflection<defaulted>;
Reflected<nested> nested_argument;
template <class T> consteval info dependent() { return ^^Defaults<T>; }
template <class... Ts> consteval info pack() { return ^^Pack<Ts...>; }
Reflected<defaulted> defaulted_argument;
Reflected<number> number_argument;
Reflected<packed> pack_argument;
Reflected<alias> ordinary_alias_argument;
}

// Instantiate before serialization so the substituted reflection values
// in these function bodies are also written to the PCH.
template <info R> consteval info identity() { return R; }
static_assert(identity<^^int>() == ^^int);
static_assert(identity<info{}>() == info{});
static_assert(identity<alias>() == alias);
static_assert(identity<AliasTemplates::defaulted>() == AliasTemplates::defaulted);
