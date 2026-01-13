// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

template <int __v> struct integral_constant {
  static constexpr int value = __v;
};
template <bool _Val> using _BoolConstant = integral_constant<_Val>;
template <int, class> struct tuple_element;
template <class...> class tuple;
template <int _Ip, class... _Tp> struct tuple_element<_Ip, tuple<_Tp...>> {
  using type = __type_pack_element<_Ip, _Tp...>;
};
template <class> struct tuple_size;
template <bool> using __enable_if_t = int;
template <template <class> class _BaseType, class _Tp, _Tp _SequenceSize>
using __make_integer_sequence_impl =
    __make_integer_seq<_BaseType, _Tp, _SequenceSize>;
template <class _Tp, _Tp...> struct __integer_sequence;
template <int... _Indices>
using __index_sequence = __integer_sequence<int, _Indices...>;
template <int _SequenceSize>
using __make_index_sequence =
    __make_integer_sequence_impl<__integer_sequence, int, _SequenceSize>;
template <class _Tp, _Tp...> struct integer_sequence {};
template <int... _Ip> using index_sequence = integer_sequence<int, _Ip...>;
template <class _Tp, _Tp _Ep>
using make_integer_sequence =
    __make_integer_sequence_impl<integer_sequence, _Tp, _Ep>;
template <int _Np> using make_index_sequence = make_integer_sequence<int, _Np>;
enum __element_count : int;
constexpr void __constexpr_memmove(char *__dest, const char *__src,
                                   __element_count __n) {
  __builtin_memmove(__dest, __src, __n);
}
template <class _Tp> using __underlying_type_t = __underlying_type(_Tp);
template <class _Tp> using underlying_type_t = __underlying_type_t<_Tp>;
template <class _Tp, class> using __enable_if_tuple_size_imp = _Tp;
template <class _Tp>
struct tuple_size<__enable_if_tuple_size_imp<
    const _Tp, __enable_if_t<_BoolConstant<__is_volatile(int)>::value>>>
    : integral_constant<tuple_size<_Tp>::value> {};
template <class... _Tp>
struct tuple_size<tuple<_Tp...>> : integral_constant<sizeof...(_Tp)> {};
template <class _Tp> constexpr int tuple_size_v = tuple_size<_Tp>::value;
template <class _T1, class _T2> struct pair {
  _T1 first;
  _T2 second;
};
template <class _T1> constexpr pair<_T1, char *> make_pair(_T1, char *__t2) {
  return pair(_T1(), __t2);
}
template <int, class _Hp> struct __tuple_leaf {
  _Hp __value_;
  constexpr const _Hp &get() const { return __value_; }
};
template <class...> struct __tuple_impl;
template <int... _Indx, class... _Tp>
struct __tuple_impl<__index_sequence<_Indx...>, _Tp...>
    : __tuple_leaf<_Indx, _Tp>... {
  template <class... _Args>
  constexpr __tuple_impl(int, _Args... __args)
      : __tuple_leaf<_Indx, _Tp>(__args)... {}
};
template <class... _Tp> struct tuple {
  __tuple_impl<__make_index_sequence<sizeof...(_Tp)>, _Tp...> __base_;
  template <class... _Up> constexpr tuple(_Up... __u) : __base_({}, __u...) {}
};
template <int _Ip, class... _Tp>
constexpr const tuple_element<_Ip, tuple<_Tp...>>::type &
get(const tuple<_Tp...> &__t) noexcept {
  return static_cast<const __tuple_leaf<
      _Ip, typename tuple_element<_Ip, tuple<_Tp...>>::type> &>(__t.__base_)
      .get();
}
template <class... _Tp> constexpr tuple<_Tp...> make_tuple(_Tp... __t) {
  return tuple<_Tp...>(__t...);
}
constexpr int __char_traits_length_checked(const char *__s) {
  return __builtin_strlen(__s);
}
struct basic_string_view {
  constexpr basic_string_view() {}
  constexpr basic_string_view(const char *__s)
      : __data_(__s), __size_(__char_traits_length_checked(__s)) {}
  constexpr const char *begin() { return __data_; }
  constexpr const char *end() {
    return __data_ + __size_;
  }
  const char *__data_;
  int __size_;
};
template <class _Algorithm>
constexpr pair<const char *, char *>
__copy_move_unwrap_iters(const char *__first, const char *__last,
                         char *__out_first) {
  pair<const char *, const char *> __range = {__first, __last};
  auto __result = _Algorithm()(__range.first, __range.second, __out_first);
  return make_pair(__result.first, __result.second);
}
struct __copy_impl {
  constexpr pair<const char *, char *>
  operator()(const char *__first, const char *__last, char *__result) {
    const int __n(__last - __first);
    __constexpr_memmove(__result, __first, __element_count(__n));
    return make_pair(__last, __result);
  }
};
constexpr char *copy(const char *__first, const char *__last, char *__result) {
  return __copy_move_unwrap_iters<__copy_impl>(__first, __last, __result).second;
}
constexpr char *copy_n(const char *__first, int __orig_n, char *__result) {
  return copy(__first, __first + __orig_n, __result);
}
template <int _Size> struct array {
  basic_string_view __elems_[_Size];
  constexpr basic_string_view &operator[](int __n) { return __elems_[__n]; }
  constexpr basic_string_view operator[](int __n) const {
    return __elems_[__n];
  }
};

template <typename> struct FieldId;

template <FieldId field> constexpr auto FieldIdToInnerValue() {
  return field.template ToInnerValue<field>();
}
struct FieldNameEnum {
  enum class type;
};
template <int N> using FieldName = FieldNameEnum::type;
template <typename, auto> struct GetParentMessageAtIndexImpl;
template <typename, auto> struct FieldInfoHelper;
template <FieldId...> struct PathImpl;
template <int N> struct LongPathLiteral {
  consteval LongPathLiteral(const char (&s)[N]) {
    copy_n(s, N, long_path)[N] = field_count = long_path_size = 1;
  }
  consteval basic_string_view to_string_view() const { return long_path; }
  char long_path[N + 1];
  int long_path_size;
  int field_count;
};
template <LongPathLiteral kLongPath> consteval auto get_field_components() {
  basic_string_view long_path(kLongPath.to_string_view());
  array<kLongPath.field_count> ret;
  for (int i = 0; i < kLongPath.field_count; ++i)
    ret[i] = long_path;
  return ret;
}
template <LongPathLiteral kLongPath>
constexpr auto kFieldComponents = get_field_components<kLongPath>();
template <LongPathLiteral kLongPath> struct LongPathHelper {
  template <int... I>
  static PathImpl<kFieldComponents<kLongPath>[I]...>
      PathForLongPath(index_sequence<I...>);
  using type =
      decltype(PathForLongPath(make_index_sequence<kLongPath.field_count>{}));
};
template <typename T> struct PathFieldId {
  template <typename Arg> constexpr PathFieldId(Arg &arg) : value(arg) {}
  T value;
};
template <PathFieldId...> constexpr auto PathImplHelper();

template <int N> using FieldName = FieldName<N>;
enum class FieldNumber;
template <PathFieldId... fields>
constexpr auto Path = PathImplHelper<fields...>();
template <typename Proto, FieldId field>
using FieldInfo =
    FieldInfoHelper<Proto, FieldIdToInnerValue<field>()>::type;
template <> struct FieldId<FieldNameEnum::type> {
  constexpr FieldId(basic_string_view);
  int size;
  long hash;
  template <auto field> static constexpr auto ToInnerValue() {
    return static_cast<FieldNameEnum::type>(field.hash);
  }
};
FieldId(basic_string_view) -> FieldId<FieldNameEnum::type>;
template <typename Proto, FieldId field, int index>
using GetParentMessageAtIndex = GetParentMessageAtIndexImpl<
    Proto, FieldIdToInnerValue<field>()>::type;

template <typename T>
PathFieldId(T &t) -> PathFieldId<decltype(LongPathLiteral(t))>;
template <FieldId... fields1, FieldId... fields2>
constexpr PathImpl<fields1..., fields2...> *ConcatPath(PathImpl<fields1...> *,
                                                       PathImpl<fields2...> *) {
  return nullptr;
}
template <LongPathLiteral long_path_literal>
constexpr LongPathHelper<long_path_literal>::type *SinglePath() {
  return nullptr;
}
template <PathFieldId... fields> constexpr auto PathImplHelper() {
  return ConcatPath(SinglePath<fields.value>()...);
}
template <auto hash_prime, auto offset_bias>
constexpr auto Fnv1a(basic_string_view str) {
  auto hash = offset_bias;
  for (char c : str) {
    hash ^= c;
    hash *= hash_prime;
  }
  return hash;
}
constexpr auto HashField(basic_string_view str) {
  return Fnv1a<1099511628211u, 1039346656037>(str);
}
template <typename FI> struct FieldInfoValueTypeAlias : FI {};
template <typename Proto, auto field> struct FieldInfoHelperBase {
  static constexpr auto MaskFieldNameHash() {
    using FieldEnum = decltype(field);
    return FieldEnum{static_cast<underlying_type_t<FieldEnum>>(field) & 31};
  }
  using internal_type =
      Proto::template FieldInfoImpl<decltype(field), MaskFieldNameHash()>;
};
template <typename Proto, auto field> struct FieldInfoHelper {
  using type = FieldInfoValueTypeAlias<
      typename FieldInfoHelperBase<Proto, field>::internal_type>;
};

template <auto... fields>
struct FieldId<const PathImpl<fields...> *> {
  constexpr FieldId(PathImpl<fields...> *) : path() {}
  template <auto field> static constexpr auto ToInnerValue() {
    return field.path;
  }
  const PathImpl<fields...> *path;
};
template <auto... fields>
FieldId(PathImpl<fields...> *)
    -> FieldId<const PathImpl<fields...> *>;

template <auto> struct UnpackedField {
  static constexpr bool is_path = false;
};
template <auto... fields, const PathImpl<fields...> *path>
struct UnpackedField<path> {
  static constexpr auto value = make_tuple(fields...);
  static constexpr bool is_path = true;
};
template <typename Proto, FieldId... fields, const PathImpl<fields...> *path>
struct GetParentMessageAtIndexImpl<Proto, path> {
  using type = Proto;
};

constexpr FieldId<FieldNameEnum::type>::FieldId(basic_string_view str)
    : size(), hash(HashField(str)) {}
template <FieldId field> constexpr bool IsPath() {
  return UnpackedField<
      FieldIdToInnerValue<field>()>::is_path;
}
template <FieldId field> constexpr auto UnpackFieldToTuple() {
  return UnpackedField<FieldIdToInnerValue<field>()>::value;
}
template <int> struct CompileTimeString {
  consteval CompileTimeString(basic_string_view &v) : internal_view_(v) {}
  basic_string_view &internal_view_;
};
CompileTimeString(basic_string_view) -> CompileTimeString<0>;

template <CompileTimeString... parts> struct NameJoiner {
  template <CompileTimeString... after>
  NameJoiner<parts...> operator+(NameJoiner<after...>);
};
template <FieldId> struct FieldNameBuilder;
template <FieldId field>
  requires(!IsPath<field>())
struct FieldNameBuilder<field> {
  template <typename Proto> static auto Get() {
    return NameJoiner<FieldInfo<Proto, field>::name>();
  }
};
template <FieldId field>
  requires(IsPath<field>())
struct FieldNameBuilder<field> {
  static constexpr auto kTuple = UnpackFieldToTuple<field>();
  static constexpr int kTupleSize = tuple_size_v<decltype(kTuple)>;
  template <typename Proto, int... Is> static void Get(index_sequence<Is...>) {
    (FieldNameBuilder<get<Is>(
         kTuple)>::template Get<GetParentMessageAtIndex<Proto, field, Is>>() +
     ...);
  }
  template <typename Proto> static void Get() {
    Get<Proto>(make_index_sequence<kTupleSize>());
  }
};

struct T {
  template <typename FieldType, FieldType> struct FieldInfoImpl;
};
void AddPathsToFieldMask() {
  FieldNameBuilder<Path<"message_field", "int32_field">>::Get<T>();
}
template <> struct T::FieldInfoImpl<FieldNumber, FieldNumber{1}> {
  static basic_string_view name;
};
template <>
struct T::FieldInfoImpl<FieldName<1>, FieldName<1>{12}>
    : FieldInfoImpl<FieldNumber, FieldNumber{1}> {};
template <> struct T::FieldInfoImpl<FieldNumber, FieldNumber{10}> {
  static basic_string_view name;
};
template <>
struct T::FieldInfoImpl<FieldName<3>, FieldName<3>{11}>
    : FieldInfoImpl<FieldNumber, FieldNumber{10}> {};
