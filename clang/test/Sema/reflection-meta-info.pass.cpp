// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -std=c++26 -freflection -fblocks -fenable-matrix -fms-extensions -fexperimental-overflow-behavior-types -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -std=c++26 -freflection -fblocks -fenable-matrix -fms-extensions -fexperimental-overflow-behavior-types -fsyntax-only -verify -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -triple x86_64-pc-windows-msvc -std=c++26 -freflection -fblocks -fenable-matrix -fms-extensions -fexperimental-overflow-behavior-types -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple x86_64-pc-windows-msvc -std=c++26 -freflection -fblocks -fenable-matrix -fms-extensions -fexperimental-overflow-behavior-types -fsyntax-only -verify -fexperimental-new-constant-interpreter
// expected-no-diagnostics

typedef int int32_t;
using A = int;

using info = decltype(^^int);

template <auto R>
consteval auto f1() {
  return R;
}

template <decltype(^^int) R>
consteval auto f2() {
  return R;
}

template <info R>
consteval auto f3() {
  return R;
}

template <const auto R>
consteval auto f4() {
  return R;
}

template <const decltype(^^int) R>
consteval auto f5() {
  return R;
}

template <const info R>
consteval auto f6() {
  return R;
}

consteval void test()
{
    constexpr auto r = ^^int;
    constexpr auto q = ^^float;
    constexpr info v{};

    static_assert(__is_same(decltype(^^int), info));
    static_assert(__is_same(decltype(^^float), info));
    static_assert(__is_same(decltype(^^double), info));
    static_assert(__is_same(decltype(^^long), info));
    static_assert(__is_same(decltype(^^long long), info));
    static_assert(__is_same(decltype(^^short), info));
    static_assert(__is_same(decltype(^^char), info));
    static_assert(__is_same(decltype(^^unsigned char), info));
    static_assert(__is_same(decltype(^^unsigned short), info));
    static_assert(__is_same(decltype(^^unsigned int), info));
    static_assert(__is_same(decltype(^^unsigned long), info));
    static_assert(__is_same(decltype(^^unsigned long long), info));
    static_assert(__is_same(decltype(^^info), info));

    static_assert(__is_same(decltype(^^int), decltype(^^int)));
    static_assert(__is_same(decltype(^^int), decltype(^^float)));
    static_assert(__is_same(decltype(^^int), decltype(^^char)));
    static_assert(__is_same(decltype(^^double), decltype(^^float)));

    static_assert(!__is_same(decltype(^^int), int));
    static_assert(__is_scalar(info));

    static_assert(f1< ^^int >() == ^^int);
    static_assert(f1< ^^A >() != ^^int);
    static_assert(f1< ^^float>() != ^^int);

    static_assert(f2<r>() == ^^int);
    static_assert(f2<^^int>() == ^^int);
    static_assert(f2<^^float>() != ^^int);
    static_assert(f2<^^int>() == f1<^^int>());

    static_assert(f3<r>() == ^^int);
    static_assert(f3<^^int>() == ^^int);
    static_assert(f3<^^float>() != ^^int);
    static_assert(f3<^^int>() == f2<^^int>());

    static_assert(f4<r>() == ^^int);
    static_assert(f4<^^int>() == ^^int);
    static_assert(f4<^^float>() != ^^int);
    static_assert(f4<^^int>() == f3<^^int>());

    static_assert(f5<r>() == ^^int);
    static_assert(f5<^^int>() == ^^int);
    static_assert(f5<^^float>() != ^^int);
    static_assert(f5<^^int>() == f4<^^int>());

    static_assert(f6<r>() == ^^int);
    static_assert(f6<^^int>() == ^^int);
    static_assert(f6<^^float>() != ^^int);
    static_assert(f6<^^int>() == f5<^^int>());

    static_assert(sizeof(info) == sizeof(void*));
    static_assert(alignof(info) == sizeof(void*));
    static_assert(sizeof(decltype(^^int)) == sizeof(decltype(^^float)));
    static_assert(^^int32_t != ^^int);
    static_assert(^^const int != ^^int);
    static_assert(^^volatile int32_t == ^^volatile int);
    static_assert(^^const volatile int32_t == ^^const volatile int);
    static_assert(^^A != ^^int);
    static_assert(^^const A == ^^const int);


    static_assert(^^int == ^^int);
    static_assert(^^int != ^^float);
    static_assert(^^float != ^^int);
    static_assert(!(^^float == ^^int));
    static_assert(r != q);

    int a;
    static_assert(^^int == ^^decltype(a));

    using foo = const int;
    static_assert(^^foo != ^^const int);
    static_assert(^^const foo == ^^const int);
    static_assert(^^const int == ^^int const);
}
namespace StandardTypes {
struct Incomplete;
struct Record { int member; };
union Union { int member; };
enum Enum { enumerator };
enum class ScopedEnum {};
enum class ForwardEnum;
template <class T> struct ClassTemplate {};
template <class T> using AliasTemplate = ClassTemplate<T>;

static_assert(^^Incomplete == ^^struct Incomplete);
static_assert(^^Record == ^^struct Record);
static_assert(^^Union == ^^union Union);
static_assert(^^Enum == ^^enum Enum);
static_assert(^^Record != ^^Union);
static_assert(^^Enum != ^^ScopedEnum);
static_assert(^^ForwardEnum != ^^ScopedEnum);
static_assert(^^const ClassTemplate<int> == ^^const AliasTemplate<int>);
static_assert(^^ClassTemplate<int> != ^^ClassTemplate<float>);
static_assert(^^const Record != ^^Record);
static_assert(^^volatile Record != ^^const Record);
static_assert(^^Record * != ^^const Record *);
static_assert(^^int * != ^^int *const);
static_assert(^^int & != ^^int &&);
static_assert(^^int[3] == ^^int[1 + 2]);
static_assert(^^int[3] != ^^int[4]);
static_assert(^^int[] != ^^int[3]);
static_assert(^^int[2][3] != ^^int[3][2]);
static_assert(^^void(int) != ^^void(float));
static_assert(^^void(int, ...) != ^^void(int));
static_assert(^^void() noexcept != ^^void());
static_assert(^^int Record::* != ^^float Record::*);
static_assert(^^void (Record::*)() const != ^^void (Record::*)());
static_assert(^^void (Record::*)() & != ^^void (Record::*)() &&);
static_assert(^^void (*)(int) != ^^void(int));

using Pointer = int *;
using OtherPointer = Pointer;
using Array = int[3];
static_assert(^^Pointer != ^^int *);
static_assert(^^Pointer != ^^OtherPointer);
static_assert(^^const Pointer == ^^int *const);
static_assert(^^Array != ^^int[3]);
static_assert(^^const Array == ^^const int[3]);

Record object;
static_assert(^^decltype(object) == ^^Record);
Pointer pointer_object;
static_assert(^^decltype(pointer_object) == ^^int *);
static_assert(^^decltype(pointer_object) != ^^Pointer);
static_assert(^^decltype((object)) == ^^Record &);

template <info R> struct Reflected {};
static_assert(!__is_same(Reflected<^^int *>, Reflected<^^float *>));
static_assert(__is_same(Reflected<^^Record>, Reflected<^^struct Record>));
static_assert(!__is_same(Reflected<^^Pointer>, Reflected<^^int *>));

template <class T, int N> consteval bool dependent() {
  static_assert(__is_same(decltype(^^T), info));
  static_assert(__is_same(decltype(^^typename T::type), info));
  return (^^const typename T::type) == (^^const int) &&
         (^^int[N]) == (^^int[3]);
}
struct WithType { using type = int; };
static_assert(dependent<WithType, 3>());
static_assert(!dependent<WithType, 4>());
template <class T> consteval info member_type() {
  return ^^typename T::type;
}
static_assert(member_type<WithType>() == ^^int);
static_assert(member_type<WithType>() != ^^WithType::type);
static_assert(^^typename WithType::type == ^^int);
static_assert(^^WithType::type != ^^int);

template <class T> using DependentAlias = T *;
template <class T> consteval info alias_template() {
  return ^^const DependentAlias<T>;
}
static_assert(alias_template<int>() == ^^int *const);

template <info R> consteval info identity() { return R; }
template <class T> consteval info dependent_argument() {
  return identity<^^T *>();
}
static_assert(dependent_argument<int>() == ^^int *);
static_assert(dependent_argument<float>() == ^^float *);

// Reflecting a class template specialization does not require its definition.
template <class T> struct NotInstantiated {
  static_assert(sizeof(T) == 0);
};
constexpr auto incomplete_specialization = ^^NotInstantiated<int>;
static_assert(incomplete_specialization == ^^NotInstantiated<int>);

template <int... Ns> constexpr info array_reflections[] = {(^^int[Ns])...};
static_assert(array_reflections<1, 2>[0] == ^^int[1]);
static_assert(array_reflections<1, 2>[1] == ^^int[2]);

struct AliasMember { template <class T> using Alias = T; };
template <class T> consteval info dependent_alias_type() {
  return ^^typename T::template Alias<int>;
}
static_assert(dependent_alias_type<AliasMember>() == ^^int);
static_assert(^^typename AliasMember::template Alias<int> == ^^int);
static_assert(^^typename StandardTypes::AliasTemplate<int> == ^^ClassTemplate<int>);

namespace Names { struct Record {}; using Alias = int; }
namespace Imported { using Names::Record; using Names::Alias; }
static_assert(^^const Imported::Record == ^^const Names::Record);
static_assert(^^struct Imported::Record == ^^Names::Record);
static_assert(^^typename Imported::Alias == ^^int);
static_assert(^^Imported::Record * == ^^Names::Record *);
}

namespace ExtensionTypes {
static_assert(^^_Complex float == ^^float _Complex);
static_assert(^^_Complex float != ^^_Complex double);
static_assert(^^_Complex float != ^^float);
static_assert(^^_BitInt(17) != ^^_BitInt(18));
static_assert(^^_BitInt(17) != ^^unsigned _BitInt(17));
static_assert(^^__int128 != ^^long long);
static_assert(^^__bf16 != ^^float);
#ifndef _WIN32
static_assert(^^__float128 != ^^long double);
#endif
static_assert(^^__fp16 != ^^_Float16);
static_assert(^^int (^)(int) != ^^int (*)(int));

using Vector = int __attribute__((vector_size(16)));
using ExtVector = int __attribute__((ext_vector_type(4)));
using Matrix = float __attribute__((matrix_type(2, 3)));
using OtherMatrix = float __attribute__((matrix_type(3, 2)));
using AS = int __attribute__((address_space(1)));
using OtherAS = int __attribute__((address_space(2)));
static_assert(^^Vector != ^^ExtVector);
static_assert(^^const Vector == ^^const int __attribute__((vector_size(16))));
static_assert(^^const ExtVector == ^^const int __attribute__((ext_vector_type(4))));
static_assert(^^const Matrix == ^^const float __attribute__((matrix_type(2, 3))));
static_assert(^^const Matrix != ^^const OtherMatrix);
static_assert(^^const AS != ^^const OtherAS);
static_assert(^^const AS == ^^const int __attribute__((address_space(1))));
static_assert(^^AS * != ^^int *);
static_assert(^^int * __restrict != ^^int *);

static_assert(^^_Atomic(int) != ^^int);

Vector vector_value;
static_assert(^^__typeof__(vector_value) ==
              ^^int __attribute__((vector_size(16))));
static_assert(^^__typeof__(vector_value) != ^^Vector);
int value;
static_assert(^^__typeof__(value) == ^^int);
static_assert(^^__typeof_unqual__(value) == ^^int);

template <int N> consteval auto bitint() { return ^^_BitInt(N); }
static_assert(bitint<17>() == ^^_BitInt(17));
static_assert(bitint<18>() != ^^_BitInt(17));
template <class T, int N> consteval auto vector() {
  return ^^T __attribute__((ext_vector_type(N)));
}
static_assert(vector<int, 4>() == ^^int __attribute__((ext_vector_type(4))));
template <class T, int R, int C> consteval auto matrix() {
  return ^^T __attribute__((matrix_type(R, C)));
}
static_assert(matrix<float, 2, 3>() == ^^float __attribute__((matrix_type(2, 3))));
}

namespace MicrosoftTypes {
static_assert(^^int * __ptr32 != ^^int * __ptr64);
static_assert(^^int * __ptr32 __sptr != ^^int * __ptr32 __uptr);
static_assert(^^__unaligned int * != ^^int *);
#ifdef _WIN32
static_assert(^^void (__stdcall *)(int) == ^^void (__stdcall *)(int));
#endif
using Pointer = int * __ptr32;
static_assert(^^const Pointer == ^^int *const __ptr32);
template <class T> consteval auto reflect() { return ^^T; }
static_assert(reflect<Pointer>() == ^^int * __ptr32);
}

namespace Weird {
struct Record {};
using Alias = Record;
template <unsigned I, class... Ts> consteval info selected() { return ^^Ts...[I]; }
static_assert(selected<0, Record, int>() == ^^Record);
static_assert(selected<1, Record, int>() == ^^int);
static_assert(selected<0, Alias>() == ^^Record);
template <class... Ts> consteval info first_pointer() { return ^^Ts...[0] *; }
static_assert(first_pointer<Record, int>() == ^^Record *);
template <class... Ts> consteval info function_pack() { return ^^void(Ts...); }
static_assert(function_pack<int, Record>() == ^^void(int, Record));
template <class T> struct Injected { static constexpr info value = ^^Injected; };
static_assert(Injected<int>::value == ^^Injected<int>);
template <class T> consteval info decltype_dependent() { return ^^decltype(T{}); }
static_assert(decltype_dependent<Record>() == ^^Record);
template <class T> consteval info typeof_dependent(T value) { return ^^__typeof__(value); }
static_assert(typeof_dependent(Record{}) == ^^Record);
static_assert(^^__typeof__(Record) == ^^Record);
enum class E : unsigned short {};
static_assert(^^__underlying_type(E) == ^^unsigned short);
template <class T> consteval info underlying() { return ^^__underlying_type(T); }
static_assert(underlying<E>() == ^^unsigned short);
static_assert(^^__remove_reference_t(int &) == ^^int);
static_assert(^^__decay(int[3]) == ^^int *);
template <class T, unsigned N> consteval info gnu_vector() {return ^^T __attribute__((vector_size(N)));}
static_assert(gnu_vector<int, 16>() == ^^int __attribute__((vector_size(16))));
template <class T, unsigned N> consteval info space() { return ^^T __attribute__((address_space(N))); }
static_assert(space<int, 1>() == ^^int __attribute__((address_space(1))));
static_assert(^^int * _Nullable == ^^int *);
static_assert(^^int __ob_wrap != ^^int __ob_trap);

template <class... Ts> struct List {};
template <class... Ts> consteval info deduplicated_function() {
  return ^^List<__builtin_dedup_pack<Ts...>...>;
}
static_assert(deduplicated_function<int, Record, int>() == ^^List<int, Record>);
void adjusted(int values[3], void callback()) {
  static_assert(^^decltype(values) == ^^int *);
  static_assert(^^decltype(callback) == ^^void (*)());
}
template <class T> struct Deduced { T value; };
Deduced deduced{3};
static_assert(^^decltype(deduced) == ^^Deduced<int>);
static_assert(^^int [[clang::annotate_type("note")]] == ^^int);

// Pack indexing produces types, not aliases.
static_assert(selected<0, const int>() == ^^const int);
static_assert(selected<0, int &>() == ^^int &);
static_assert(selected<0, Alias>() != ^^Alias);
static_assert(^^int (*)[3] != ^^int *[3]);

// Builtin templates produce types, not aliases.
static_assert(^^__type_pack_element<0, Alias> == ^^Record);
template <unsigned I, class... Ts> consteval info builtin_selected() {
  return ^^__type_pack_element<I, Ts...>;
}
static_assert(builtin_selected<0, Alias>() != ^^Alias);

template <class T> consteval info remove_reference() {
  return ^^__remove_reference_t(T);
}
static_assert(remove_reference<Record &>() == ^^Record);
template <class T> consteval info typeof_type() { return ^^__typeof__(T); }
static_assert(typeof_type<Record>() == ^^Record);

template <class T> struct Base { using type = T; };
template <class T> struct Inherited : Base<T> {
  using typename Base<T>::type;
  static constexpr info value = ^^const type;
  // Substitution preserves typename on the resulting UsingType operand.
  static constexpr info type_value = ^^typename Inherited::type;
};
static_assert(Inherited<Record>::value == ^^const Record);
static_assert(Inherited<Record>::type_value == ^^Record);

static_assert(^^decltype([] {}) != ^^decltype([] {}));
}

static_assert(f1<info{}>() == info{});

namespace AliasTemplates {
template <class T> using A = T;
template <class T> using B = T;
using Int = int;
using OrdinaryAlias = A<int>;
static_assert(^^A<int> == ^^AliasTemplates::A<int>);
namespace Imported {
using AliasTemplates::A;
// A simple-template-id is a type-id, so [expr.reflect]/5.1's restriction on
// reflection-names found through using-declarations does not apply.
static_assert(^^A<int> == ^^AliasTemplates::A<int>);
static_assert(^^const A<int> == ^^const int);
}
static_assert(^^Imported::A<int> == ^^A<int>);
static_assert(^^A<Int> == ^^A<int>);
static_assert(^^A<int> != ^^int);
static_assert(^^A<int> != ^^A<float>);
static_assert(^^A<int> != ^^A<const int>);
static_assert(^^A<int> != ^^B<int>);
static_assert(^^A<int> != ^^OrdinaryAlias);
static_assert(^^typename AliasTemplates::A<int> == ^^int);
static_assert(^^const A<int> == ^^const int);
static_assert(^^A<int> * == ^^int *);
static_assert(^^decltype(static_cast<A<int>>(0)) == ^^int);

template <info R> using Reflection = int;
static_assert(^^Reflection<^^A<Int>> == ^^Reflection<^^A<int>>);
static_assert(^^Reflection<^^A<int>> != ^^Reflection<^^int>);

template <class T = int, class U = T> using Defaults = T;
static_assert(^^Defaults<> == ^^Defaults<int>);
static_assert(^^Defaults<int> == ^^Defaults<Int, int>);
static_assert(^^Defaults<int, int> != ^^Defaults<int, float>);
template <class T = int> using Ignored = int;
static_assert(^^Ignored<int> != ^^Ignored<float>);

template <unsigned N, unsigned M = N + 1> using Numbers = int;
static_assert(^^Numbers<1> == ^^Numbers<1, 2>);
static_assert(^^Numbers<1 + 1> == ^^Numbers<2>);
static_assert(^^Numbers<2, 3> == ^^Numbers<2u, 3u>);
static_assert(^^Numbers<1> != ^^Numbers<2>);
template <bool V> using Boolean = int;
static_assert(^^Boolean<1> == ^^Boolean<true>);
template <auto V> using Value = int;
static_assert(^^Value<1> != ^^Value<1u>);

int object;
template <int *P> using Pointer = int;
static_assert(^^Pointer<&object> != ^^Pointer<nullptr>);
static_assert(^^Pointer<nullptr> == ^^Pointer<static_cast<int *>(nullptr)>);
struct Key { int value; };
template <Key K> using Structural = int;
static_assert(^^Structural<Key{1}> == ^^Structural<Key{0 + 1}>);
static_assert(^^Structural<Key{1}> != ^^Structural<Key{2}>);

template <class... Ts> using Pack = int;
static_assert(^^Pack<A<int>, Int> == ^^Pack<int, int>);
static_assert(^^Pack<int, float> != ^^Pack<float, int>);
static_assert(^^Pack<> != ^^Pack<int>);
template <auto... Vs> using Values = int;
static_assert(^^Values<1 + 1, 3> == ^^Values<2, 3>);
static_assert(^^Values<1, 2> != ^^Values<1, 2u>);
template <template <class> class TT> using Template = int;
static_assert(^^Template<A> == ^^Template<AliasTemplates::A>);
static_assert(^^Template<A> != ^^Template<B>);

template <class T> struct Owner {
  template <class U> using Member = int;
  using Plain = int;
  static constexpr info plain = ^^Plain;
  static constexpr info specialized = ^^Member<int>;
};
static_assert(Owner<int>::plain != Owner<float>::plain);
static_assert(Owner<int>::specialized != Owner<float>::specialized);
static_assert(Owner<int>::plain == ^^Owner<int>::Plain);
static_assert(Owner<int>::specialized == ^^Owner<int>::Member<int>);
static_assert(^^Owner<int>::Member<float> != ^^Owner<int>::Member<int>);
static_assert(^^typename Owner<int>::Member<float> == ^^int);

template <info R> struct Reflected {};
static_assert(__is_same(Reflected<^^Numbers<1 + 1>>, Reflected<^^Numbers<2>>));
static_assert(!__is_same(Reflected<^^Ignored<int>>, Reflected<^^Ignored<float>>));
static_assert(!__is_same(Reflected<^^A<int>>, Reflected<^^int>));

template <class T> consteval info dependent() { return ^^A<T>; }
static_assert(dependent<int>() == ^^A<int>);
static_assert(dependent<float>() != ^^A<int>);
template <class T> consteval info ignored() { return ^^Ignored<T>; }
static_assert(ignored<int>() == ^^Ignored<int>);
static_assert(ignored<float>() != ^^Ignored<int>);
template <unsigned N> consteval info numbers() { return ^^Numbers<N>; }
static_assert(numbers<2>() == ^^Numbers<1 + 1>);
template <class... Ts> consteval info pack() { return ^^Pack<Ts...>; }
static_assert(pack<>() == ^^Pack<>);
static_assert(pack<Int, float>() == ^^Pack<int, float>);
template <template <class> class TT> consteval info substituted_template() {
  return ^^TT<int>;
}
static_assert(substituted_template<A>() == ^^A<int>);
static_assert(substituted_template<B>() != ^^A<int>);
static_assert(substituted_template<Owner>() == ^^Owner<int>);
template <template <class> class TT>
consteval info substituted_template_defaults() { return ^^TT<float>; }
static_assert(substituted_template_defaults<Defaults>() == ^^Defaults<float, float>);

// The template-template parameter's default overrides the alias's own default.
template <template <class, class = char> class TT>
consteval info substituted_parameter_default() { return ^^TT<float>; }
static_assert(substituted_parameter_default<Defaults>() == ^^Defaults<float, char>);

template <class T> requires (sizeof(T) > 1) using Constrained = int;
static_assert(^^Constrained<int> != ^^int);
template <class T> consteval int select()
  requires (^^Ignored<T> == ^^Ignored<int>) { return 1; }
template <class T> consteval int select()
  requires (^^Ignored<T> != ^^Ignored<int>) { return 2; }
static_assert(select<int>() == 1);
static_assert(select<float>() == 2);

template <class T> constexpr int profile(Reflected<^^A<T>>) { return 1; }
template <class T> constexpr int profile(Reflected<^^B<T>>) { return 2; }
static_assert(profile<int>(Reflected<^^A<int>>{}) == 1);
static_assert(profile<int>(Reflected<^^B<int>>{}) == 2);

template <class T, class... Ts> using MixedPack = int;
static_assert(^^MixedPack<int> != ^^MixedPack<int, int>);
template <class T, class... Ts> consteval info mixed_pack() {
  return ^^MixedPack<T, Ts...>;
}
static_assert(mixed_pack<Int, float, double>() == ^^MixedPack<int, float, double>);
template <auto... Vs> consteval info values() { return ^^Values<Vs...>; }
static_assert(values<1 + 1, 3>() == ^^Values<2, 3>);
template <template <class> class... TTs> using Templates = int;
static_assert(^^Templates<> != ^^Templates<A>);
static_assert(^^Templates<A, B> != ^^Templates<B, A>);
template <template <class> class... TTs> consteval info templates() {
  return ^^Templates<TTs...>;
}
static_assert(templates<>() == ^^Templates<>);
static_assert(templates<A, B>() == ^^Templates<A, B>);

template <int *P> consteval info pointer() { return ^^Pointer<P>; }
static_assert(pointer<&object>() == ^^Pointer<&object>);
static_assert(pointer<nullptr>() == ^^Pointer<nullptr>);
template <Key K> consteval info structural() { return ^^Structural<K>; }
static_assert(structural<Key{1}>() == ^^Structural<Key{0 + 1}>);
template <unsigned I, class... Ts> consteval info indexed() {
  return ^^Ignored<Ts...[I]>;
}
static_assert(indexed<1, int, float>() == ^^Ignored<float>);

template <class... Fs>
consteval bool check_lambdas(const info (&expected)[sizeof...(Fs)], Fs... fs) {
  info actual[] = {fs()...};
  for (unsigned I = 0; I != sizeof...(Fs); ++I)
    if (actual[I] != expected[I])
      return false;
  return true;
}

template <class... Ts> consteval bool expanded_lambdas() {
  constexpr info classes[] = {^^StandardTypes::ClassTemplate<Ts>...};
  constexpr info aliases[] = {^^Ignored<Ts>...};
  constexpr info types[] = {^^Ts...};
  constexpr info local_aliases[] = {[] {
    using Local = Ts;
    return ^^Local;
  }()...};
  static_assert(classes[0] == ^^StandardTypes::ClassTemplate<int>);
  static_assert(classes[1] == ^^StandardTypes::ClassTemplate<float>);
  static_assert(aliases[0] == ^^Ignored<int>);
  static_assert(aliases[1] == ^^Ignored<float>);
  static_assert(types[0] == ^^int);
  static_assert(types[1] == ^^float);
  static_assert(local_aliases[0] != local_aliases[1]);
  static_assert(local_aliases[0] != ^^int);
  return check_lambdas(classes,
                       [] { return ^^StandardTypes::ClassTemplate<Ts>; }...) &&
         check_lambdas(aliases, [] { return ^^Ignored<Ts>; }...) &&
         check_lambdas(types, [] { return ^^Ts; }...);
}
static_assert(expanded_lambdas<Int, float>());

}
