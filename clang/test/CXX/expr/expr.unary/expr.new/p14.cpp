// RUN: %clang_cc1 -std=c++1z -fsized-deallocation -fexceptions %s -verify

using size_t = decltype(sizeof(0));
namespace std { enum class align_val_t : size_t {}; }

struct Arg {} arg;

// If the type is aligned, first try with an alignment argument and then
// without. If not, never consider supplying an alignment.

template<unsigned Align, typename ...Ts>
struct alignas(Align) Unaligned {
  void *operator new(size_t, Ts...) = delete; // expected-note 4{{deleted}}
};
auto *ua = new Unaligned<__STDCPP_DEFAULT_NEW_ALIGNMENT__>; // expected-error {{deleted}}
auto *ub = new Unaligned<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2>; // expected-error {{deleted}}
auto *uap = new (arg) Unaligned<__STDCPP_DEFAULT_NEW_ALIGNMENT__, Arg>; // expected-error {{deleted}}
auto *ubp = new (arg) Unaligned<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2, Arg>; // expected-error {{deleted}}

template<unsigned Align, typename ...Ts>
struct alignas(Align) Aligned {
  void *operator new(size_t, std::align_val_t, Ts...) = delete; // expected-note 2{{deleted}} expected-note 2{{not viable}}
};
auto *aa = new Aligned<__STDCPP_DEFAULT_NEW_ALIGNMENT__>; // expected-error {{no matching}}
auto *ab = new Aligned<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2>; // expected-error {{deleted}}
auto *aap = new (arg) Aligned<__STDCPP_DEFAULT_NEW_ALIGNMENT__, Arg>; // expected-error {{no matching}}
auto *abp = new (arg) Aligned<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2, Arg>; // expected-error {{deleted}}

// If both are available, we prefer the aligned version for an overaligned
// type, and only use the unaligned version for a non-overaligned type.

template<unsigned Align, typename ...Ts>
struct alignas(Align) Both1 {
  void *operator new(size_t, Ts...); // expected-note 2{{not viable}}
  void *operator new(size_t, std::align_val_t, Ts...) = delete; // expected-note 2{{deleted}}
};
template<unsigned Align, typename ...Ts>
struct alignas(Align) Both2 {
  void *operator new(size_t, Ts...) = delete; // expected-note 2{{deleted}}
  void *operator new(size_t, std::align_val_t, Ts...); // expected-note 2{{not viable}}
};
auto *b1a = new Both1<__STDCPP_DEFAULT_NEW_ALIGNMENT__>;
auto *b1b = new Both1<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2>; // expected-error {{deleted}}
auto *b2a = new Both2<__STDCPP_DEFAULT_NEW_ALIGNMENT__>; // expected-error {{deleted}}
auto *b2b = new Both2<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2>;
auto *b1ap = new (arg) Both1<__STDCPP_DEFAULT_NEW_ALIGNMENT__, Arg>;
auto *b1bp = new (arg) Both1<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2, Arg>; // expected-error {{deleted}}
auto *b2ap = new (arg) Both2<__STDCPP_DEFAULT_NEW_ALIGNMENT__, Arg>; // expected-error {{deleted}}
auto *b2bp = new (arg) Both2<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2, Arg>;

// Note that the aligned form can select a function with a parameter different
// from std::align_val_t.

struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) WeirdAlignedAlloc1 {
  void *operator new(size_t, ...) = delete; // expected-note 2{{deleted}}
};
auto *waa1 = new WeirdAlignedAlloc1; // expected-error {{deleted}}
auto *waa1p = new (arg) WeirdAlignedAlloc1; // expected-error {{deleted}}

struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) WeirdAlignedAlloc2 {
  template<typename ...T>
  void *operator new(size_t, T...) {
    using U = void(T...); // expected-note 2{{previous}}
    using U = void; // expected-error {{different types ('void' vs 'void (std::align_val_t)')}} \
                       expected-error {{different types ('void' vs 'void (std::align_val_t, Arg)')}}
  }
};
auto *waa2 = new WeirdAlignedAlloc2; // expected-note {{instantiation of}}
auto *waa2p = new (arg) WeirdAlignedAlloc2; // expected-note {{instantiation of}}
