// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedCallArgsChecker -std=c++20 -verify %s
// expected-no-diagnostics

// Previously [[clang::suppress]] failed for implicit template specializations when
// namespace has multiple redeclarations and suppression is in a different
// redeclaration than the lexical parent of the implicit specialization.
//
// The bug occurred when:
// 1. A template is forward-declared in one namespace block
// 2. A partial specialization with [[clang::suppress]] is in a different block
// 3. An implicit instantiation's getLexicalDeclContext() points to the wrong block

typedef __SIZE_TYPE__ size_t;

namespace std {
template<class T> struct remove_cvref { using type = T; };
template<class T> struct remove_cvref<T&> { using type = T; };
template<class T> using remove_cvref_t = typename remove_cvref<T>::type;

template<class... Ts> struct tuple {};
template<size_t I, class T> struct tuple_element;
template<class T, class... Rs> struct tuple_element<0, tuple<T, Rs...>> { using type = T; };
template<size_t I, class T, class... Rs> struct tuple_element<I, tuple<T, Rs...>> : tuple_element<I-1, tuple<Rs...>> {};
template<size_t I, class T> using tuple_element_t = typename tuple_element<I, T>::type;

template<class T> struct optional { T v; T&& operator*() && { return static_cast<T&&>(v); } operator bool() const { return true; } };
template<class T, class... As> optional<T> make_optional(As...);
} // namespace std

namespace WTF { template<class T> T&& move(T&); }

void WTFCrash();
template<class T> struct CanMakeCheckedPtrBase {
  void incrementCheckedPtrCount() const { ++m; }
  void decrementCheckedPtrCount() const { if (!m) WTFCrash(); --m; }
  mutable T m{};
};
struct Checked : CanMakeCheckedPtrBase<unsigned> {};
Checked* get();

// First namespace block - forward declaration
# 1 "Header1.h" 1
namespace N {
template<class> struct C;
template<class T> struct C<T*> {
  template<class D> static std::optional<T*> decode(D&) { return {get()}; }
};
} // namespace N
# 50 "test.cpp" 2

// Second namespace block - this becomes the lexical parent for implicit specializations
# 1 "Header2.h" 1
namespace N {
template<class> struct C;
struct Dec {
  template<class T> std::optional<T> decode() { return {C<std::remove_cvref_t<T>>::decode(*this)}; }
};
} // namespace N
# 100 "test.cpp" 2

// Third namespace block - [[clang::suppress]] is here
# 1 "Header3.h" 1
namespace N {
template<class... Es> struct C<std::tuple<Es...>> {
  template<class D, class... Os>
  static std::optional<std::tuple<Es...>> decode(D& d, std::optional<Os>&&... os) {
    if constexpr (sizeof...(Os) < sizeof...(Es)) {
      auto o = d.template decode<std::tuple_element_t<sizeof...(Os), std::tuple<Es...>>>();
      if (!o) return {};
      return decode(d, WTF::move(os)..., WTF::move(o));
    } else {
      // This [[clang::suppress]] should suppress the warning.
      // Bug: Without the fix, it doesn't work because the implicit
      // specialization's lexical parent points to Header2.h's namespace block.
      [[clang::suppress]] return std::make_optional<std::tuple<Es...>>(*WTF::move(os)...);
    }
  }
};
} // namespace N
# 150 "test.cpp" 2

void rdar_168941095() {
  N::Dec d;
  (void)d.decode<std::tuple<Checked*, Checked*>>();
}
