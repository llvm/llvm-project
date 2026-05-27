// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fms-extensions -std=c++20 -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20 -fms-extensions -verify=ref,both %s

namespace std {
  typedef decltype(sizeof(int)) size_t;
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    constexpr initializer_list() : __begin_(nullptr), __size_(0) {}

    constexpr size_t    size()  const {return __size_;}
    constexpr const _E* begin() const {return __begin_;}
    constexpr const _E* end()   const {return __begin_ + __size_;}
  };
}

class Thing {
public:
  int m = 12;
  constexpr Thing(int m) : m(m) {}
  constexpr bool operator==(const Thing& that) const {
    return this->m == that.m;
  }
};

constexpr bool is_contained(std::initializer_list<Thing> Set, const Thing &Element) {
   return (*Set.begin() == Element);
}

constexpr int foo() {
  const Thing a{12};
  const Thing b{14};
  return is_contained({a}, b);
}

static_assert(foo() == 0);


namespace rdar13395022 {
  struct MoveOnly { // both-note {{candidate}}
    MoveOnly(MoveOnly&&); // both-note 2{{copy constructor is implicitly deleted because}} both-note {{candidate}}
  };

  void test(MoveOnly mo) {
    auto &&list1 = {mo}; // both-error {{call to implicitly-deleted copy constructor}} both-note {{in initialization of temporary of type 'std::initializer_list}}
    MoveOnly (&&list2)[1] = {mo}; // both-error {{call to implicitly-deleted copy constructor}} both-note {{in initialization of temporary of type 'MoveOnly[1]'}}
    std::initializer_list<MoveOnly> &&list3 = {};
    MoveOnly (&&list4)[1] = {}; // both-error {{no matching constructor}}
    // both-note@-1 {{in implicit initialization of array element 0 with omitted initializer}}
    // both-note@-2 {{in initialization of temporary of type 'MoveOnly[1]' created to list-initialize this reference}}
  }
}

namespace cwg2765 {
  constexpr bool same(std::initializer_list<int> a,
                      std::initializer_list<int> b) {
    return a.begin() != b.begin(); // #cwg2765-init-list-compare
  }
  static_assert(same({1}, {1}), "");
  // both-error@-1 {{static assertion expression is not an integral constant expression}}
  // both-note@#cwg2765-init-list-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@-3 {{in call to}}

  template <class T>
  constexpr bool begins_equal(T a, T b) {
    return a.begin() == b.begin(); // #cwg2765-template-compare
  }
  constexpr bool same_three =
      begins_equal<std::initializer_list<int>>({1, 2, 3}, {1, 2, 3});
  // both-error@-2 {{constexpr variable 'same_three' must be initialized by a constant expression}}
  // both-note@#cwg2765-template-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@-3 {{in call to}}

  constexpr bool different_three =
      begins_equal<std::initializer_list<int>>({1, 2, 3}, {4, 5, 6});
  static_assert(!different_three, "");

  constexpr bool same_object() {
    std::initializer_list<int> il = {1, 1, 1};
    return il.begin() == il.begin() && il.begin() != il.begin() + 1;
  }
  static_assert(same_object(), "");

  constexpr bool same_pointer_value() {
    std::initializer_list<int> il = {1, 1, 1};
    const int *p = il.begin();
    return p + 0 == p && p != p + 1;
  }
  static_assert(same_pointer_value(), "");

  constexpr bool local_list(const int *p) {
    std::initializer_list<int> il = {1, 2, 3};
    return p ? (p == il.begin()) : local_list(il.begin()); // #cwg2765-local-compare
  }
  constexpr bool same_local = local_list(nullptr); // #cwg2765-local-call
  // both-error@-1 {{constexpr variable 'same_local' must be initialized by a constant expression}}
  // both-note@#cwg2765-local-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@#cwg2765-local-compare {{in call to}}
  // both-note@#cwg2765-local-call {{in call to}}

  constexpr bool shifted(std::initializer_list<int> a,
                         std::initializer_list<int> b) {
    return a.begin() != b.begin() + 1; // #cwg2765-shifted-compare
  }
  constexpr bool annex_c = shifted({2, 3}, {1, 2, 3});
  // both-error@-1 {{constexpr variable 'annex_c' must be initialized by a constant expression}}
  // both-note@#cwg2765-shifted-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@-3 {{in call to}}

  constexpr bool end_shifted(std::initializer_list<int> a,
                             std::initializer_list<int> b) {
    return a.end() == b.begin() + 1; // #cwg2765-end-shifted-compare
  }
  constexpr bool same_end_shifted = end_shifted({1}, {1, 2});
  // both-error@-1 {{constexpr variable 'same_end_shifted' must be initialized by a constant expression}}
  // both-note@#cwg2765-end-shifted-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@-3 {{in call to}}

  constexpr bool different_end_shifted = end_shifted({1}, {2, 1});
  static_assert(!different_end_shifted, "");

  struct Box {
    int n;
  };
  constexpr bool class_same(std::initializer_list<Box> a,
                            std::initializer_list<Box> b) {
    return a.begin() == b.begin(); // #cwg2765-class-compare
  }
  constexpr bool same_box = class_same({{1}}, {{1}});
  // both-error@-1 {{constexpr variable 'same_box' must be initialized by a constant expression}}
  // both-note@#cwg2765-class-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@-3 {{in call to}}

  constexpr bool different_box = class_same({{1}}, {{2}});
  static_assert(!different_box, "");

  constexpr bool class_field_same(std::initializer_list<Box> a,
                                  std::initializer_list<Box> b) {
    return &a.begin()->n == &b.begin()->n; // #cwg2765-field-compare
  }
  constexpr bool same_box_field = class_field_same({{1}}, {{1}});
  // both-error@-1 {{constexpr variable 'same_box_field' must be initialized by a constant expression}}
  // both-note@#cwg2765-field-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@-3 {{in call to}}

  constexpr bool different_box_field = class_field_same({{1}}, {{2}});
  static_assert(!different_box_field, "");

  struct WithPointer {
    const int *p;
    int n;
  };
  constexpr int pointer_anchor = 0;
  constexpr int pointer_anchor_2 = 0;
  constexpr bool class_with_pointer_same(std::initializer_list<WithPointer> a,
                                         std::initializer_list<WithPointer> b) {
    return a.begin() == b.begin(); // #cwg2765-with-pointer-compare
  }
  // Both elements compare equal in the scalar field; the pointer field
  // points to the same global, so observable equality is Unknown overall
  // and the address comparison is unspecified.
  constexpr bool same_after_pointer_field =
      class_with_pointer_same({{&pointer_anchor, 1}},
                              {{&pointer_anchor, 1}});
  // both-error@-3 {{constexpr variable 'same_after_pointer_field' must be initialized by a constant expression}}
  // both-note@#cwg2765-with-pointer-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@-4 {{in call to}}

  // The non-pointer field already disagrees, so we can conclude the
  // backing arrays are distinct even though a pointer field is present.
  constexpr bool different_after_pointer_field =
      class_with_pointer_same({{&pointer_anchor, 1}},
                              {{&pointer_anchor, 2}});
  static_assert(!different_after_pointer_field, "");

  // Distinct strong-symbol globals have distinct addresses, so the
  // pointer field alone is enough to conclude the backing arrays differ.
  constexpr bool different_pointer_field =
      class_with_pointer_same({{&pointer_anchor, 1}},
                              {{&pointer_anchor_2, 1}});
  static_assert(!different_pointer_field, "");

  // A null pointer is distinct from a pointer-to-object.
  constexpr bool one_null_pointer =
      class_with_pointer_same({{nullptr, 1}}, {{&pointer_anchor, 1}});
  static_assert(!one_null_pointer, "");

  // Both null and scalar fields agree -> potentially overlapping.
  constexpr bool both_null_pointer =
      class_with_pointer_same({{nullptr, 1}}, {{nullptr, 1}}); // #cwg2765-both-null-call
  // both-error@-2 {{constexpr variable 'both_null_pointer' must be initialized by a constant expression}}
  // both-note@#cwg2765-with-pointer-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@#cwg2765-both-null-call {{in call to}}

  struct WithStringPointer { const char *p; };
  constexpr bool class_string_pointer_same(
      std::initializer_list<WithStringPointer> a,
      std::initializer_list<WithStringPointer> b) {
    return a.begin() == b.begin(); // #cwg2765-string-ptr-compare
  }
  constexpr bool different_string_pointer =
      class_string_pointer_same({{"abc"}}, {{"def"}});
  static_assert(!different_string_pointer, "");

  constexpr bool same_string_pointer =
      class_string_pointer_same({{"abc"}}, {{"abc"}}); // #cwg2765-same-string-ptr-call
  // both-error@-2 {{constexpr variable 'same_string_pointer' must be initialized by a constant expression}}
  // both-note@#cwg2765-string-ptr-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@#cwg2765-same-string-ptr-call {{in call to}}

  // Both pointers are at offset 0 and the literals have different sizes, so
  // "abc" cannot be merged into the start of "abcd": that would require a
  // null terminator at offset 3 of "abcd", which holds 'd'. The string-overlap
  // predicate resolves this to NotEqual.
  constexpr bool prefix_string_pointer =
      class_string_pointer_same({{"abc"}}, {{"abcd"}});
  static_assert(!prefix_string_pointer, "");

  // A null const char * is distinct from any non-null string literal.
  constexpr bool null_vs_string =
      class_string_pointer_same({{nullptr}}, {{"abc"}});
  static_assert(!null_vs_string, "");

  // Floating-point uses bitwise equality: -0.0 is distinguishable from +0.0.
  struct WithFloat { double d; };
  constexpr bool class_float_same(std::initializer_list<WithFloat> a,
                                  std::initializer_list<WithFloat> b) {
    return a.begin() == b.begin();
  }
  constexpr bool different_signed_zero = class_float_same({{-0.0}}, {{0.0}});
  static_assert(!different_signed_zero, "");

  // Anonymous union field: the active member differs between LHS and RHS, so
  // the backing arrays must have distinct addresses.
  struct WithAnonUnion {
    union { int i; float f; };
    int tag;
  };
  constexpr bool class_anon_union_same(std::initializer_list<WithAnonUnion> a,
                                       std::initializer_list<WithAnonUnion> b) {
    return a.begin() == b.begin();
  }
  constexpr bool different_union_member =
      class_anon_union_same({{.i = 0, .tag = 0}}, {{.f = 0.0f, .tag = 0}});
  static_assert(!different_union_member, "");

  // Member pointers compare structurally: distinct target fields imply
  // distinct member-pointer values, so the backing arrays differ.
  struct Holder { int x; int y; };
  struct WithMemberPtr { int Holder::*p; };
  constexpr bool class_member_ptr_same(
      std::initializer_list<WithMemberPtr> a,
      std::initializer_list<WithMemberPtr> b) {
    return a.begin() == b.begin(); // #cwg2765-member-ptr-compare
  }
  constexpr bool different_member_ptr =
      class_member_ptr_same({{&Holder::x}}, {{&Holder::y}});
  static_assert(!different_member_ptr, "");

  // Same member pointer in both arrays + only this field -> potentially
  // overlapping, so address comparison is unspecified.
  constexpr bool same_member_ptr =
      class_member_ptr_same({{&Holder::x}}, {{&Holder::x}}); // #cwg2765-same-mptr-call
  // both-error@-2 {{constexpr variable 'same_member_ptr' must be initialized by a constant expression}}
  // both-note@#cwg2765-member-ptr-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@#cwg2765-same-mptr-call {{in call to}}

  // Weak member functions: pointers to the same weak decl still resolve to
  // a single merged target at link time, so two arrays holding pointers to
  // the same weak member are still potentially-overlapping (not provably
  // distinct, but not provably equal either — the link-time merge is fine).
  struct HolderWithWeak {
    __attribute__((weak)) void f();
    __attribute__((weak)) void g();
  };
  struct WithWeakMemberPtr { void (HolderWithWeak::*p)(); };
  constexpr bool class_weak_mptr_same(
      std::initializer_list<WithWeakMemberPtr> a,
      std::initializer_list<WithWeakMemberPtr> b) {
    return a.begin() == b.begin(); // #cwg2765-weak-mptr-compare
  }
  // Distinct weak decls may merge at link time -> Unknown.
  constexpr bool different_weak_member_ptr =
      class_weak_mptr_same({{&HolderWithWeak::f}}, // #cwg2765-diff-weak-mptr-call
                           {{&HolderWithWeak::g}});
  // both-error@-3 {{constexpr variable 'different_weak_member_ptr' must be initialized by a constant expression}}
  // both-note@#cwg2765-weak-mptr-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@#cwg2765-diff-weak-mptr-call {{in call to}}

  // nullptr_t fields: the only value of nullptr_t is null, so this field
  // never contributes inequality. The other field decides.
  struct WithNullptrT { decltype(nullptr) np; int n; };
  constexpr bool class_nullptr_t_same(std::initializer_list<WithNullptrT> a,
                                      std::initializer_list<WithNullptrT> b) {
    return a.begin() == b.begin();
  }
  constexpr bool different_after_nullptr_t =
      class_nullptr_t_same({{nullptr, 1}}, {{nullptr, 2}});
  static_assert(!different_after_nullptr_t, "");

  // Aggregate carrying a std::initializer_list member: the backing array's
  // extending declaration is the enclosing aggregate, not the
  // initializer_list. The dedicated marker on MaterializeTemporaryExpr
  // still recognises the backing array.
  struct WithIL { std::initializer_list<int> il; };
  constexpr WithIL agg_a{{1}}, agg_b{{1}};
  constexpr bool agg_same = agg_a.il.begin() == agg_b.il.begin(); // #cwg2765-agg-compare
  // both-error@-1 {{constexpr variable 'agg_same' must be initialized by a constant expression}}
  // both-note@#cwg2765-agg-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}

  constexpr WithIL agg_c{{1}}, agg_d{{2}};
  constexpr bool agg_different = agg_c.il.begin() == agg_d.il.begin();
  static_assert(!agg_different, "");

  // Rvalue reference to array bound to a braced-init-list: the materialized
  // array is NOT a std::initializer_list backing array.
  constexpr bool rvref_arr_same(const int (&&a)[3], const int (&&b)[3]) {
    return &a[0] == &b[0];
  }
  constexpr bool rvref_ok = rvref_arr_same({1, 2, 3}, {1, 2, 3});
  static_assert(!rvref_ok, "");

  // Vector elements: APValue::Profile recurses element-wise, so the
  // mergeability judgment is correctly element-sensitive (NOT "hard-code
  // these rare kinds to NotEqual"). Confirmed by runtime observation:
  // identical-content backing arrays are mergeable, differing ones are not.
  typedef int v4 __attribute__((ext_vector_type(4)));
  constexpr v4 va = {1, 2, 3, 4};
  constexpr v4 vb = {1, 2, 3, 5};
  struct WithVec { v4 v; };
  constexpr bool class_vec_same(std::initializer_list<WithVec> a,
                                std::initializer_list<WithVec> b) {
    return a.begin() == b.begin(); // #cwg2765-vec-compare
  }
  constexpr bool same_vec = class_vec_same({{va}}, {{va}}); // #cwg2765-same-vec-call
  // both-error@-1 {{constexpr variable 'same_vec' must be initialized by a constant expression}}
  // both-note@#cwg2765-vec-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
  // both-note@#cwg2765-same-vec-call {{in call to}}

  constexpr bool different_vec = class_vec_same({{va}}, {{vb}});
  static_assert(!different_vec, "");

  // Complex elements: Profile bitwise-compares real / imag, so two
  // complex values differing in only the imaginary part are correctly
  // recognised as distinct.
  struct WithComplex { _Complex double c; };
  constexpr bool class_complex_same(std::initializer_list<WithComplex> a,
                                    std::initializer_list<WithComplex> b) {
    return a.begin() == b.begin();
  }
  constexpr bool different_complex = class_complex_same(
      {{__builtin_complex(1.0, 2.0)}}, {{__builtin_complex(1.0, 3.0)}});
  static_assert(!different_complex, "");
}
