// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-shared-ptr-array %t

namespace std {

template <typename T>
struct default_delete {
  constexpr default_delete() noexcept = default;
  void operator()(T *ptr) const;
};

template <typename T>
struct default_delete<T[]> {
  constexpr default_delete() noexcept = default;

  template <typename U>
  default_delete(const default_delete<U[]> &) noexcept {}

  void operator()(T *ptr) const;
};

template <typename T>
class shared_ptr {
public:
  constexpr shared_ptr() noexcept = default;

  template <typename Y>
  explicit shared_ptr(Y *ptr) {}

  template <typename Y, typename D>
  shared_ptr(Y *ptr, D d) {}

  template <typename Y, typename D>
  void reset(Y *ptr, D d) {}

  shared_ptr &operator=(const shared_ptr &) { return *this; }

  T *get() const noexcept { return nullptr; } 
};

} // namespace std


// Test types and helpers

struct A { int x; };
struct B { int x; };
struct Base {};
struct Derived : Base {};

struct WithDtor {
  ~WithDtor() {}
};

struct ArrayFunctorDeleter {
  void operator()(A *p) const { delete[] p; }
};

template <typename T1, typename T2>
struct PairLike {};

void destroy_array(A *p) { delete[] p; }
void destroy_single(A *p) { delete p; }

void destroy_multi(A *p) {
  int x = 0;
  delete[] p;
}

void destroy_conditional(A *p) {
  if (p)
    delete[] p;
}

A *GlobalArray;
void delete_global(A *p) { delete[] GlobalArray; }
void delete_other(A *p) { A *q = nullptr; delete[] q; }

using MyDelete = std::default_delete<A[]>;
using MyA = A;
typedef A AliasA;
typedef std::default_delete<A[]> TDDelete;

constexpr int kBufSize = 128;

struct Wrapper {
  explicit Wrapper(std::shared_ptr<A> p) {}
};

std::shared_ptr<A> make_shared_from(std::shared_ptr<A> p) { return p; }


// Positive:

void positive_default_delete_basic() {
  std::shared_ptr<A> basicSp(new A[10], std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<A[]>' instead of 'std::shared_ptr<A>' with explicit array deleter [modernize-use-shared-ptr-array]
  // CHECK-FIXES: std::shared_ptr<A[]> basicSp(new A[10]);
}

void positive_default_delete_brace_init() {
  std::shared_ptr<A> braceSp(new A[10], std::default_delete<A[]>{});
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> braceSp(new A[10]);
}

void positive_deleter_using_alias() {
  std::shared_ptr<A> sp(new A[10], MyDelete());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(new A[10]);
}

void positive_deleter_typedef_alias() {
  std::shared_ptr<A> sp(new A[10], TDDelete());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(new A[10]);
}

void positive_fully_qualified() {
  ::std::shared_ptr<A> qualifiedSp(
      new A[10],
      ::std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: ::std::shared_ptr<A[]> qualifiedSp(
  // CHECK-FIXES-NEXT:       new A[10]);
}

namespace st = std;

void positive_namespace_alias() {
  st::shared_ptr<A> sp(
      new A[10],
      st::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: st::shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       new A[10]);
}

using std::shared_ptr;
using std::default_delete;

void positive_using_declarations() {
  shared_ptr<A> sp(
      new A[10],
      default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       new A[10]);
}

void positive_lambda_basic() {
  std::shared_ptr<A> sp(new A[10], [](A *p) { delete[] p; });
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(new A[10]);
}

void positive_lambda_noexcept() {
  std::shared_ptr<A> sp(
      new A[10],
      [](A *p) noexcept { delete[] p; });
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       new A[10]);
}

void positive_lambda_explicit_return_type() {
  std::shared_ptr<A> sp(new A[10], [](A *p) -> void { delete[] p; });
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(new A[10]);
}

void positive_lambda_paren_delete() {
  std::shared_ptr<A> sp(
      new A[10],
      [](A *p) { (delete[] p); });
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       new A[10]);
}

void positive_function_deleter() {
  std::shared_ptr<A> sp(new A[10], destroy_array);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(new A[10]);
}

void positive_primitive() {
  std::shared_ptr<int> sp(
      new int[32],
      std::default_delete<int[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<int[]> sp(
  // CHECK-FIXES-NEXT:       new int[32]);
}

void positive_lambda_primitive() {
  std::shared_ptr<int> sp(
      new int[64],
      [](int *p) { delete[] p; });
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<int[]> sp(
  // CHECK-FIXES-NEXT:       new int[64]);
}

void positive_using_alias() {
  std::shared_ptr<MyA> sp(
      new MyA[10],
      std::default_delete<MyA[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<MyA[]> sp(
  // CHECK-FIXES-NEXT:       new MyA[10]);
}

void positive_typedef_alias() {
  std::shared_ptr<AliasA> sp(
      new AliasA[10],
      std::default_delete<AliasA[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<AliasA[]> sp(
  // CHECK-FIXES-NEXT:       new AliasA[10]);
}

void positive_nested_template_type() {
  std::shared_ptr<PairLike<int, int>>
      sp(new PairLike<int, int>[4],
         std::default_delete<PairLike<int, int>[]>());
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<PairLike<int, int>[]>
  // CHECK-FIXES-NEXT:       sp(new PairLike<int, int>[4]);
}

void positive_const_element() {
  std::shared_ptr<const A> sp(
      new const A[10](),
      std::default_delete<const A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<const A[]> sp(
  // CHECK-FIXES-NEXT:       new const A[10]());
}

void positive_volatile_element() {
  std::shared_ptr<volatile int> sp(
      new volatile int[8],
      std::default_delete<volatile int[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<volatile int[]> sp(
  // CHECK-FIXES-NEXT:       new volatile int[8]);
}

void positive_const_volatile_element() {
  std::shared_ptr<const volatile A> sp(
      new const volatile A[4](),
      std::default_delete<const volatile A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<const volatile A[]> sp(
  // CHECK-FIXES-NEXT:       new const volatile A[4]());
}

void positive_nontrivial_destructor_default_delete() {
  std::shared_ptr<WithDtor> sp(
      new WithDtor[10],
      std::default_delete<WithDtor[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<WithDtor[]> sp(
  // CHECK-FIXES-NEXT:       new WithDtor[10]);
}

void positive_nontrivial_destructor_lambda() {
  std::shared_ptr<WithDtor> sp(
      new WithDtor[5],
      [](WithDtor *p) { delete[] p; });
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<WithDtor[]> sp(
  // CHECK-FIXES-NEXT:       new WithDtor[5]);
}

void positive_rangle_merge_context() {
  std::shared_ptr<std::shared_ptr<A>> sp(
      new std::shared_ptr<A>[4],
      std::default_delete<std::shared_ptr<A>[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr
  // CHECK-FIXES: std::shared_ptr<std::shared_ptr<A>[]> sp(
  // CHECK-FIXES-NEXT:       new std::shared_ptr<A>[4]);
}

void positive_comment_before_type() {
  std::shared_ptr</*before*/ A> sp(new A[10], std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr</*before*/ A[]> sp(new A[10]);
}

void positive_comment_after_type() {
  std::shared_ptr<A /*after*/> sp(new A[10], std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[] /*after*/> sp(new A[10]);
}

void positive_comment_both_sides() {
  std::shared_ptr</*before*/ A /*after*/> sp(new A[10], std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr</*before*/ A[] /*after*/> sp(new A[10]);
}

void positive_comment_multiline() {
  std::shared_ptr<
      /*before*/ A /*after*/
  > sp(new A[10], std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<
  // CHECK-FIXES-NEXT:       /*before*/ A[] /*after*/
  // CHECK-FIXES-NEXT:   > sp(new A[10]);
}

void positive_template_comments() {
  std::shared_ptr</*a*/ A /*b*/> sp(
      new A[10],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr</*a*/ A[] /*b*/> sp(
  // CHECK-FIXES-NEXT:       new A[10]);
}

void positive_runtime_size() {
  int n = 10;
  std::shared_ptr<A> sp(
      new A[n],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       new A[n]);
}

void positive_constexpr_size() {
  std::shared_ptr<A> sp(
      new A[kBufSize],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       new A[kBufSize]);
}

void positive_zero_size() {
  std::shared_ptr<A> sp(
      new A[0],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       new A[0]);
}

void positive_value_initialized_array() {
  std::shared_ptr<A> sp(
      new A[10](),
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       new A[10]());
}

void positive_brace_initialized_array() {
  std::shared_ptr<int> sp(
      new int[3]{1, 2, 3},
      std::default_delete<int[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<int[]> sp(
  // CHECK-FIXES-NEXT:       new int[3]{1, 2, 3});
}

void positive_extra_parens() {
  std::shared_ptr<A> sp(
      (new A[10]),
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       (new A[10]));
}

void positive_brace_init_ctor() {
  std::shared_ptr<A> sp{new A[10], std::default_delete<A[]>()};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp{new A[10]};
}

void positive_static_decl() {
  static std::shared_ptr<A> sp(
      new A[10],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: static std::shared_ptr<A[]> sp(
  // CHECK-FIXES-NEXT:       new A[10]);
}

// Copy-init: VarDecl declared type and the temporary both need '[]' inserted.
void positive_copy_init() {
  std::shared_ptr<A> sp =
      std::shared_ptr<A>(
          new A[10],
          std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp =
  // CHECK-FIXES-NEXT:       std::shared_ptr<A[]>(
  // CHECK-FIXES-NEXT:           new A[10]);
}

// auto direct-init: no VarDecl TypeLoc to patch; only the constructor is rewritten.
void positive_auto_deduction() {
  auto sp =
      std::shared_ptr<A>(
          new A[10],
          std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: auto sp =
  // CHECK-FIXES-NEXT:       std::shared_ptr<A[]>(
  // CHECK-FIXES-NEXT:           new A[10]);
}

void positive_auto_copy_init() {
  auto sp = std::shared_ptr<A>(new A[10], std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: auto sp = std::shared_ptr<A[]>(new A[10]);
}

void positive_unqualified_using_namespace() {
  using namespace std;
  shared_ptr<A> sp(new A[10], default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr
  // CHECK-FIXES: shared_ptr<A[]> sp(new A[10]);
}

// Member VarDecl with in-class initializer: both the member type and the
// temporary constructor expression require independent insertions.
struct Holder {
  std::shared_ptr<A> member =
      std::shared_ptr<A>(new A[10], std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> member =
  // CHECK-FIXES-NEXT:       std::shared_ptr<A[]>(new A[10]);

  // Constructor initializer list: member is already declared; warn only since
  // the declaration site is not reachable for transformation here.
  Holder() : member(new A[10], std::default_delete<A[]>()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
};

void positive_multiline_input() {
  std::shared_ptr<A>
      sp(
          new A[10],
          std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]>
  // CHECK-FIXES-NEXT:       sp(
  // CHECK-FIXES-NEXT:           new A[10]);
}

void positive_inline_comment_preserve() {
  std::shared_ptr<A> /*type-comment*/ sp(
      new A[10], /*deleter-comment*/
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> /*type-comment*/ sp(
  // CHECK-FIXES-NEXT:       new A[10]);
}


// Warn only:

// Multi-declarator: shared TypeLoc makes independent fix-its unsafe.
void warn_only_multi_declarator() {
  std::shared_ptr<A> sp1(new A[10], std::default_delete<A[]>()),
                     sp2(new A[8],  std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A> sp1(new A[10], std::default_delete<A[]>()),
  // CHECK-FIXES-NEXT:                  sp2(new A[8],  std::default_delete<A[]>());
}

void warn_only_pointer_declarator() {
  std::shared_ptr<A> *sp =
      new std::shared_ptr<A>(
          new A[10],
          std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr
}

void warn_only_reference_declarator() {
  std::shared_ptr<A> &sp =
      *new std::shared_ptr<A>(
          new A[10],
          std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr
}

// Assignment: declaration site is not reachable for transformation.
void warn_only_assignment_after_decl() {
  std::shared_ptr<A> sp;
  sp = std::shared_ptr<A>(
      new A[10],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
}

void warn_only_chained_assignment() {
  std::shared_ptr<A> sp, sp2;
  sp = sp2 = std::shared_ptr<A>(
      new A[10],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
}

std::shared_ptr<A> warn_only_return_stmt() {
  return std::shared_ptr<A>(
      new A[10],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning:
}

std::shared_ptr<PairLike<int,int>> warn_only_return_nested_template() {
  return std::shared_ptr<PairLike<int,int>>(
      new PairLike<int,int>[4],
      std::default_delete<PairLike<int,int>[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr
}


using std::shared_ptr;    
using std::default_delete;

void warn_only_nested_as_get_arg() {
  std::shared_ptr<A> outer(
      std::shared_ptr<A>(
          new A[10],
          std::default_delete<A[]>())
          .get(),
      std::default_delete<A[]>());

  // CHECK-MESSAGES: :[[@LINE-6]]:{{[0-9]+}}: warning:
}

void warn_only_passed_to_wrapper_ctor() {
  Wrapper w(
      std::shared_ptr<A>(
          new A[10],
          std::default_delete<A[]>()));

  // CHECK-MESSAGES: :[[@LINE-4]]:{{[0-9]+}}: warning:
}

void warn_only_auto_wrapper_temporary() {
  auto outer =
      Wrapper(std::shared_ptr<A>(
          new A[10],
          std::default_delete<A[]>()));

  // CHECK-MESSAGES: :[[@LINE-4]]:{{[0-9]+}}: warning:
}

void warn_only_passed_to_function_call() {
  std::shared_ptr<A> outer =
      make_shared_from(std::shared_ptr<A>(new A[10], std::default_delete<A[]>()));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr
 
}

std::shared_ptr<A> make(std::shared_ptr<A>);
void warn_only_passed_to_factory() {
  std::shared_ptr<A> outer =
      make(std::shared_ptr<A>(
          new A[10],
          std::default_delete<A[]>()));

  // CHECK-MESSAGES: :[[@LINE-4]]:{{[0-9]+}}: warning:
}

// Negative:

void negative_already_array_type() {
  std::shared_ptr<A[]> sp(new A[10]);
}

void negative_already_array_with_default_delete() {
  std::shared_ptr<A[]> sp(new A[10], std::default_delete<A[]>());
}

void negative_single_object_no_deleter() {
  std::shared_ptr<A> sp(new A);
}

void negative_single_object_default_delete() {
  std::shared_ptr<A> sp(new A, std::default_delete<A>());
}

void negative_single_object_lambda() {
  std::shared_ptr<A> sp(new A, [](A *p) { delete p; });
}

void negative_array_new_no_deleter() {
  std::shared_ptr<A> sp(new A[10]);
}

void negative_array_new_single_deleter() {
  std::shared_ptr<A> sp(new A[10], std::default_delete<A>());
}

void negative_array_new_lambda_single_delete() {
  std::shared_ptr<A> sp(new A[10], [](A *p) { delete p; });
}

void negative_type_mismatch_allocated() {
  std::shared_ptr<A> sp(new B[10], std::default_delete<B[]>());
}

void negative_type_mismatch_deleter() {
  std::shared_ptr<A> sp(new A[10], std::default_delete<B[]>());
}

void negative_covariant_array() {
  std::shared_ptr<Base> sp(new Derived[10], std::default_delete<Derived[]>());
}

void negative_nontrivial_destructor_wrong_deleter() {
  std::shared_ptr<WithDtor> sp(new WithDtor[5], std::default_delete<WithDtor>());
}

void negative_functor_deleter() {
  std::shared_ptr<A> sp(new A[10], ArrayFunctorDeleter{});
}

void negative_function_pointer_variable() {
  auto fn = destroy_array;
  std::shared_ptr<A> sp(new A[10], fn);
}

void negative_lambda_capture() {
  int x = 0;
  std::shared_ptr<A> sp(new A[10], [x](A *p) { delete[] p; });
}

void negative_lambda_multiple_statements() {
  std::shared_ptr<A> sp(
      new A[10],
      [](A *p) { int x = 0; delete[] p; });
}

void negative_lambda_delete_alias() {
  std::shared_ptr<A> sp(
      new A[10],
      [](A *p) { A *q = p; delete[] q; });
}

void negative_lambda_conditional_delete() {
  std::shared_ptr<A> sp(new A[10], [](A *p) { if (p) delete[] p; });
}

void negative_lambda_return_before_delete() {
  std::shared_ptr<A> sp(
      new A[10],
      [](A *p) { if (!p) return; delete[] p; });
}

void negative_lambda_no_delete() {
  std::shared_ptr<A> sp(new A[10], [](A *) {});
}

void negative_void_ptr_cast_lambda() {
  std::shared_ptr<void> sp(
      new A[10],
      [](void *p) { delete[] static_cast<A *>(p); });
}

void negative_function_single_delete() {
  std::shared_ptr<A> sp(new A[10], destroy_single);
}

void negative_function_multi_stmt() {
  std::shared_ptr<A> sp(new A[10], destroy_multi);
}

void negative_function_conditional_delete() {
  std::shared_ptr<A> sp(new A[10], destroy_conditional);
}

void negative_function_global_delete() {
  std::shared_ptr<A> sp(new A[10], delete_global);
}

void negative_function_delete_other() {
  std::shared_ptr<A> sp(new A[10], delete_other);
}

void negative_reset() {
  std::shared_ptr<A> sp;
  sp.reset(new A[10], std::default_delete<A[]>());
}

void negative_reset_lambda() {
  std::shared_ptr<A> sp;
  sp.reset(new A[10], [](A *p) { delete[] p; });
}

#define MAKE_SHARED_PTR \
  std::shared_ptr<A>(new A[10], std::default_delete<A[]>())

void negative_macro_full_expansion() {
  std::shared_ptr<A> sp = MAKE_SHARED_PTR;
}

#define ARR_DEL std::default_delete<A[]>()

void negative_macro_deleter_only() {
  std::shared_ptr<A> sp(new A[10], ARR_DEL);
}

template <typename T>
void negative_dependent_element(int n) {
  std::shared_ptr<T> sp(
      new T[n],
      std::default_delete<T[]>());
}

template <typename T>
struct Box {};

template <typename T>
void negative_dependent_template_element(int n) {
  std::shared_ptr<Box<T>> sp(
      new Box<T>[n],
      std::default_delete<Box<T>[]>());
}

template <template <typename> class C, typename T>
void negative_dependent_template_template_element(int n) {
  std::shared_ptr<C<T>> sp(
      new C<T>[n],
      std::default_delete<C<T>[]>());
}

template <typename T>
using Alias = Box<T>;

template <typename T>
void negative_dependent_alias_template_element(int n) {
  std::shared_ptr<Alias<T>> sp(
      new Alias<T>[n],
      std::default_delete<Alias<T>[]>());
}
void instantiate_templates() {
  negative_dependent_element<A>(10);
  negative_dependent_template_element<A>(10);
  negative_dependent_alias_template_element<A>(10);
}