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


// Positive: default_delete<T[]> deleter forms

void positive_default_delete_basic() {
  std::shared_ptr<A> basicSp(new A[10], std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
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


// Positive: lambda deleter forms

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


// Positive: free function deleter

void positive_function_deleter() {
  std::shared_ptr<A> sp(new A[10], destroy_array);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A[]> sp(new A[10]);
}


// Positive: element type variants

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

// Non-trivial destructor: default_delete<T> would silently call the wrong destructor on all but the first element.
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

// Nested template context exercises >> token splitting for fix-it insertion (tok:greatergreater).
void positive_rangle_merge_context() {
  std::shared_ptr<std::shared_ptr<A>> sp(
      new std::shared_ptr<A>[4],
      std::default_delete<std::shared_ptr<A>[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr
  // CHECK-FIXES: std::shared_ptr<std::shared_ptr<A>[]> sp(
  // CHECK-FIXES-NEXT:       new std::shared_ptr<A>[4]);
}


// Positive: array size variants

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


// Positive: array initializer forms

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


// Positive: VarDecl constructor styles

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


// Positive: non-VarDecl contexts

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

// auto copy-init: same as direct-init — auto VarDecl has no written template-id.
void positive_auto_copy_init() {
  auto sp = std::shared_ptr<A>(new A[10], std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: auto sp = std::shared_ptr<A[]>(new A[10]);
}

// Return statement: constructor is rewritten; the function's declared return
// type is not updated and may require manual adjustment.
std::shared_ptr<A> positive_return_stmt() {
  return std::shared_ptr<A>(
      new A[10],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: return std::shared_ptr<A[]>(
  // CHECK-FIXES-NEXT:       new A[10]);
}

std::shared_ptr<PairLike<int,int>> positive_return_nested_template() {
  return std::shared_ptr<PairLike<int,int>>(
      new PairLike<int,int>[4],
      std::default_delete<PairLike<int,int>[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: return std::shared_ptr<PairLike<int,int>[]>(
  // CHECK-FIXES-NEXT:       new PairLike<int,int>[4]);
}

void positive_unqualified_using_namespace() {
  using namespace std;
  shared_ptr<A> sp(new A[10], default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'std::shared_ptr
  // CHECK-FIXES: shared_ptr<A[]> sp(new A[10]);
}


// Positive: formatting edge cases

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

void positive_template_comments() {
  std::shared_ptr</*a*/ A /*b*/> sp(
      new A[10],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr</*a*/ A /*b*/[]> sp(
  // CHECK-FIXES-NEXT:       new A[10]);
}



// Warn only: diagnostic emitted, no fix-it

// Multi-declarator: shared TypeLoc makes independent fix-its unsafe.
void warn_only_multi_declarator() {
  std::shared_ptr<A> sp1(new A[10], std::default_delete<A[]>()),
                     sp2(new A[8],  std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: std::shared_ptr<A> sp1(new A[10], std::default_delete<A[]>()),
  // CHECK-FIXES-NEXT:                  sp2(new A[8],  std::default_delete<A[]>());
}

// Pointer/reference declarators: rewriting both the declared type and
// constructor expression independently is not safe. Warn only.
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
  // CHECK-FIXES: sp = std::shared_ptr<A>(
}

// Chained assignment: same reasoning as single assignment.
void warn_only_chained_assignment() {
  std::shared_ptr<A> sp, sp2;
  sp = sp2 = std::shared_ptr<A>(
      new A[10],
      std::default_delete<A[]>());
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: use 'std::shared_ptr<
  // CHECK-FIXES: sp = sp2 = std::shared_ptr<A>(
}


// Negative: already correct

void negative_already_array_type() {
  std::shared_ptr<A[]> sp(new A[10]);
}

void negative_already_array_with_default_delete() {
  std::shared_ptr<A[]> sp(new A[10], std::default_delete<A[]>());
}


// Negative: wrong new/delete combination

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

// Array new with non-array default_delete: would silently corrupt — not our
// job to warn here since it's a separate bug, and the pattern doesn't match.
void negative_array_new_single_deleter() {
  std::shared_ptr<A> sp(new A[10], std::default_delete<A>());
}

void negative_array_new_lambda_single_delete() {
  std::shared_ptr<A> sp(new A[10], [](A *p) { delete p; });
}


// Negative: type mismatches

void negative_type_mismatch_allocated() {
  std::shared_ptr<A> sp(new B[10], std::default_delete<B[]>());
}

void negative_type_mismatch_deleter() {
  std::shared_ptr<A> sp(new A[10], std::default_delete<B[]>());
}

// Covariant array: Derived[] is not safely manageable via shared_ptr<Base[]>
// without a virtual destructor and matching sizes — leave it alone.
void negative_covariant_array() {
  std::shared_ptr<Base> sp(new Derived[10], std::default_delete<Derived[]>());
}

// Non-trivial destructor with the wrong (non-array) deleter: the existing code
// is already a bug, but it's not our pattern — the deleter pointee won't match.
void negative_nontrivial_destructor_wrong_deleter() {
  std::shared_ptr<WithDtor> sp(new WithDtor[5], std::default_delete<WithDtor>());
}


// Negative: unsupported deleter forms

// Functor: operator() is not a directly inspectable function body from here.
void negative_functor_deleter() {
  std::shared_ptr<A> sp(new A[10], ArrayFunctorDeleter{});
}

// Template-dependent: element type is dependent; can't canonically compare.
template <typename T>
void negative_template_context(int n) {
  std::shared_ptr<T> sp(new T[n], std::default_delete<T[]>());
}

// Function pointer variable: declRefExpr targets a VarDecl, not a FunctionDecl.
// The matcher binds only to functionDecl() refs; this silently falls through.
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

// Lambda casting to void* before delete[]: the DeclRefExpr inside the delete
// refers to the cast result, not the original parameter.
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

// Function deletes a global rather than its parameter: the DeclRefExpr inside
// the delete doesn't refer to the param.
void negative_function_global_delete() {
  std::shared_ptr<A> sp(new A[10], delete_global);
}

void negative_function_delete_other() {
  std::shared_ptr<A> sp(new A[10], delete_other);
}


// Negative: unsupported contexts

// reset() takes the same two-argument form but is a member function call, not
// a constructor — the cxxConstructExpr matcher won't fire.
void negative_reset() {
  std::shared_ptr<A> sp;
  sp.reset(new A[10], std::default_delete<A[]>());
}

void negative_reset_lambda() {
  std::shared_ptr<A> sp;
  sp.reset(new A[10], [](A *p) { delete[] p; });
}

// Negative: macros

// Full expansion in macro: new-expression and deleter are both inside the macro
// body; both source ranges are macro IDs, bail silently.
#define MAKE_SHARED_PTR \
  std::shared_ptr<A>(new A[10], std::default_delete<A[]>())

void negative_macro_full_expansion() {
  std::shared_ptr<A> sp = MAKE_SHARED_PTR;
}

// Deleter-only macro: the deleter arg's end location is a macro ID; the
// rewritten range would be unsafe to emit even though new A[10] is clean.
#define ARR_DEL std::default_delete<A[]>()

void negative_macro_deleter_only() {
  std::shared_ptr<A> sp(new A[10], ARR_DEL);
}

