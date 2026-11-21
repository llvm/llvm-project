// RUN: %clang_cc1 -fsyntax-only -verify=expected,eager %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

// The counted_by family on a function parameter. The count is parsed where it
// is written, so it names a parameter declared before the annotated one. With
// late parsing, a count written after the declarator is parsed at the end of
// the parameter list.

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))
#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

typedef int *int_ptr;
typedef void *void_ptr;
int global_count;

//==============================================================================
// Valid: the count is declared before the annotated pointer
//==============================================================================

void back_ref(int count, int *__counted_by(count) buf);
void back_ref_or_null(int count, int *__counted_by_or_null(count) buf);
void back_ref_sized(int count, void *__sized_by(count) buf);
void back_ref_sized_or_null(int count, void *__sized_by_or_null(count) buf);

// Two pointers may share a count.
void two_buffers(int count, int *__counted_by(count) a,
                 int *__counted_by(count) b);
void two_buffers_or_null(int count, int *__counted_by_or_null(count) a,
                         int *__counted_by_or_null(count) b);
void two_buffers_sized(int count, void *__sized_by(count) a,
                       void *__sized_by(count) b);
void two_buffers_sized_or_null(int count, void *__sized_by_or_null(count) a,
                               void *__sized_by_or_null(count) b);

// Written after the declarator.
void trailing(int count, int *buf __counted_by(count));
void trailing_or_null(int count, int *buf __counted_by_or_null(count));
void trailing_sized(int count, void *buf __sized_by(count));
void trailing_sized_or_null(int count, void *buf __sized_by_or_null(count));

// Written in the declaration specifiers.
void declspec(int count, int_ptr __counted_by(count) buf);
void declspec_or_null(int count, int_ptr __counted_by_or_null(count) buf);
void declspec_sized(int count, void_ptr __sized_by(count) buf);
void declspec_sized_or_null(int count, void_ptr __sized_by_or_null(count) buf);

// The C23 spelling is checked the same way.
void c23_spelling(int count, int *[[clang::counted_by(count)]] buf);
void c23_spelling_or_null(int count, int *[[clang::counted_by_or_null(count)]] buf);
void c23_spelling_sized(int count, void *[[clang::sized_by(count)]] buf);
void c23_spelling_sized_or_null(int count, void *[[clang::sized_by_or_null(count)]] buf);

// A count on the outer pointer of a pointer to pointer.
void nested_outer(int count, int **__counted_by(count) buf);
void nested_outer_or_null(int count, int **__counted_by_or_null(count) buf);
void nested_outer_sized(int count, int **__sized_by(count) buf);
void nested_outer_sized_or_null(int count, int **__sized_by_or_null(count) buf);

int sum(int count, int *__counted_by(count) buf) {
  int total = 0;
  for (int i = 0; i < count; ++i)
    total += buf[i];
  return total;
}

//==============================================================================
// A pointer has only one count
//==============================================================================

// expected-error@+1{{'counted_by' attribute cannot be applied to a pointer that already has a 'counted_by' attribute}}
void repeated(int n, int m, int *buf __counted_by(n) __counted_by(m));
// expected-error@+1{{'counted_by_or_null' attribute cannot be applied to a pointer that already has a 'counted_by_or_null' attribute}}
void repeated_or_null(int n, int m, int *buf __counted_by_or_null(n) __counted_by_or_null(m));
// expected-error@+1{{'sized_by' attribute cannot be applied to a pointer that already has a 'sized_by' attribute}}
void repeated_sized(int n, int m, void *buf __sized_by(n) __sized_by(m));
// expected-error@+1{{'sized_by_or_null' attribute cannot be applied to a pointer that already has a 'sized_by_or_null' attribute}}
void repeated_sized_or_null(int n, int m, void *buf __sized_by_or_null(n) __sized_by_or_null(m));

//==============================================================================
// The count is parsed where it is written
//==============================================================================

// expected-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref(int *__counted_by(count) buf, int count);
// expected-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref_or_null(int *__counted_by_or_null(count) buf, int count);
// expected-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref_sized(void *__sized_by(count) buf, int count);
// expected-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref_sized_or_null(void *__sized_by_or_null(count) buf, int count);

// eager-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref_trailing(int *buf __counted_by(count), int count);
// eager-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref_trailing_or_null(int *buf __counted_by_or_null(count), int count);
// eager-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref_trailing_sized(void *buf __sized_by(count), int count);
// eager-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref_trailing_sized_or_null(void *buf __sized_by_or_null(count), int count);

//==============================================================================
// What the count may name
//==============================================================================

// A parameter of the function, or of an enclosing function declarator.
void inner(void (*cb)(int len, int *__counted_by(len) p));
void inner_or_null(void (*cb)(int len, int *__counted_by_or_null(len) p));
void inner_sized(void (*cb)(int len, void *__sized_by(len) p));
void inner_sized_or_null(void (*cb)(int len, void *__sized_by_or_null(len) p));
void outer(int n, void (*cb)(int *__counted_by(n) p));
void outer_or_null(int n, void (*cb)(int *__counted_by_or_null(n) p));
void outer_sized(int n, void (*cb)(void *__sized_by(n) p));
void outer_sized_or_null(int n, void (*cb)(void *__sized_by_or_null(n) p));

// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void not_a_param(int *__counted_by(global_count) buf);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void not_a_param_or_null(int *__counted_by_or_null(global_count) buf);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void not_a_param_sized(void *__sized_by(global_count) buf);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void not_a_param_sized_or_null(void *__sized_by_or_null(global_count) buf);

// expected-error@+1{{'counted_by' requires a non-boolean integer type argument}}
void not_an_integer(float count, int *__counted_by(count) buf);
// expected-error@+1{{'counted_by_or_null' requires a non-boolean integer type argument}}
void not_an_integer_or_null(float count, int *__counted_by_or_null(count) buf);
// expected-error@+1{{'sized_by' requires a non-boolean integer type argument}}
void not_an_integer_sized(float count, void *__sized_by(count) buf);
// expected-error@+1{{'sized_by_or_null' requires a non-boolean integer type argument}}
void not_an_integer_sized_or_null(float count, void *__sized_by_or_null(count) buf);

// expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
void not_a_reference(int count, int *__counted_by(count + 1) buf);
// expected-error@+1{{'counted_by_or_null' argument must be a simple declaration reference}}
void not_a_reference_or_null(int count, int *__counted_by_or_null(count + 1) buf);
// expected-error@+1{{'sized_by' argument must be a simple declaration reference}}
void not_a_reference_sized(int count, void *__sized_by(count + 1) buf);
// expected-error@+1{{'sized_by_or_null' argument must be a simple declaration reference}}
void not_a_reference_sized_or_null(int count, void *__sized_by_or_null(count + 1) buf);

//==============================================================================
// Array parameters
//==============================================================================

// An array without a size decays to a pointer that carries the count, so each
// kind is allowed, although a flexible array member takes only counted_by.
void array(int count, int arr[] __counted_by(count));
void array_or_null(int count, int arr[] __counted_by_or_null(count));
void array_sized(int count, int arr[] __sized_by(count));
void array_sized_or_null(int count, int arr[] __sized_by_or_null(count));

// expected-error@+1{{'counted_by' cannot be applied to an array parameter with an explicit size}}
void array_with_size(int count, int arr[10] __counted_by(count));
// expected-error@+1{{'counted_by_or_null' cannot be applied to an array parameter with an explicit size}}
void array_with_size_or_null(int count, int arr[10] __counted_by_or_null(count));
// expected-error@+1{{'sized_by' cannot be applied to an array parameter with an explicit size}}
void array_with_size_sized(int count, int arr[10] __sized_by(count));
// expected-error@+1{{'sized_by_or_null' cannot be applied to an array parameter with an explicit size}}
void array_with_size_sized_or_null(int count, int arr[10] __sized_by_or_null(count));

// A type attribute written after the count does not hide it, so the count
// still moves to the pointer.
void array_attr_after(int count,
                      int arr[] [[clang::counted_by(count),
                                  clang::annotate_type("x")]]);
void array_attr_after(int count, int *__counted_by(count) arr);

//==============================================================================
// Indirect parameters
//==============================================================================

// A parameter may point to a counted pointer, as an out parameter does, and a
// parameter's count may dereference a parameter.
void out(int *len, int *__counted_by(*len) *buf);
void out_or_null(int *len, int *__counted_by_or_null(*len) *buf);
void out_sized(int *len, void *__sized_by(*len) *buf);
void out_sized_or_null(int *len, void *__sized_by_or_null(*len) *buf);
void out_count(int count, int *__counted_by(count) *buf);
void out_array(int count, int *__counted_by(count) buf[]);
void own_deref(int *count, int *__counted_by(*count) buf);
void own_deref_or_null(int *count, int *__counted_by_or_null(*count) buf);
void own_deref_sized(int *count, void *__sized_by(*count) buf);
void own_deref_sized_or_null(int *count, void *__sized_by_or_null(*count) buf);

// Only one level down, and not below a pointer that has a count itself.
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void two_levels(int count, int *__counted_by(count) **buf);
// expected-error@+1{{'counted_by_or_null' attribute on nested pointer type is not allowed}}
void two_levels_or_null(int count, int *__counted_by_or_null(count) **buf);
// expected-error@+1{{'sized_by' attribute on nested pointer type is not allowed}}
void two_levels_sized(int count, void *__sized_by(count) **buf);
// expected-error@+1{{'sized_by_or_null' attribute on nested pointer type is not allowed}}
void two_levels_sized_or_null(int count, void *__sized_by_or_null(count) **buf);
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void both_counted(int n, int m, int *__counted_by(n) *__counted_by(m) buf);
// expected-error@+1{{'counted_by_or_null' attribute on nested pointer type is not allowed}}
void both_counted_or_null(int n, int m, int *__counted_by_or_null(n) *__counted_by_or_null(m) buf);
// expected-error@+1{{'sized_by' attribute on nested pointer type is not allowed}}
void both_counted_sized(int n, int m, void *__sized_by(n) *__sized_by(m) buf);
// expected-error@+1{{'sized_by_or_null' attribute on nested pointer type is not allowed}}
void both_counted_sized_or_null(int n, int m, void *__sized_by_or_null(n) *__sized_by_or_null(m) buf);
// FIXME: With late parsing, a count after the declarator is parsed once the
// pointer that the parameter points to already has its count, so both stay.
// eager-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void both_counted_trailing(int *len, int n, int *__counted_by(*len) *buf __counted_by(n));

// The count of the pointer that a parameter points to names a parameter too.
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void nested_inner(int *__counted_by(global_count) *buf);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void nested_inner_or_null(int *__counted_by_or_null(global_count) *buf);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void nested_inner_sized(void *__sized_by(global_count) *buf);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void nested_inner_sized_or_null(void *__sized_by_or_null(global_count) *buf);

// So does a count on the parameter's own pointer.
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void nested_outer_global(int **__counted_by(global_count) buf);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void nested_outer_global_or_null(int **__counted_by_or_null(global_count) buf);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void nested_outer_global_sized(int **__sized_by(global_count) buf);
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void nested_outer_global_sized_or_null(int **__sized_by_or_null(global_count) buf);

// A count that comes with another declaration's type, through '__typeof__', was
// checked where it was written, so it is not nested here.
void typeof_param(int count, int *__counted_by(count) buf,
                  __typeof__(buf) **out, __typeof__(buf) (**out_paren));
struct with_count { int n; int *__counted_by(n) p; };
void typeof_field(struct with_count *s) {
  __typeof__(s->p) (**x);
  (void)x;
}

//==============================================================================
// Not supported yet
//==============================================================================

// expected-warning@+2{{a function definition without a prototype is deprecated}}
// expected-warning@+1{{'counted_by' attribute ignored}}
void knr(count, buf) int count; int *__counted_by(count) buf; {}
// expected-warning@+2{{a function definition without a prototype is deprecated}}
// expected-warning@+1{{'counted_by_or_null' attribute ignored}}
void knr_or_null(count, buf) int count; int *__counted_by_or_null(count) buf; {}
// expected-warning@+2{{a function definition without a prototype is deprecated}}
// expected-warning@+1{{'sized_by' attribute ignored}}
void knr_sized(count, buf) int count; void *__sized_by(count) buf; {}
// expected-warning@+2{{a function definition without a prototype is deprecated}}
// expected-warning@+1{{'sized_by_or_null' attribute ignored}}
void knr_sized_or_null(count, buf) int count; void *__sized_by_or_null(count) buf; {}

// expected-error@+1{{'counted_by' attribute on a function return type is not yet supported}}
void cb_return(int len, int *__counted_by(len) (*cb)(void));
// expected-error@+1{{'counted_by_or_null' attribute on a function return type is not yet supported}}
void cb_return_or_null(int len, int *__counted_by_or_null(len) (*cb)(void));
// expected-error@+1{{'sized_by' attribute on a function return type is not yet supported}}
void cb_return_sized(int len, void *__sized_by(len) (*cb)(void));
// expected-error@+1{{'sized_by_or_null' attribute on a function return type is not yet supported}}
void cb_return_sized_or_null(int len, void *__sized_by_or_null(len) (*cb)(void));
