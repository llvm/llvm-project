// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

// The counted_by family on a function parameter. The count is parsed where it
// is written, also with late parsing, so it names a parameter declared before
// the annotated one.

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))
#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

typedef int *int_ptr;
int global_count;

//==============================================================================
// Valid: the count is declared before the annotated pointer
//==============================================================================

void back_ref(int count, int *__counted_by(count) buf);
void back_ref_or_null(int count, int *__counted_by_or_null(count) buf);
void back_ref_sized(int count, void *__sized_by(count) buf);
void back_ref_sized_or_null(int count, void *__sized_by_or_null(count) buf);

// Two pointers may share a count, but a pointer has only one.
void two_buffers(int count, int *__counted_by(count) a,
                 int *__counted_by(count) b);
// expected-error@+1{{pointer cannot have more than one count attribute}}
void repeated(int n, int m, int *buf __counted_by(n) __counted_by(m));

// Written after the declarator, or in the declaration specifiers.
void trailing(int count, int *buf __counted_by(count));
void declspec(int count, int_ptr __counted_by(count) buf);

// The C23 spelling is checked the same way.
void c23_spelling(int count, int *[[clang::counted_by(count)]] buf);

int sum(int count, int *__counted_by(count) buf) {
  int total = 0;
  for (int i = 0; i < count; ++i)
    total += buf[i];
  return total;
}

//==============================================================================
// The count is parsed where it is written
//==============================================================================

// expected-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref(int *__counted_by(count) buf, int count);

// A callback field's count names the callback's parameters, not the record's
// fields, so it is not late-parsed with them.
struct callback_field {
  int n;
  void (*cb)(int len, int *__counted_by(len) p);
  // expected-error@+1{{use of undeclared identifier 'n'}}
  void (*cb_field)(int *__counted_by(n) p);
};

//==============================================================================
// What the count may name
//==============================================================================

// A parameter of the function, or of an enclosing function declarator.
void inner(void (*cb)(int len, int *__counted_by(len) p));
void outer(int n, void (*cb)(int *__counted_by(n) p));

// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void not_a_param(int *__counted_by(global_count) buf);
// expected-error@+1{{'counted_by' requires a non-boolean integer type argument}}
void not_an_integer(float count, int *__counted_by(count) buf);
// expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
void not_a_reference(int count, int *__counted_by(count + 1) buf);

//==============================================================================
// Array parameters
//==============================================================================

// An array without a size adjusts to a pointer that carries the count, so each
// kind is allowed, although a flexible array member takes only counted_by.
void array(int count, int arr[] __counted_by(count));
void array_sized_or_null(int count, int arr[] __sized_by_or_null(count));
// expected-error@+1{{'counted_by' cannot be applied to an array parameter with an explicit size}}
void array_with_size(int count, int arr[10] __counted_by(count));

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
void out_count(int count, int *__counted_by(count) *buf);
void out_array(int count, int *__counted_by(count) buf[]);
void own_deref(int *count, int *__counted_by(*count) buf);

// Only one level down, and not below a pointer that has a count itself.
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void two_levels(int count, int *__counted_by(count) **buf);
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void both_counted(int n, int m, int *__counted_by(n) *__counted_by(m) buf);

// A callback's return type is below the callback's pointer.
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void cb_return(int len, int *__counted_by(len) (*cb)(void));

// A count that comes with another declaration's type, through '__typeof__', was
// checked where it was written, so it is not nested here.
void typeof_param(int count, int *__counted_by(count) buf,
                  __typeof__(buf) **out, __typeof__(buf) (**out_paren));
struct with_count { int n; int *__counted_by(n) p; };
void typeof_field(struct with_count *s) {
  __typeof__(s->p) (**x);
  (void)x;
}
