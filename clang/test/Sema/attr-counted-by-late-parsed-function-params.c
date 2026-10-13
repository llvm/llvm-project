// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fblocks -fsyntax-only -verify -x objective-c %s

// The counted_by family on a function parameter, where late parsing changes
// the result. A parameter clause completes the counts written in it at its
// closing parenthesis, so a count may name a parameter declared after the
// annotated one.

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

typedef int *int_ptr;
int global_count;

//==============================================================================
// Valid: the count is declared after the annotated pointer
//==============================================================================

void fwd_ref(int *__counted_by(count) buf, int count);
void fwd_ref_sized_or_null(void *__sized_by_or_null(count) buf, int count);
void variadic(int *__counted_by(count) buf, int count, ...);

// Qualifiers and nullability on the annotated pointer do not disturb it.
void qualified_ptr(int *__counted_by(count) const buf, int count);
void nullable_ptr(int *__counted_by(count) _Nonnull buf, int count);

// Written after the declarator, it describes the parameter's own type too.
void trailing(int *buf __counted_by(count), int count);
void trailing_outer_of_two(int **buf __counted_by(count), int count);

// In the declaration specifiers it applies to the type they name.
void on_ptr_typedef(int_ptr __counted_by(count) buf, int count);

int sum(int *__counted_by(count) buf, int count) {
  int total = 0;
  for (int i = 0; i < count; ++i)
    total += buf[i];
  return total;
}

//==============================================================================
// Array parameters
//==============================================================================

// The count moves to the adjusted pointer when it is completed.
void array(int arr[] __counted_by(count), int count);

// Qualifiers on the array apply to its element (C99 6.7.3p8), also when they
// are outside the count, as when it is written in the declaration specifiers.
typedef int int_array[];
void array_typedef_const(const int_array __counted_by(count) arr, int count) {
  arr[0] = 1; // expected-error{{read-only variable is not assignable}}
}

// The count describes the pointer's type for its redeclarations too.
void array_redecl(int arr[] __counted_by(count), int count);
void array_redecl(int *__counted_by(count) arr, int count);

//==============================================================================
// What the count may name
//==============================================================================

// Parsed at the end of the parameter clause, not of the translation unit.
// expected-error@+1{{use of undeclared identifier 'later_global'}}
void global_declared_later(int *__counted_by(later_global) buf);
int later_global;

// A parameter shadowing a global is the one named.
void shadows_global(int *__counted_by(global_count) buf, int global_count);

// A count on a callback's parameter may name its own parameters, and those of
// an enclosing prototype that are already declared.
void inner_param(void (*cb)(int *__counted_by(m) p, int m));
void outer_param_two_levels(int n,
                            void (*cb)(void (*inner)(int *__counted_by(n) p)));
// The callback's clause ends before the enclosing one declares 'n'.
// expected-error@+1{{use of undeclared identifier 'n'}}
void outer_param_later(void (*cb)(int *__counted_by(n) p), int n);

// An argument that fails to parse is diagnosed once.
// expected-error@+1{{expected expression}}
void unparseable(int *__counted_by(+) buf, int count);

//==============================================================================
// Positions that are not the parameter's own pointer
//==============================================================================

// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
void block_pointer(int (^__counted_by(count) blk)(void), int count);
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
void atomic_pointer(int *__counted_by(count) _Atomic p, int count);

//==============================================================================
// Indirect parameters
//==============================================================================

// A parameter may point to a counted pointer, as an out parameter does.
void out(int *__counted_by(*len) *buf, int *len);
void out_typedef(int_ptr __counted_by(count) *buf, int count);

// Only one level down. A nested count is dropped, keeping the parentheses
// around it.
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void two_levels_paren(int *__counted_by(count) *(*buf), int count);
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void two_levels_attr(int *__counted_by(count) (__attribute__((btf_type_tag("t"))) **buf), int count);
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void two_levels_array(int *__counted_by(count) buf[2][2], int count);
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void nested_array(int_array __counted_by(count) *buf, int count);
// A count that comes with another parameter's type, through '__typeof__', was
// checked where it was written, so it is not nested here. In the late order,
// 'buf' still completes its own count.
void typeof_two_levels(int *__counted_by(count) buf, __typeof__(buf) **out,
                       int count);
// An array that fails to build drops the count on its elements with it.
// expected-error@+1{{'buf' declared as an array with a negative size}}
void invalid_array(int *__counted_by(count) buf[-1], int count);

//==============================================================================
// Return types of callbacks
//==============================================================================

// A count on a callback's return type names that callback's own parameters.
void cb_return(int *__counted_by(len) (*cb)(int len), int other);
void cb_return_and_param(
    int *__counted_by(len) (*cb)(int *__counted_by(len) p, int len));
void cb_return_shadows_outer(int len, int *__counted_by(len) (*cb)(int len));
// A count in the declaration specifiers is not parsed with a callback's
// parameters.
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void cb_return_declspec(int_ptr __counted_by(len) (*cb)(int len));

// It is parsed with the callback's type, before 'len' is declared.
// expected-error@+1{{use of undeclared identifier 'len'}}
void cb_return_later_outer(int *__counted_by(len) (*cb)(void), int len);
// A callback without a prototype has no parameters to name.
// expected-error@+1{{use of undeclared identifier 'len'}}
void cb_return_no_proto(int *__counted_by(len) (*cb)(), int len);

// A parameter declared with a function type adjusts to a function pointer,
// which the count is taken to describe.
// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'int * __counted_by(len)(int)' (aka 'int *(int)') is a function type}}
void fn_typed(int *__counted_by(len) cb(int len));
// One that comes with the function type, through '__typeof__', stays on its
// return type.
struct callback_ret { int *__counted_by(m) (*ret)(int m); };
void fn_typed_typeof(__typeof__(*((struct callback_ret *)0)->ret) cb);
void fn_typed_typeof_param(int *__counted_by(m) (*cb)(int m),
                           __typeof__(*cb) cb2);

// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void cb_return_nested(int *__counted_by(len) *(*cb)(int len));
// Only a parameter's count may be a dereferenced parameter.
// expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
void cb_return_deref(int *__counted_by(*len) (*cb)(int *len));

//==============================================================================
// Attributes after the '(' of an unnamed callback
//==============================================================================

// They belong to the callback's first parameter.
void unnamed_first(void (__counted_by(n) int_ptr, int n));
void unnamed_first_outer(int n, void (__counted_by(n) int_ptr));
// expected-error@+1{{argument required after attribute}}
void unnamed_empty(void (__counted_by(n)));

//==============================================================================
// Records defined in a parameter clause complete their own members
//==============================================================================

// expected-warning@+1{{declaration of 'struct in_params' will not be visible outside of this function}}
void record_in_params(struct in_params { int *__counted_by(n) p; int n; } *s);
