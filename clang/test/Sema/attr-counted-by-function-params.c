// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fblocks -Wpointer-arith -fsyntax-only -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))
#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

struct size_unknown;
struct size_known {
  int field;
};

//==============================================================================
// Valid: the count parameter is declared before the annotated pointer
//==============================================================================

void back_ref(int count, int *__counted_by(count) buf);
void back_ref_or_null(int count, int *__counted_by_or_null(count) buf);
void back_ref_sized(int count, void *__sized_by(count) buf);
void back_ref_sized_or_null(int count, void *__sized_by_or_null(count) buf);

//==============================================================================
// Valid: the count parameter is declared *after* the annotated pointer
//==============================================================================

// This is the case that requires late parsing: at the point the attribute is
// written, `count` is not yet in scope.

void fwd_ref(int *__counted_by(count) buf, int count);
void fwd_ref_or_null(int *__counted_by_or_null(count) buf, int count);
void fwd_ref_sized(void *__sized_by(count) buf, int count);

// The annotated pointer and the count may be separated by other parameters.
void fwd_ref_interleaved(int *__counted_by(count) buf, int other, int count);

// Several parameters may refer to the same count.
void two_buffers(int *__counted_by(count) a, int *__counted_by(count) b,
                 int count);

// A pointer may be annotated with a count declared before it and another
// pointer in the same prototype with a count declared after it.
void mixed(int first, int *__counted_by(first) a, int *__counted_by(second) b,
           int second);

// Incomplete pointee types are allowed, as for struct fields.
void incomplete_pointee(struct size_unknown *__counted_by(count) buf,
                        int count);
void const_pointee(const struct size_known *__counted_by(count) buf, int count);

// The attribute binds to the pointer formed by the `*` immediately preceding
// it, so here it applies to the parameter itself, which is a pointer.
void outer_of_two(int **__counted_by(count) buf, int count);

// Qualifiers on the annotated pointer itself do not disturb the attribute.
void qualified_ptr(int *__counted_by(count) const buf, int count);

// A definition, not just a prototype.
int sum(int *__counted_by(count) buf, int count) {
  int total = 0;
  for (int i = 0; i < count; ++i)
    total += buf[i];
  return total;
}

//==============================================================================
// Invalid: the argument does not name a parameter of this function
//==============================================================================

int global_count;

// expected-error@+1{{'counted_by' argument 'global_count' is not a parameter of the same function as the annotated pointer}}
void not_a_param(int *__counted_by(global_count) buf);

// expected-error@+1{{use of undeclared identifier 'nope'}}
void undeclared(int *__counted_by(nope) buf);

// A nested function-pointer parameter has its own parameter list. Its count
// must come from that list, not from the enclosing prototype, even though the
// enclosing prototype's parameters are in scope here.
// expected-error@+1{{'counted_by' argument 'n' is not a parameter of the same function as the annotated pointer}}
void outer_param_leak(int n, void (*cb)(int *__counted_by(n) p));

// The same shape, but referring to the inner prototype's own parameter, is OK.
void inner_param_ok(void (*cb)(int *__counted_by(m) p, int m));

// A global declared *after* the prototype is not visible either: the tokens are
// parsed at the end of the parameter clause, not at the end of the translation
// unit, so this fails in lookup rather than at the parameter check above.
// expected-error@+1{{use of undeclared identifier 'later_global'}}
void global_declared_later(int *__counted_by(later_global) buf);
int later_global;

//==============================================================================
// Invalid: the argument must be a simple declaration reference
//==============================================================================

// expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
void not_simple_ref(int *__counted_by(count + 1) buf, int count);

// expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
void not_simple_ref_deref(int *__counted_by(*count) buf, int *count);

// A parenthesized reference is rejected too, matching the struct-field path:
// the count expression itself must be the declaration reference.
// expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
void not_simple_ref_paren(int *__counted_by((count)) buf, int count);

// A constant count is meaningful under -fbounds-safety but not supported
// upstream, where the argument must name a parameter.
// expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
void count_is_literal(int *__counted_by(4) buf);

//==============================================================================
// Invalid: the count parameter must have a non-boolean integer type
//==============================================================================

// expected-error@+1{{'counted_by' requires a non-boolean integer type argument}}
void count_is_ptr(int *__counted_by(count) buf, int *count);

// expected-error@+1{{'counted_by' requires a non-boolean integer type argument}}
void count_is_float(int *__counted_by(count) buf, float count);

// expected-error@+1{{'counted_by' requires a non-boolean integer type argument}}
void count_is_bool(int *__counted_by(count) buf, _Bool count);

//==============================================================================
// Attributes in the declaration-specifiers are not part of this mechanism
//==============================================================================

// They are still parsed eagerly and handled as declaration attributes. Both
// diagnostics below are pre-existing behavior, recorded here so a future change
// to the decl-attribute path is noticed.

// Parsed eagerly, so `count` is not yet in scope.
// expected-error@+1{{use of undeclared identifier 'count'}}
void on_int(int __counted_by(count) buf, int count);

// With `count` already in scope, it reaches the declaration-attribute path,
// whose subject list only permits struct fields.
// expected-error@+1{{'counted_by' attribute only applies to non-static data members}}
void on_int_count_first(int count, int __counted_by(count) buf);

//==============================================================================
// Invalid: nested pointers are not supported for parameters
//==============================================================================

// Bounds on a pointer that is not the parameter itself are neither checked nor
// updated by anything, so reject rather than silently accept.

// expected-error@+1{{'counted_by' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void nested(int *__counted_by(count) *buf, int count);

// expected-error@+1{{'sized_by' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void nested_sized(void *__sized_by(count) *buf, int count);

// expected-error@+1{{'counted_by_or_null' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void nested_or_null(int *__counted_by_or_null(count) *buf, int count);

// An _Atomic qualifier on the outer pointer becomes a real AtomicType node
// above the placeholder, not sugar; the placeholder walk must step through it
// or the attribute would escape resolution entirely.
// expected-error@+1{{'counted_by' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void nested_atomic(int *__counted_by(count) * _Atomic buf, int count);

//==============================================================================
// Valid: the count names a nested function-pointer prototype's own parameters
//==============================================================================

// A count written inside a function-pointer parameter's own prototype names
// that prototype's parameters rather than this function's. Both the parameters
// and the return type of the inner prototype may be annotated this way.

void inner_return(int *__counted_by(len) (*cb)(int len), int other);
void inner_return_sized(void *__sized_by(len) (*cb)(int len));
void inner_param_and_return(int *__counted_by(len) (*cb)(int *__counted_by(len) p,
                                                         int len));

// A parameter spelled as a function rather than a function pointer behaves the
// same, since it adjusts to a function pointer.
void inner_return_fn(int *__counted_by(len) cb(int len));

// The inner prototype's parameter shadows an outer name of the same name, and
// the inner one wins.
void inner_shadows_outer(int len, int *__counted_by(len) (*cb)(int len));

// A parameter shadowing a *global* of the same name binds to the parameter. The
// parameter list alone could not produce this: it can reject what lookup found,
// but only a scope can make lookup prefer the parameter over the file-scope
// declaration. The second one needs that scope to be *re-entered*, since the
// inner prototype's own scope is long gone by the time its count is parsed.
// Accepting these is what proves the count bound to the parameter; had it
// bound to `global_count`, the parameter check would have rejected it, exactly
// as it does for `not_a_param` above.
void shadows_global(int *__counted_by(global_count) buf, int global_count);
void inner_shadows_global(void (*cb)(int *__counted_by(global_count) p,
                                     int global_count));

// ... and without a parameter to shadow it, the same spelling in a nested
// prototype still reaches the global and is still rejected.
// expected-error@+1{{'counted_by' argument 'global_count' is not a parameter of the same function as the annotated pointer}}
void inner_reaches_global(void (*cb)(int *__counted_by(global_count) p));

//==============================================================================
// Invalid: a count in the return type of a *no-prototype* inner function
//==============================================================================

// `()` is a FunctionNoProtoType in C, not a zero-parameter FunctionProtoType,
// so it declares no parameters at all and no count written in its return type
// can name one.
//
// The enclosing prototype's parameters are in scope here, so lookup does find
// them; what makes these ill-formed is that they do not belong to the inner
// function, whose callers cannot see them.

// expected-error@+1{{'counted_by' argument 'len' is not a parameter of the same function as the annotated pointer}}
void noproto_reaches_outer_param(int *__counted_by(len) (*cb)(), int len);

// expected-error@+1{{'sized_by' argument 'len' is not a parameter of the same function as the annotated pointer}}
void noproto_reaches_outer_param_sized(void *__sized_by(len) (*cb)(), int len);

// The same shape spelled as a function rather than a function pointer.
// expected-error@+1{{'counted_by' argument 'len' is not a parameter of the same function as the annotated pointer}}
void noproto_fn_reaches_outer_param(int *__counted_by(len) cb(), int len);

// A no-prototype inner function reaching a global is rejected too, though that
// falls out of the count not naming a parameter at all rather than of the
// no-prototype handling.
// expected-error@+1{{'counted_by' argument 'global_count' is not a parameter of the same function as the annotated pointer}}
void noproto_reaches_global(int *__counted_by(global_count) (*cb)());

// `(void)` *is* a zero-parameter FunctionProtoType, so it takes the prototype
// path. Kept alongside the cases above so a regression that conflates the two
// spellings shows up as a diagnostic appearing on only one of them.
// expected-error@+1{{'counted_by' argument 'len' is not a parameter of the same function as the annotated pointer}}
void protovoid_reaches_outer_param(int *__counted_by(len) (*cb)(void), int len);

//==============================================================================
// Invalid: the attribute is not on any declaration's own type
//==============================================================================

// These positions have to be rejected rather than ignored: the placeholder the
// parser leaves behind must never reach the finalized AST, which asserts on it
// when the AST is serialized.

// The attribute applies to the array's element type, not to the parameter.
// expected-error@+1{{'counted_by' is not supported on an array element; it must apply to the annotated declaration's own type}}
void on_array_element(int *__counted_by(count) buf[10], int count);

// A function-pointer parameter's return type may only name that prototype's own
// parameters, and this prototype has none.
// expected-error@+1{{'counted_by' argument 'count' is not a parameter of the same function as the annotated pointer}}
void on_fn_ptr_return(int *__counted_by(count) (*cb)(void), int count);

// The same, spelled as a function parameter rather than a function pointer.
// expected-error@+1{{'counted_by' argument 'count' is not a parameter of the same function as the annotated pointer}}
void on_fn_return(int *__counted_by(count) cb(void), int count);

// A prototype that is already invalid must not leave a placeholder behind for
// the enclosing prototype to trip over.
// expected-error@+1{{'counted_by' argument 'n' is not a parameter of the same function as the annotated pointer}}
void outer_then_nested(int n, void (*cb)(int *__counted_by(n) p));
// expected-error@+1{{'counted_by' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void after_invalid_nested(int *__counted_by(count) *buf, int count);

//==============================================================================
// Invalid: the annotated type is not a pointer at all.
//==============================================================================

// A block pointer is written with the same declarator syntax as a pointer, and
// so collects late-parsed attributes the same way, but it is not a pointer
// type. Rejecting it has to happen on the late-parsed path too: if the
// attribute were simply dropped there, turning the flag on would make clang
// accept what it otherwise diagnoses.
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
void on_block_pointer(int (^__counted_by(count) blk)(void), int count);

// An _Atomic pointer is an AtomicType, not a pointer type, so no placeholder
// is created for it in the first place.
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
void on_atomic_pointer(int *__counted_by(count) _Atomic p, int count);

//==============================================================================
// Invalid: the pointee type cannot support bounds computation
//==============================================================================

// Parameters share the struct-field path's pointee rules, so both accept and
// reject the same shapes.

struct fam { int count; int elems[]; };

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'void (int)' is a function type}}
void fn_pointee(void (*__counted_by(count) fp)(int), int count);

// expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'void (int)' is a function type}}
void fn_pointee_sized(void (*__sized_by(count) fp)(int), int count);

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'struct fam' is a struct type with a flexible array member}}
void fam_pointee(struct fam *__counted_by(count) p, int count);

// counted_by on a pointer to void counts bytes, like sized_by; a GNU
// extension, warned on exactly as for a struct field.
// expected-warning@+2{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
// expected-note@+1{{use '__sized_by' to suppress this warning}}
void void_pointee(void *__counted_by(count) p, int count);

//==============================================================================
// The deferred incomplete-pointee model applies to parameters as to fields
//==============================================================================

// The attribute is accepted while the pointee may still be completed later, and
// each use or assignment then requires the complete type.

// expected-note@+1 2{{consider providing a complete definition for 'struct deferred_pointee'}}
struct deferred_pointee;

void take_deferred(struct deferred_pointee *q);

// expected-note@+1{{consider using '__sized_by' instead of '__counted_by'}}
void deferred_use(struct deferred_pointee *__counted_by(count) p, int count) {
  // expected-error@+1{{cannot use 'p' with '__counted_by' attributed type 'struct deferred_pointee * __counted_by(count)' (aka 'struct deferred_pointee *') because the pointee type 'struct deferred_pointee' is incomplete}}
  take_deferred(p);
}

// expected-note@+1{{consider using '__sized_by' instead of '__counted_by'}}
void deferred_assign(struct deferred_pointee *__counted_by(count) p, int count,
                     struct deferred_pointee *q) {
  // expected-error@+1{{cannot assign to 'p' with '__counted_by' attributed type 'struct deferred_pointee * __counted_by(count)' (aka 'struct deferred_pointee *') because the pointee type 'struct deferred_pointee' is incomplete}}
  p = q;
}
