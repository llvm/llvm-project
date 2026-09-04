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

// Requires late parsing: `count` is not yet in scope where it is written.
void fwd_ref(int *__counted_by(count) buf, int count);
void fwd_ref_or_null(int *__counted_by_or_null(count) buf, int count);
void fwd_ref_sized(void *__sized_by(count) buf, int count);

// The annotated pointer and the count may be separated by other parameters.
void fwd_ref_interleaved(int *__counted_by(count) buf, int other, int count);

// Several parameters may refer to the same count.
void two_buffers(int *__counted_by(count) a, int *__counted_by(count) b,
                 int count);

// Both orderings in one prototype.
void mixed(int first, int *__counted_by(first) a, int *__counted_by(second) b,
           int second);

// Incomplete pointee types are allowed, as for struct fields.
void incomplete_pointee(struct size_unknown *__counted_by(count) buf,
                        int count); // ok
void const_pointee(const struct size_known *__counted_by(count) buf,
                   int count); // ok

// Binds to the `*` immediately preceding it, so it annotates the parameter.
void outer_of_two(int **__counted_by(count) buf, int count);

// Qualifiers on the annotated pointer itself do not disturb the attribute.
void qualified_ptr(int *__counted_by(count) const buf, int count); // ok

// A definition, not just a prototype.
int sum(int *__counted_by(count) buf, int count) {
  int total = 0;
  for (int i = 0; i < count; ++i)
    total += buf[i];
  return total;
}

//==============================================================================
// Valid: the attribute written after the declarator
//==============================================================================

// The spelling struct fields use. It describes the parameter's own type just as
// the spelling inside the declarator does, in either order.
void trailing_or_null(int *buf __counted_by_or_null(count), int count);
void trailing_sized(void *buf __sized_by(count), int count);
void trailing_sized_or_null(void *buf __sized_by_or_null(count), int count);

// Applies to the parameter's own pointer, as the leading spelling does.
void trailing_outer_of_two(int **buf __counted_by(count), int count);

// The count may be separated from the annotated pointer.
void trailing_interleaved(int *buf __counted_by(count), int other, int count);

// The two spellings may be mixed within one prototype.
void trailing_mixed(int *__counted_by(count) a, int *b __counted_by(count),
                    int count);

// A definition, not just a prototype.
int trailing_sum(int *buf __counted_by(count), int count) {
  int total = 0;
  for (int i = 0; i < count; ++i)
    total += buf[i];
  return total;
}

//==============================================================================
// Invalid: the attribute written after the declarator, on an unsupported type
//==============================================================================

// The trailing spelling reaches the same checks as the leading one.

// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
void trailing_on_array(int arr[10] __counted_by(count), int count);

// No declarator chunk, so it is left to the declaration-attribute path.
// expected-error@+1{{'counted_by' attribute only applies to non-static data members}}
void trailing_on_int(int count, int x __counted_by(count));
// Being eagerly parsed, it also cannot see a count declared after it.
// expected-error@+1{{use of undeclared identifier 'count'}}
void trailing_on_int_fwd(int x __counted_by(count), int count);

//==============================================================================
// Invalid: the argument does not name a parameter of this function
//==============================================================================

int global_count;

// expected-error@+1{{'counted_by' argument 'global_count' is not a parameter of the same function as the annotated pointer}}
void not_a_param(int *__counted_by(global_count) buf);

// expected-error@+1{{use of undeclared identifier 'nope'}}
void undeclared(int *__counted_by(nope) buf);

// A count on a nested prototype's parameter may name an enclosing prototype's
// parameter: the callback is only called where that parameter is in scope.
void outer_param_ok(int n, void (*cb)(int *__counted_by(n) p));

// Two levels of nesting are allowed for the same reason.
void outer_param_two_levels(int n,
                            void (*cb)(void (*inner)(int *__counted_by(n) p)));

// The same shape, but referring to the inner prototype's own parameter.
void inner_param_ok(void (*cb)(int *__counted_by(m) p, int m));

// A return type is stricter: the pointer outlives the call, so a bound naming
// the caller's parameter says nothing checkable about it.
// expected-error@+1{{'counted_by' argument 'n' is not a parameter of the same function as the annotated pointer}}
void outer_param_leak_return(int n, int *__counted_by(n) (*cb)(int m));

// Its own prototype's parameter is fine there.
void inner_param_ok_return(int *__counted_by(m) (*cb)(int m));

// Parsed at the end of the parameter clause, so this fails in lookup.
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

// Rejected as on the struct-field path: the count must itself be the reference.
// expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
void not_simple_ref_paren(int *__counted_by((count)) buf, int count);

// Meaningful under -fbounds-safety, but not supported here.
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

// Parsed eagerly and handled as declaration attributes; pre-existing behavior,
// recorded so a change is noticed.

// expected-error@+1{{use of undeclared identifier 'count'}}
void on_int(int __counted_by(count) buf, int count);

// With `count` already in scope, it reaches the declaration-attribute path,
// whose subject list only permits struct fields.
// expected-error@+1{{'counted_by' attribute only applies to non-static data members}}
void on_int_count_first(int count, int __counted_by(count) buf);

//==============================================================================
// Invalid: nested pointers are not supported for parameters
//==============================================================================

// Bounds on a pointer that is not the parameter itself are never checked.

// expected-error@+1{{'counted_by' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void nested(int *__counted_by(count) *buf, int count);

// expected-error@+1{{'sized_by' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void nested_sized(void *__sized_by(count) *buf, int count);

// expected-error@+1{{'counted_by_or_null' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void nested_or_null(int *__counted_by_or_null(count) *buf, int count);

// _Atomic is a real AtomicType node, not sugar; the walk must step through it.
// expected-error@+1{{'counted_by' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void nested_atomic(int *__counted_by(count) * _Atomic buf, int count);

//==============================================================================
// Valid: the count names a nested function-pointer prototype's own parameters
//==============================================================================

// Both its parameters and its return type may be annotated.
void inner_return(int *__counted_by(len) (*cb)(int len), int other);
void inner_return_sized(void *__sized_by(len) (*cb)(int len));
void inner_param_and_return(
    int *__counted_by(len) (*cb)(int *__counted_by(len) p, int len));

// Spelled as a function, which adjusts to a function pointer.
void inner_return_fn(int *__counted_by(len) cb(int len)); // ok

// The inner prototype's parameter shadows the outer one, and wins.
void inner_shadows_outer(int len, int *__counted_by(len) (*cb)(int len));

// A parameter shadowing a global binds to the parameter. The second needs the
// inner prototype's scope re-entered, since it is gone by then.
void shadows_global(int *__counted_by(global_count) buf, int global_count);
void inner_shadows_global(void (*cb)(int *__counted_by(global_count) p,
                                     int global_count));

// With no parameter to shadow it, the same spelling reaches the global.
// expected-error@+1{{'counted_by' argument 'global_count' is not a parameter of the same function as the annotated pointer}}
void inner_reaches_global(void (*cb)(int *__counted_by(global_count) p));

//==============================================================================
// Invalid: a count in the return type of a *no-prototype* inner function
//==============================================================================

// `()` is a FunctionNoProtoType, which declares no parameters, so no count in
// its return type can name one. Lookup still finds the enclosing prototype's.

// expected-error@+1{{'counted_by' argument 'len' is not a parameter of the same function as the annotated pointer}}
void noproto_reaches_outer_param(int *__counted_by(len) (*cb)(), int len);

// expected-error@+1{{'sized_by' argument 'len' is not a parameter of the same function as the annotated pointer}}
void noproto_reaches_outer_param_sized(void *__sized_by(len) (*cb)(), int len);

// The same shape spelled as a function rather than a function pointer.
// expected-error@+1{{'counted_by' argument 'len' is not a parameter of the same function as the annotated pointer}}
void noproto_fn_reaches_outer_param(int *__counted_by(len) cb(), int len);

// Rejected for not naming a parameter, not by the no-prototype handling.
// expected-error@+1{{'counted_by' argument 'global_count' is not a parameter of the same function as the annotated pointer}}
void noproto_reaches_global(int *__counted_by(global_count) (*cb)());

// `(void)` is a zero-parameter FunctionProtoType, so it takes that path.
// expected-error@+1{{'counted_by' argument 'len' is not a parameter of the same function as the annotated pointer}}
void protovoid_reaches_outer_param(int *__counted_by(len) (*cb)(void), int len);

//==============================================================================
// Invalid: the attribute is not on any declaration's own type
//==============================================================================

// Rejected rather than ignored: a placeholder must never reach the finalized
// AST, which asserts on it when serialized.

// Applies to the array's element type, not to the parameter.
// expected-error@+1{{'counted_by' is not supported on an array element; it must apply to the annotated declaration's own type}}
void on_array_element(int *__counted_by(count) buf[10], int count);

// The return type may only name that prototype's own parameters; it has none.
// expected-error@+1{{'counted_by' argument 'count' is not a parameter of the same function as the annotated pointer}}
void on_fn_ptr_return(int *__counted_by(count) (*cb)(void), int count);

// The same, spelled as a function parameter rather than a function pointer.
// expected-error@+1{{'counted_by' argument 'count' is not a parameter of the same function as the annotated pointer}}
void on_fn_return(int *__counted_by(count) cb(void), int count);

// An already-invalid prototype must not leave a placeholder behind. The count
// is a global, since naming the enclosing parameter would be valid here.
// expected-error@+1{{'counted_by' argument 'global_count' is not a parameter of the same function as the annotated pointer}}
void outer_then_nested(int n, void (*cb)(int *__counted_by(global_count) p));
// expected-error@+1{{'counted_by' is not supported on a nested pointer; it must apply to the annotated declaration's own type}}
void after_invalid_nested(int *__counted_by(count) *buf, int count);

//==============================================================================
// Invalid: the annotated type is not a pointer at all.
//==============================================================================

// A block pointer shares the declarator syntax but is not a pointer type, so it
// must be rejected here too rather than silently dropped.
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
void on_block_pointer(int (^__counted_by(count) blk)(void), int count);

// An _Atomic pointer is an AtomicType, so no placeholder is created at all.
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
void on_atomic_pointer(int *__counted_by(count) _Atomic p, int count);

//==============================================================================
// Invalid: the pointee type cannot support bounds computation
//==============================================================================

// Parameters share the struct-field path's pointee rules.

struct fam { int count; int elems[]; };

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'void (int)' is a function type}}
void fn_pointee(void (*__counted_by(count) fp)(int), int count);

// expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'void (int)' is a function type}}
void fn_pointee_sized(void (*__sized_by(count) fp)(int), int count);

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'struct fam' is a struct type with a flexible array member}}
void fam_pointee(struct fam *__counted_by(count) p, int count);

// A GNU extension, warned on exactly as for a struct field.
// expected-warning@+2{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
// expected-note@+1{{use '__sized_by' to suppress this warning}}
void void_pointee(void *__counted_by(count) p, int count);

//==============================================================================
// The deferred incomplete-pointee model applies to parameters as to fields
//==============================================================================

// Accepted while the pointee may still be completed; each use then requires the
// complete type.

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

//==============================================================================
// Where the attribute sits in the parameter's declarator
//==============================================================================

// The type position -- after the '*', before the identifier -- is late parsed.
void pos_type_fwd(int *__counted_by(count) buf, int count);
void pos_type_back(int count, int *__counted_by(count) buf);

// So is the position after the declarator; both give the same type.
void pos_declname_back(int count, int *buf __counted_by(count));
void pos_declname_fwd(int *buf __counted_by(count), int count);

// The declaration-specifier position is not: there it stays a declaration
// attribute, never reaching the type.

// expected-error@+1{{'counted_by' attribute only applies to non-static data members}}
void pos_declspec_back(int count, __counted_by(count) int *buf);
// expected-error@+1{{use of undeclared identifier 'count'}}
void pos_declspec_fwd(__counted_by(count) int *buf, int count);
