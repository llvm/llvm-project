// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

#define __counted_by(N) __attribute__((counted_by(N)))

// A late-parsed type attribute in declarator-specifier position after a nested
// record definition used to be registered both in the enclosing record's
// field-attribute list and in its late-parsed-type-attribute list, so its
// cached tokens were parsed and freed twice -- an assertion / use-after-free.
// The point of this test is that it no longer crashes.
//
// FIXME: 'counted_by' on a non-pointer (here struct-typed) field should be
// diagnosed as "only applies to pointers or C99 flexible array members"; the
// late path currently accepts it silently. That missing diagnostic is a
// separate issue from the double-free guarded here.
struct nested_record_declspec_attr {
  struct inner1 {
    int x;
    int *p;
  } __counted_by(x) f;
  int y;
};

// A 'counted_by' type attribute on a free-function parameter has no enclosing
// record to complete it. Late-parsing it left a CountAttributedType with a
// null count expression in the AST, which crashed on serialization / PCH
// round-trip. Parameters now fall back to eager handling, so the attribute is
// resolved (or rejected) immediately instead of escaping unfinished.

// Forward reference: eager handling can't see 'n' yet, so it is diagnosed
// rather than silently building a null-count type.
void fwd_ref_param(int *__counted_by(n) p, // expected-error {{use of undeclared identifier 'n'}}
                   int n);

// FIXME: counted_by on a function parameter isn't supported yet; the eager
// decl-attribute path rejects it. What matters for this regression is that it
// is diagnosed here, not left as an unfinished type for a later crash.
void bwd_ref_param(int n,
                   int *__counted_by(n) p); // expected-error {{'counted_by' attribute only applies to non-static data members}}

// A declaration-specifier-position attribute is shared by every declarator in
// the declaration, and ConvertDeclSpecToType walks the DeclSpec's late-attribute
// list once per declarator. Building a fresh (deliberately un-uniqued)
// CountAttributedType on each walk left every field but the last holding a node
// whose count expression was never filled in -- silently, with no diagnostic --
// and any use of such a field then tripped FieldDecl::findCountedByField's
// unconditional cast<DeclRefExpr>. The attribute now reuses one node for all
// declarators, matching the eager path where uniquing has the same effect.
typedef int *shared_ptr_ty;

struct shared_declspec_attr {
  int n;
  shared_ptr_ty __counted_by(n) a, b, c;
};

// Exercising the *earlier* declarators is the point: 'c' was always fine.
void use_shared_declspec_attr(struct shared_declspec_attr *s) {
  (void)__builtin_counted_by_ref(s->a);
  (void)__builtin_counted_by_ref(s->b);
  (void)__builtin_counted_by_ref(s->c);
}

// The same, with the count declared after the fields, so the attribute really is
// late parsed rather than resolved eagerly.
struct shared_declspec_attr_fwd {
  shared_ptr_ty __counted_by(n) a, b;
  int n;
};

void use_shared_declspec_attr_fwd(struct shared_declspec_attr_fwd *s) {
  (void)__builtin_counted_by_ref(s->a);
}

// Each field is still checked in its own declaration context, so a shared
// attribute reports once per field rather than once per attribute.
union shared_declspec_attr_in_union {
  int n;
  // expected-error@+2 {{'counted_by' cannot be applied to a union member}}
  // expected-error@+1 {{'counted_by' cannot be applied to a union member}}
  shared_ptr_ty __counted_by(n) a, b;
};

// A grouping-paren declarator whose base type is already a pointer/array (via a
// typedef) used to late-parse the attribute even at file scope, where there is
// no enclosing record to complete it -- leaving a CountAttributedType with a
// null count expression in the AST (and skipping the "non-static data members"
// diagnostic entirely). ParseParenDeclarator now late-parses only inside a
// record, so at file scope the attribute is handled eagerly and rejected.
typedef int *ptr_ty;
typedef int arr_ty[4];
int global_count;
ptr_ty (__counted_by(global_count) file_ptr); // expected-error {{'counted_by' attribute only applies to non-static data members}}
arr_ty (__counted_by(global_count) file_arr); // expected-error {{'counted_by' attribute only applies to non-static data members}}

// A nested counted_by is diagnosed and dropped while the declarator is built,
// which orphans the incomplete CountAttributedType. The completion pass used to
// notice that only after re-parsing the argument, so a bad count name produced a
// second, cascading diagnostic for an attribute that was already rejected. The
// rejection is now recorded when the node is dropped, so the argument is never
// re-parsed.
struct nested_with_unresolvable_count {
  int n;
  // expected-error@+1 {{'counted_by' attribute on nested pointer type is not allowed}}
  int *__counted_by(does_not_exist) *pp;
};
