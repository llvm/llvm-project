// RUN: %clang_cc1 -x c -ftyped-memory-operations -Rtmo-remarks -Wtyped-memory-inference-failure -verify -fsyntax-only \
// RUN:            -std=c23 -include %S/Inputs/tmo_allocation_kinds.h -o - %s

// RUN: %clang_cc1 -x c -ftyped-memory-operations -std=c23 -emit-pch -o %t %S/Inputs/tmo_allocation_kinds.h
// RUN: %clang_cc1 -x c -ftyped-memory-operations -Rtmo-remarks -Wtyped-memory-inference-failure -verify -fsyntax-only \
// RUN:            -std=c23 -include-pch %t %s

static void call_in_pch_kinds_function(__SIZE_TYPE__ n) {
  in_pch_kinds_function(n);
}

void out_of_pch_kinds_function(__SIZE_TYPE__ n) {
  void *tuple_alloc = malloc(sizeof(struct KindTypeA) + sizeof(struct KindTypeB)); // #kinds_tuple
  // expected-remark@#kinds_tuple {{passing TMO information for type 'struct KindTypeA' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#kinds_tuple {{inferred tuple of ('struct KindTypeA', 'struct KindTypeB') from expression 'sizeof(struct KindTypeA) + sizeof(struct KindTypeB)'}}
  // expected-note@#kinds_tuple {{encoding 'struct KindTypeA' as 72057870300512784}}

  void *unknown_alloc = malloc(sizeof(struct KindTypeA) + n); // #kinds_unknown
  // expected-remark@#kinds_unknown {{passing TMO information for type 'struct KindTypeA' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#kinds_unknown {{inferred indeterminate set of {'struct KindTypeA'} from expression 'sizeof(struct KindTypeA) + n'}}
  // expected-note@#kinds_unknown {{encoding 'struct KindTypeA' as 72057595422605840}}
}
