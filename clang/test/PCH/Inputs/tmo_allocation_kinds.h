
#define _TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_malloc(__SIZE_TYPE__ size, unsigned long long);
void *malloc(__SIZE_TYPE__ size) _TYPED(typed_malloc, 1);

struct KindTypeA {
  int value;
};

struct KindTypeB {
  int count;
  void *ptr;
};

void in_pch_kinds_function(__SIZE_TYPE__ n) {
  void *tuple_alloc = malloc(sizeof(struct KindTypeA) + sizeof(struct KindTypeB)); // #pch_kinds_tuple
  // expected-remark@#pch_kinds_tuple {{passing TMO information for type 'struct KindTypeA' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_kinds_tuple {{inferred tuple of ('struct KindTypeA', 'struct KindTypeB') from expression 'sizeof(struct KindTypeA) + sizeof(struct KindTypeB)'}}
  // expected-note@#pch_kinds_tuple {{encoding 'struct KindTypeA' as 72057870300512784}}

  void *unknown_alloc = malloc(sizeof(struct KindTypeA) + n); // #pch_kinds_unknown
  // expected-remark@#pch_kinds_unknown {{passing TMO information for type 'struct KindTypeA' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_kinds_unknown {{inferred indeterminate set of {'struct KindTypeA'} from expression 'sizeof(struct KindTypeA) + n'}}
  // expected-note@#pch_kinds_unknown {{encoding 'struct KindTypeA' as 72057595422605840}}
}
