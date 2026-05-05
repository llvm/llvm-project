
#define _TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_malloc(__SIZE_TYPE__ size, unsigned long long);
void *malloc(__SIZE_TYPE__ size);

struct S1 {
  void *p;
  int i;
  int j;
  void (*fptr)();
};

void in_pch_function1() {
  int *iptr1 = malloc(sizeof(int)); // #pch_iptr1
}

void *malloc(__SIZE_TYPE__ size) _TYPED(typed_malloc, 1);

void in_pch_function2() {
  int *iptr2 = malloc(sizeof(int) * 2); // #pch_iptr2
  // expected-remark@#pch_iptr2 {{passing TMO information for array of type 'int' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_iptr2 {{inferred array of 'int' from expression 'sizeof(int) * 2'}}
  // expected-note@#pch_iptr2 {{encoding array of 'int' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
}
