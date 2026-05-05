
#define _TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_malloc(__SIZE_TYPE__ size, unsigned long long);
void *malloc(__SIZE_TYPE__ size) _TYPED(typed_malloc, 1);

struct S1 {
  void *p;
  int i;
  int j;
  void (*fptr)();
};

void in_pch_function() {
  void *iptr_failed_inference = malloc(1000); // #pch_failed_inference
  // expected-warning@#pch_failed_inference {{could not infer allocation type in call to 'malloc'}}
  // expected-note@#pch_failed_inference {{unable to infer allocation type from expression '1000'}}
  int *iptr1 = malloc(sizeof(int)); // #pch_iptr1
  // expected-remark@#pch_iptr1 {{passing TMO information for type 'int' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_iptr1 {{inferred 'int' from expression 'sizeof(int)'}}
  // expected-note@#pch_iptr1 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  int *iptr2 = (int *)malloc(sizeof(int)); // #pch_iptr2
  // expected-remark@#pch_iptr2 {{passing TMO information for type 'int' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_iptr2 {{inferred 'int' from expression 'sizeof(int)'}}
  // expected-note@#pch_iptr2 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  int *iptr3 = (int *)malloc(100); // #pch_iptr3
  // expected-remark@#pch_iptr3 {{passing TMO information for type 'int' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_iptr3 {{inferred 'int' from cast of result from call to '(int *)malloc(100)'}}
  // expected-note@#pch_iptr3 {{encoding 'int' as 72057870300512784. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1384677904 }}}
  struct S1 *s1ptr1 = malloc(sizeof(struct S1)); // #pch_s1ptr1
  // expected-remark@#pch_s1ptr1 {{passing TMO information for type 'struct S1' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_s1ptr1 {{inferred 'struct S1' from expression 'sizeof(struct S1)'}}
  // expected-note@#pch_s1ptr1 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S1 *s1ptr2 = (struct S1 *)malloc(sizeof(struct S1)); // #pch_s1ptr2
  // expected-remark@#pch_s1ptr2 {{passing TMO information for type 'struct S1' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_s1ptr2 {{inferred 'struct S1' from expression 'sizeof(struct S1)'}}
  // expected-note@#pch_s1ptr2 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S1 *s1ptr3 = (struct S1 *)malloc(100); // #pch_s1ptr3
  // expected-remark@#pch_s1ptr3 {{passing TMO information for type 'struct S1' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_s1ptr3 {{inferred 'struct S1' from cast of result from call to '(struct S1 *)malloc(100)'}}
  // expected-note@#pch_s1ptr3 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
}

struct Elem {
  int value;
};

struct PrefixedArrayHeader {
  int count;
  struct Elem items[];
};

void in_pch_fam_function(__SIZE_TYPE__ n) {
  struct Elem *ep = malloc(sizeof(struct Elem) * n); // #pch_fam_array
  // expected-remark@#pch_fam_array {{passing TMO information for array of type 'struct Elem' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_fam_array {{inferred array of 'struct Elem' from expression 'sizeof(struct Elem) * n'}}
  // expected-note@#pch_fam_array {{encoding array of 'struct Elem' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}

  struct PrefixedArrayHeader *fp1 = malloc(sizeof(struct PrefixedArrayHeader) + sizeof(struct Elem) * n); // #pch_fam_explicit
  // expected-remark@#pch_fam_explicit {{passing TMO information for type 'struct PrefixedArrayHeader' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_fam_explicit {{inferred header prefixed array of {'struct PrefixedArrayHeader':'struct Elem'} from expression 'sizeof(struct PrefixedArrayHeader) + sizeof(struct Elem) * n'}}
  // expected-note@#pch_fam_explicit {{encoding 'struct PrefixedArrayHeader' as 72058694934233616. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "HeaderPrefixedArray" ] }, "TypeHash": 1384677904 }}}

  struct PrefixedArrayHeader *fp2 = malloc(sizeof(struct PrefixedArrayHeader) + n); // #pch_fam_resolved
  // expected-remark@#pch_fam_resolved {{passing TMO information for type 'struct PrefixedArrayHeader' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_fam_resolved {{inferred header prefixed array of {'struct PrefixedArrayHeader':'struct Elem'} from expression 'sizeof(struct PrefixedArrayHeader) + n'}}
  // expected-note@#pch_fam_resolved {{encoding 'struct PrefixedArrayHeader' as 72058694934233616. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "HeaderPrefixedArray" ] }, "TypeHash": 1384677904 }}}

  void *tuple1 = malloc(sizeof(struct S1) + sizeof(struct Elem)); // #pch_fam_tuple
  // expected-remark@#pch_fam_tuple {{passing TMO information for type 'struct S1' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_fam_tuple {{inferred tuple of ('struct S1', 'struct Elem') from expression 'sizeof(struct S1) + sizeof(struct Elem)'}}
  // expected-note@#pch_fam_tuple {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}

  void *unknown1 = malloc(sizeof(struct Elem) + n); // #pch_fam_unknown
  // expected-remark@#pch_fam_unknown {{passing TMO information for type 'struct Elem' to 'typed_malloc' (retargeted from 'malloc')}}
  // expected-note@#pch_fam_unknown {{inferred indeterminate set of {'struct Elem'} from expression 'sizeof(struct Elem) + n'}}
  // expected-note@#pch_fam_unknown {{encoding 'struct Elem' as 72057595422605840. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
}
