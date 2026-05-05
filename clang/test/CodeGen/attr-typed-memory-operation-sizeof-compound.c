// RUN: %clang_cc1 -Wtyped-memory-inference-failure -ftyped-memory-operations -DTMO=1 \
// RUN:            -triple x86_64-apple-macos -nostdsysteminc -O0 -fsyntax-only \
// RUN:            -verify -Rtmo-remarks %s
// RUN: %clang_cc1 -ftyped-memory-operations -DTMO=1 -triple x86_64-apple-macos -nostdsysteminc -O0 \
// RUN:            -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fno-typed-memory-operations -DTMO=0 -triple x86_64-apple-macos -nostdsysteminc -O0 \
// RUN:            -disable-llvm-passes -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DISABLED %s

#define _TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_real_malloc(__SIZE_TYPE__ size, unsigned long long);
void *malloc(__SIZE_TYPE__ size) _TYPED(typed_real_malloc, 1);

struct Header {
  int count;
  void *base;
};

struct Element {
  int data;
};

// CHECK-LABEL: @test_header_prefixed_array
void *test_header_prefixed_array(__SIZE_TYPE__ n) {
  return malloc(sizeof(struct Header) + sizeof(struct Element) * n);
  // expected-remark@-1 {{passing TMO information for type 'struct Header' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@-2 {{inferred header prefixed array of {'struct Header':'struct Element'} from expression 'sizeof(struct Header) + sizeof(struct Element) * n'}}
  // expected-note@-3 {{encoding 'struct Header' as 74310495310558338. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "HeaderPrefixedArray" ] }, "TypeHash": 1947317378 }}}
  // CHECK:       call ptr @typed_real_malloc(i64 noundef %add, i64 noundef 74310495310558338)
  // CHECK-DISABLED: call ptr @malloc
}

// CHECK-LABEL:  @test_header_prefixed_array_commuted
void *test_header_prefixed_array_commuted(__SIZE_TYPE__ n) {
  return malloc(sizeof(struct Element) * n + sizeof(struct Header));
  // expected-remark@-1 {{passing TMO information for type 'struct Header' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@-2 {{inferred header prefixed array of {'struct Header':'struct Element'} from expression 'sizeof(struct Element) * n + sizeof(struct Header)'}}
  // expected-note@-3 {{encoding 'struct Header' as 74310495310558338. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "HeaderPrefixedArray" ] }, "TypeHash": 1947317378 }}}
  // CHECK:       call ptr @typed_real_malloc(i64 noundef %add, i64 noundef 74310495310558338)
  // CHECK-DISABLED: call ptr @malloc
}

// CHECK-LABEL: @test_multi_header_prefixed_array
void *test_multi_header_prefixed_array(__SIZE_TYPE__ n) {
  return malloc(sizeof(struct Header) + sizeof(int) + sizeof(struct Element) * n);
  // expected-remark@-1 {{passing TMO information for type 'struct Header' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@-2 {{inferred header prefixed array of {'struct Header':'struct Element'} from expression 'sizeof(struct Header) + sizeof(int) + sizeof(struct Element) * n'}}
  // expected-note@-3 {{encoding 'struct Header' as 74310495310558338. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "HeaderPrefixedArray" ] }, "TypeHash": 1947317378 }}}
  // CHECK:       call ptr @typed_real_malloc(i64 noundef %add, i64 noundef 74310495310558338)
  // CHECK-DISABLED: call ptr @malloc
}

// CHECK-LABEL: @test_tuple_different_types
void *test_tuple_different_types(void) {
  return malloc(sizeof(struct Header) + sizeof(struct Element));
  // expected-remark@-1 {{passing TMO information for type 'struct Header' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@-2 {{inferred tuple of ('struct Header', 'struct Element') from expression 'sizeof(struct Header) + sizeof(struct Element)'}}
  // expected-note@-3 {{encoding 'struct Header' as 74309670676837506. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1947317378 }}}
  // CHECK: call ptr @typed_real_malloc(i64 noundef 20, i64 noundef 74309670676837506)
  // CHECK-DISABLED: call ptr @malloc
}

// CHECK-LABEL: define {{.*}} @test_var_sizeof_header_array
void *test_var_sizeof_header_array(__SIZE_TYPE__ n) {
  __SIZE_TYPE__ hdr_sz = sizeof(struct Header);
  return malloc(hdr_sz + sizeof(struct Element) * n);
  // expected-remark@-1 {{passing TMO information for type 'struct Element' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@-2 {{inferred indeterminate set of {'struct Element'} from expression 'hdr_sz + sizeof(struct Element) * n'}}
  // expected-note@-3 {{encoding 'struct Element' as 72057595422605840. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
  // CHECK: call ptr @typed_real_malloc(i64 noundef %add, i64 noundef 72057595422605840)
  // CHECK-DISABLED: call ptr @malloc
}

// CHECK-LABEL: @test_homogeneous_header_array
void *test_homogeneous_header_array(__SIZE_TYPE__ n) {
  return malloc(sizeof(struct Header) + sizeof(struct Header) * n);
  // expected-remark@-1 {{passing TMO information for type 'struct Header' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@-2 {{inferred header prefixed array of {'struct Header':'struct Header'} from expression 'sizeof(struct Header) + sizeof(struct Header) * n'}}
  // expected-note@-3 {{encoding 'struct Header' as 74310495310558338. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "HeaderPrefixedArray" ] }, "TypeHash": 1947317378 }}}
  // CHECK: call ptr @typed_real_malloc(i64 noundef %add, i64 noundef 74310495310558338)
  // CHECK-DISABLED: call ptr @malloc
}

void *test_prefixed_array_comma(__SIZE_TYPE__ n) {
  return malloc(((void)0, sizeof(struct Header) + sizeof(struct Element) * n));
  // expected-remark@-1 {{passing TMO information for type 'struct Header' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@-2 {{inferred header prefixed array of {'struct Header':'struct Element'} from expression '(void)0 , sizeof(struct Header) + sizeof(struct Element) * n'}}
  // expected-note@-3 {{encoding 'struct Header' as 74310495310558338. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "HeaderPrefixedArray" ] }, "TypeHash": 1947317378 }}}
}

void *test_array_comma(__SIZE_TYPE__ n) {
  return malloc(((void)0, sizeof(struct Element) * n));
  // expected-remark@-1 {{passing TMO information for array of type 'struct Element' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@-2 {{inferred array of 'struct Element' from expression '(void)0 , sizeof(struct Element) * n'}}
  // expected-note@-3 {{encoding array of 'struct Element' as 72058145178419728. { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "Array" ] }, "TypeHash": 1384677904 }}}
}

void *test_tuple_comma(void) {
  return malloc(((void)0, sizeof(struct Header) + sizeof(struct Element)));
  // expected-remark@-1 {{passing TMO information for type 'struct Header' to 'typed_real_malloc' (retargeted from 'malloc')}}
  // expected-note@-2 {{inferred tuple of ('struct Header', 'struct Element') from expression '(void)0 , sizeof(struct Header) + sizeof(struct Element)'}}
  // expected-note@-3 {{encoding 'struct Header' as 74309670676837506. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1947317378 }}}
}
