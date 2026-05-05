// RUN: %clang_cc1 -Rtmo-remarks -verify -fsyntax-only \
// RUN:               -ftyped-memory-operations -fblocks -triple x86_64-apple-macos -nostdsysteminc -O0 %s
// RUN: %clang_cc1    -ftyped-memory-operations -fblocks -triple x86_64-apple-macos -nostdsysteminc -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck --check-prefix=CHECK          %s

#define _TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

extern "C" void *typed_func1(float, __SIZE_TYPE__ size, unsigned long long);
extern "C" void *test_func1(float, __SIZE_TYPE__ size) _TYPED(typed_func1, 2);
extern "C" void *typed_func2(__SIZE_TYPE__ size, unsigned long long, float);
extern "C" void *test_func2(__SIZE_TYPE__ size, float) _TYPED(typed_func2, 1);

struct S1 {
  void *p;
  int i;
  int j;
  void (*fptr)();
};
struct S2 {
  void *p;
  int i;
  int j;
};
struct S3 {
  void *p1;
  void *p2;
  int j;
};
struct S4 {
  void *p1;
  int j;
  void *p2;
};
struct S5 {
  int j;
  void *p1;
  void *p2;
};

extern "C" void f(void) {
  // CHECK-LABEL: define void @f()
  struct S1* alloc1 = (struct S1*)test_func1(({0;}), 100); // #alloc1
  // CHECK: call ptr @typed_func1(float {{.*}}, i64 noundef 100, i64 noundef [[S1_DESC:74309672738655766]])
  // expected-remark@#alloc1 {{passing TMO information for type 'struct S1' to 'typed_func1' (retargeted from 'test_func1')}}
  // expected-note@#alloc1 {{inferred 'struct S1' from cast of result from call to '(struct S1 *)test_func1(({}}
  // expected-note@#alloc1 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S1* alloc2 = (struct S1*)test_func2(100, ({0;})); // #alloc2
  // CHECK: call ptr @typed_func2(i64 noundef 100, i64 noundef [[S1_DESC]], float {{.*}})
  // expected-remark@#alloc2 {{passing TMO information for type 'struct S1' to 'typed_func2' (retargeted from 'test_func2')}}
  // expected-note@#alloc2 {{inferred 'struct S1' from cast of result from call to '(struct S1 *)test_func2(100, ({}}
  // expected-note@#alloc2 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S1* alloc3 = (struct S1*)test_func1((^(){return 0.0;})(), 100); // #alloc3
  // CHECK: call ptr @typed_func1(float {{.*}}, i64 noundef 100, i64 noundef [[S1_DESC]])
  // expected-remark@#alloc3 {{passing TMO information for type 'struct S1' to 'typed_func1' (retargeted from 'test_func1')}}
  // expected-note@#alloc3 {{inferred 'struct S1' from cast of result from call to '(struct S1 *)test_func1((^{ })(), 100)'}}
  // expected-note@#alloc3 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S1* alloc4 = (struct S1*)test_func2(100, ([](){return 0.0;})()); // #alloc4
  // CHECK: call ptr @typed_func2(i64 noundef 100, i64 noundef [[S1_DESC]], float {{.*}})
  // expected-remark@#alloc4 {{passing TMO information for type 'struct S1' to 'typed_func2' (retargeted from 'test_func2')}}
  // expected-note@#alloc4 {{inferred 'struct S1' from cast of result from call to '(struct S1 *)test_func2(100, ([]() {}}
  // expected-note@#alloc4 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S1* alloc5 = (struct S1*)test_func1((^(){return 0.0;})(), 100); // #alloc5
  // CHECK: call ptr @typed_func1(float {{.*}}, i64 noundef 100, i64 noundef [[S1_DESC]])
  // expected-remark@#alloc5 {{passing TMO information for type 'struct S1' to 'typed_func1' (retargeted from 'test_func1')}}
  // expected-note@#alloc5 {{inferred 'struct S1' from cast of result from call to '(struct S1 *)test_func1((^{ })(), 100)'}}
  // expected-note@#alloc5 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S1* alloc6 = (struct S1*)test_func2(100, ([](){return 0.0;})()); // #alloc6
  // CHECK: call ptr @typed_func2(i64 noundef 100, i64 noundef [[S1_DESC]], float noundef %conv15)
  // expected-remark@#alloc6 {{passing TMO information for type 'struct S1' to 'typed_func2' (retargeted from 'test_func2')}}
  // expected-note@#alloc6 {{inferred 'struct S1' from cast of result from call to '(struct S1 *)test_func2(100, ([]() {}}
  // expected-note@#alloc6 {{encoding 'struct S1' as 74309672738655766. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4009135638 }}}
  struct S2* alloc7 = (struct S2*)test_func1((^(){ // #alloc7
    // expected-remark@#alloc7 {{passing TMO information for type 'struct S2' to 'typed_func1' (retargeted from 'test_func1')}}
    // expected-note@#alloc7 {{inferred 'struct S2' from cast of result from call to '(struct S2 *)test_func1((^{ })(), 100)'}}
    // expected-note@#alloc7 {{encoding 'struct S2' as 74309672963957711. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 4234437583 }}}
    struct S3* alloc8 = (struct S3*)test_func1(({0;}), 100); // #alloc8
    // expected-remark@#alloc8 {{passing TMO information for type 'struct S3' to 'typed_func1' (retargeted from 'test_func1')}}
    // expected-note@#alloc8 {{inferred 'struct S3' from cast of result from call to '(struct S3 *)test_func1(({}}
    // expected-note@#alloc8 {{encoding 'struct S3' as 74309670376583829. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 1647063701 }}}
    return (!!alloc8) * 0.0;
  })(), 100);
  // CHECK: call ptr @typed_func1(float {{.*}}, i64 noundef 100, i64 noundef [[S2_DESC:74309672963957711]])

  struct S4* alloc9 = (struct S4*)test_func2(100, ([](){ // #alloc9
  // expected-remark@#alloc9 {{passing TMO information for type 'struct S4' to 'typed_func2' (retargeted from 'test_func2')}}
  // expected-note@#alloc9 {{inferred 'struct S4' from cast of result from call to '(struct S4 *)test_func2(100, ([]() {}} }}
  // expected-note@#alloc9 {{encoding 'struct S4' as 74309671930320854. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3200800726 }}}
    struct S5* alloc10 = (struct S5*)test_func1(({0;}), 100); // #alloc10
    // expected-remark@#alloc10 {{passing TMO information for type 'struct S5' to 'typed_func1' (retargeted from 'test_func1')}}
    // expected-note@#alloc10 {{inferred 'struct S5' from cast of result from call to '(struct S5 *)test_func1(({}}
    // expected-note@#alloc10 {{encoding 'struct S5' as 74309672024422210. { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ "FixedSize" ] }, "TypeHash": 3294902082 }}}
    return 0.0;
  })());
  // CHECK: call ptr @typed_func2(i64 noundef 100, i64 noundef [[S4_DESC:74309671930320854]], float {{.*}})
}

// CHECK-LABEL: double @__f_block_invoke_3
// CHECK: call ptr @typed_func1(float {{.*}}, i64 noundef 100, i64 noundef [[S3_DESC:74309670376583829]])

// CHECK-LABEL: double @"_ZZ1fENK3$_2clEv"
// CHECK: call ptr @typed_func1(float noundef %conv, i64 noundef 100, i64 noundef [[S5_DESC:74309672024422210]])

// CHECK: !{!"type-descriptor", !"[[S1_DESC]]", !"4009135638", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S2_DESC]]", !"4234437583", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S4_DESC]]", !"3200800726", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S3_DESC]]", !"1647063701", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
// CHECK: !{!"type-descriptor", !"[[S5_DESC]]", !"3294902082", !"\22LayoutSemantics\22: [ \22AnonymousPointer\22, \22GenericData\22 ], \22TypeFlags\22: [ ], \22TypeKind\22: \22KindC\22, \22CallsiteFlags\22: [ \22FixedSize\22 ]"}
