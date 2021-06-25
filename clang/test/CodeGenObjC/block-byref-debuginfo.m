// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -debug-info-kind=limited -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck %s

// CHECK: define internal void @__Block_byref_object_copy_({{.*}} !dbg ![[BYREF_COPY_SP:.*]] {
// CHECK: getelementptr inbounds {{.*}}, !dbg ![[BYREF_COPY_LOC:.*]]

// CHECK: !DILocalVariable(name: "foo", {{.*}}type: ![[FOOTY:[0-9]+]])
// CHECK: ![[FOOTY]] = {{.*}}!DICompositeType({{.*}}, name: "Foo"

// CHECK-NOT: DIFlagBlockByrefStruct
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "__block_literal_1",
// CHECK-SAME:             size: 320, elements: ![[BL_ELTS:[0-9]+]])
// CHECK: ![[BL_ELTS]] = !{{.*}}![[WFOO:[0-9]+]]}

// Test that the foo is aligned at an 8 byte boundary in the DWARF
// expression (256) that locates it inside of the byref descriptor:
// CHECK: ![[WFOO]] = !DIDerivedType(tag: DW_TAG_member, name: "foo",
// CHECK-SAME:                       baseType: ![[PTR:[0-9]+]]
// CHECK-SAME:                       offset: 256)

// CHECK: ![[PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type,
// CHECK-SAME:                      baseType: ![[WRAPPER:[0-9]+]]
// CHECK: ![[WRAPPER]] = !DICompositeType(tag: DW_TAG_structure_type, scope:
// CHECK:                                 elements: ![[WR_ELTS:[0-9]+]])
// CHECK: ![[WR_ELTS]] = !{{.*}}![[WFOO:[0-9]+]]}
// CHECK: ![[WFOO]] = !DIDerivedType(tag: DW_TAG_member, name: "foo",
// CHECK-SAME:                       baseType: ![[FOOTY]]

// CHECK: !DILocalVariable(name: "foo", {{.*}}type: ![[FOOTY]])

// CHECK: ![[BYREF_COPY_SP]] = distinct !DISubprogram(linkageName: "__Block_byref_object_copy_",
// CHECK: ![[BYREF_COPY_LOC]] = !DILocation(line: 0, scope: ![[BYREF_COPY_SP]])

struct Foo {
  unsigned char *data;
};

struct Foo2 {
  id f0;
};

void (^bptr)(void);

int func() {
  __attribute__((__blocks__(byref))) struct Foo foo;
  ^{ foo.data = 0; }();
  __block struct Foo2 foo2;
  bptr = ^{ foo2.f0 =0; };
  return 0;
}
