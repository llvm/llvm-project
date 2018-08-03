// RUN: c-index-test -test-load-source all %s | FileCheck %s

struct __attribute__((packed)) Test2 {
  char a;
};

void pure_fn() __attribute__((pure));
void const_fn() __attribute__((const));
void noduplicate_fn() __attribute__((noduplicate));

enum __attribute((flag_enum)) FlagEnum {
  Foo
};

// CHECK: attributes.c:3:32: StructDecl=Test2:3:32 (Definition) Extent=[3:1 - 5:2]
// CHECK: attributes.c:3:23: attribute(packed)=packed Extent=[3:23 - 3:29]
// CHECK: attributes.c:4:8: FieldDecl=a:4:8 (Definition) Extent=[4:3 - 4:9] [access=public]

// CHECK: attributes.c:7:6: FunctionDecl=pure_fn:7:6 Extent=[7:1 - 7:37]
// CHECK: attributes.c:7:31: attribute(pure)= Extent=[7:31 - 7:35]
// CHECK: attributes.c:8:6: FunctionDecl=const_fn:8:6 Extent=[8:1 - 8:39]
// CHECK: attributes.c:8:32: attribute(const)= Extent=[8:32 - 8:37]
// CHECK: attributes.c:9:6: FunctionDecl=noduplicate_fn:9:6 Extent=[9:1 - 9:51]
// CHECK: attributes.c:9:38: attribute(noduplicate)= Extent=[9:38 - 9:49]
// CHECK: attributes.c:11:31: EnumDecl=FlagEnum:11:31 (Definition) Extent=[11:1 - 13:2]
// CHECK: attributes.c:11:19: attribute(flag_enum)= Extent=[11:19 - 11:28]
