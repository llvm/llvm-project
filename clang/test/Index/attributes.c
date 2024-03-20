// RUN: c-index-test -test-load-source all %s | FileCheck %s

struct __attribute__((packed)) Test2 {
  char a;
};

char pure_fn() __attribute__((pure));
char const_fn() __attribute__((const));
void noduplicate_fn() __attribute__((noduplicate));

enum __attribute((flag_enum)) FlagEnum {
  Foo
};

void convergent_fn() __attribute__((convergent));

int warn_unused_result_fn() __attribute__((warn_unused_result));

struct __attribute__((warn_unused)) WarnUnused {
  int b;
};

struct __attribute__((aligned(64))) Aligned1 {
  int c;
};

struct Aligned2 {
  int c;
} __attribute__((aligned(64)));

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
// CHECK: attributes.c:12:3: EnumConstantDecl=Foo:12:3 (Definition) Extent=[12:3 - 12:6]
// CHECK: attributes.c:15:6: FunctionDecl=convergent_fn:15:6 Extent=[15:1 - 15:49]
// CHECK: attributes.c:15:37: attribute(convergent)= Extent=[15:37 - 15:47]
// CHECK: attributes.c:17:5: FunctionDecl=warn_unused_result_fn:17:5 Extent=[17:1 - 17:64]
// CHECK: attributes.c:17:44: attribute(warn_unused_result)= Extent=[17:44 - 17:62]
// CHECK: attributes.c:19:37: StructDecl=WarnUnused:19:37 (Definition) Extent=[19:1 - 21:2]
// CHECK: attributes.c:19:23: attribute(warn_unused)= Extent=[19:23 - 19:34]
// CHECK: attributes.c:23:37: StructDecl=Aligned1:23:37 (Definition) Extent=[23:1 - 25:2]
// CHECK: attributes.c:23:23: attribute(aligned)= Extent=[23:23 - 23:34]
// CHECK: attributes.c:27:8: StructDecl=Aligned2:27:8 (Definition) Extent=[27:1 - 29:2]
// CHECK: attributes.c:29:18: attribute(aligned)= Extent=[29:18 - 29:29]
