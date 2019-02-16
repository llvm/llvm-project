// Run lines below; this test is line- and column-sensitive.

void foo(void) __attribute__((availability(macosx,introduced=10.4,deprecated=10.5,obsoleted=10.7), availability(ios,introduced=3.2,deprecated=4.1)));

enum {
  old_enum
} __attribute__((deprecated));

enum {
  old_enum_plat
} __attribute__((availability(macosx,introduced=10.4,deprecated=10.5,obsoleted=10.7)));

void bar(void) __attribute__((availability(macosx,introduced=10.4))) __attribute__((availability(macosx,obsoleted=10.6))) __attribute__((availability(ios,introduced=3.2))) __attribute__((availability(macosx,deprecated=10.5,message="use foobar")));

void bar2(void) __attribute__((availability(macosx,introduced=10.4,deprecated=10.5,obsoleted=10.7))) __attribute__((availability(ios,introduced=3.2,deprecated=10.0))) __attribute__((availability(macosx,introduced=10.4,deprecated=10.5,obsoleted=10.7))) __attribute__((availability(ios,introduced=3.2,deprecated=10.0)));

void foo2(void) __attribute__((availability(swift,unavailable)));
void foo3(void) __attribute__((availability(swift,deprecated)));

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: FunctionDecl=foo:3:6{{.*}}(ios, introduced=3.2, deprecated=4.1) (macos, introduced=10.4, deprecated=10.5, obsoleted=10.7)
// CHECK: EnumConstantDecl=old_enum:6:3 (Definition) (deprecated)
// CHECK: EnumConstantDecl=old_enum_plat:10:3 {{.*}} (macos, introduced=10.4, deprecated=10.5, obsoleted=10.7)
// CHECK: FunctionDecl=bar:13:6{{.*}}(ios, introduced=3.2) (macos, introduced=10.4, deprecated=10.5, obsoleted=10.6, message="use foobar")
// CHECK: FunctionDecl=bar2:15:6{{.*}}(ios, introduced=3.2, deprecated=10.0) (macos, introduced=10.4, deprecated=10.5, obsoleted=10.7)
// CHECK: FunctionDecl=foo2:17:6{{.*}}(swift, unavailable)
// CHECK: FunctionDecl=foo3:18:6{{.*}}(swift, deprecated=1)
