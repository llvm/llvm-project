// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - %s | FileCheck %s

// Test ignorelist functionality.
// RUN: echo "src:%s=init" | sed -e 's/\\/\\\\/g' > %t-file.ignorelist
// RUN: echo "type:PODWithCtorAndDtor=init" > %t-type.ignorelist
// RUN: echo "type:NS::PODWithCtor=init" >> %t-type.ignorelist
// RUN: %clang_cc1 -fsanitize=address -fsanitize-ignorelist=%t-file.ignorelist -emit-llvm -o - %s | FileCheck %s --check-prefix=IGNORELIST
// RUN: %clang_cc1 -fsanitize=address -fsanitize-ignorelist=%t-type.ignorelist -emit-llvm -o - %s | FileCheck %s --check-prefix=IGNORELIST

struct PODStruct {
  int x;
};
PODStruct s1;

struct PODWithDtor {
  ~PODWithDtor() { }
  int x;
};
PODWithDtor s2;

struct PODWithCtorAndDtor {
  PODWithCtorAndDtor() { }
  ~PODWithCtorAndDtor() { }
  int x;
};
PODWithCtorAndDtor s3;

namespace NS {
class PODWithCtor {
public:
  PODWithCtor() {}
};

const volatile PODWithCtor array[5][5];
}

// Check that ASan init-order checking ignores structs with trivial default
// constructor.

// CHECK: @{{.*}}s1{{.*}} ={{.*}} global
// CHECK-NOT: sanitize_address_dyninit
// CHECK: @{{.*}}s2{{.*}} ={{.*}} global
// CHECK-NOT: sanitize_address_dyninit
// CHECK: @{{.*}}s3{{.*}} ={{.*}} global {{.*}}, sanitize_address_dyninit
// CHECK: @{{.*}}array{{.*}} ={{.*}} global {{.*}}, sanitize_address_dyninit

// IGNORELIST: @{{.*}}s1{{.*}} ={{.*}} global
// IGNORELIST-NOT: sanitize_address_dyninit
// IGNORELIST: @{{.*}}s2{{.*}} ={{.*}} global
// IGNORELIST-NOT: sanitize_address_dyninit
// IGNORELIST: @{{.*}}s3{{.*}} ={{.*}} global
// IGNORELIST-NOT: sanitize_address_dyninit
// IGNORELIST: @{{.*}}array{{.*}} ={{.*}} global
// IGNORELIST-NOT: sanitize_address_dyninit
