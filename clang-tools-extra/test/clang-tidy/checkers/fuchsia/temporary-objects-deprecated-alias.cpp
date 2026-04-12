// RUN: clang-tidy %s -checks='-*,zircon-temporary-objects' \
// RUN:   -config="{CheckOptions: {zircon-temporary-objects.Names: 'Foo'}}" \
// RUN:   -header-filter='.*' -- 2>&1 | FileCheck %s --check-prefix=CHECK-OLD
// RUN: clang-tidy %s -checks='-*,zircon-temporary-objects,fuchsia-temporary-objects' \
// RUN:   -config="{CheckOptions: {zircon-temporary-objects.Names: 'Foo', \
// RUN:                            fuchsia-temporary-objects.Names: 'Foo'}}" \
// RUN:   -header-filter='.*' -- 2>&1 | FileCheck %s --check-prefix=CHECK-BOTH \
// RUN:   -implicit-check-not='deprecated and will be removed in a future release'

class Foo {
public:
  Foo() = default;
};

void f() {
  Foo();
  // CHECK-OLD: warning: zircon-temporary-objects is deprecated and will be removed in a future release; consider using fuchsia-temporary-objects instead [clang-tidy-config]
  // CHECK-OLD: :[[@LINE-2]]:3: warning: creating a temporary object of type 'Foo' is prohibited [zircon-temporary-objects]
  // CHECK-BOTH: :[[@LINE-3]]:3: warning: creating a temporary object of type 'Foo' is prohibited
}
