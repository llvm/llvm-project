// RUN: %check_clang_tidy %s zircon-temporary-objects %t -- \
// RUN:   -config="{CheckOptions: {zircon-temporary-objects.Names: 'Foo'}}"
// RUN: %check_clang_tidy -check-suffix=BOTH %s \
// RUN:   zircon-temporary-objects,fuchsia-temporary-objects %t -- \
// RUN:   -config="{CheckOptions: {zircon-temporary-objects.Names: 'Foo', \
// RUN:                            fuchsia-temporary-objects.Names: 'Foo'}}"

class Foo {
public:
  Foo() = default;
};

void f() {
  Foo();
  // CHECK-MESSAGES: warning: 'zircon-temporary-objects' check is deprecated and will be removed in a future release; consider using 'fuchsia-temporary-objects' instead [clang-tidy-config]
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: creating a temporary object of type 'Foo' is prohibited [zircon-temporary-objects]
  // CHECK-MESSAGES-BOTH: :[[@LINE-3]]:3: warning: creating a temporary object of type 'Foo' is prohibited
}
