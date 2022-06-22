// RUN: %check_clang_tidy %s misc-confusable-identifiers %t

int fo;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: fo is confusable with 𝐟o [misc-confusable-identifiers]
int 𝐟o;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: note: other declaration found here

void no() {
  int 𝐟oo;
}

void worry() {
  int foo;
}

int 𝐟i;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: 𝐟i is confusable with fi [misc-confusable-identifiers]
int fi;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: note: other declaration found here

// should not print anything
namespace ns {
struct Foo {};
} // namespace ns
auto f = ns::Foo();
