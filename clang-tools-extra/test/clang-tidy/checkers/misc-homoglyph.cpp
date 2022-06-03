// RUN: %check_clang_tidy %s misc-homoglyph %t

int fo;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: fo is confusable with 𝐟o [misc-homoglyph]
int 𝐟o;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: note: other definition found here

void no() {
  int 𝐟oo;
}

void worry() {
  int foo;
}

int 𝐟i;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: 𝐟i is confusable with fi [misc-homoglyph]
int fi;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: note: other definition found here
