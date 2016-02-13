// RUN: %clang_cc1 -ast-print -std=c++1z %s -o - | FileCheck %s

char c = u8'1';
char d = '1';
char e = U'1';
char f = L'1';
char g = u'1';

template <char c = u8'1'>
void h();

void i() {
  h<u8'2'>();
}

char j = '\xFF';

// CHECK: char c = u8'1';
// CHECK-NEXT: char d = '1';
// CHECK-NEXT: char e = U'1';
// CHECK-NEXT: char f = L'1';
// CHECK-NEXT: char g = u'1';

// CHECK: template <char c = u8'1'>

// CHECK: h<u8'2'>();
// CHECK: char j = '\xff';
