// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -fwchar-type=short -fno-signed-wchar %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=ITANIUM
// RUN: %clang_cc1 -triple %ms_abi_triple -emit-llvm -fwchar-type=short -fno-signed-wchar %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=MSABI

// Run in C mode as wide multichar literals are not valid in C++

// XFAIL: target=hexagon-{{.*}}
// Hexagon aligns arrays of size 8+ bytes to a 64-bit boundary, which fails
// the first check line with "align 1".

typedef __WCHAR_TYPE__ wchar_t;

int main(void) {
  // This should convert to utf8.
  // CHECK: private unnamed_addr constant [10 x i8] c"\E1\84\A0\C8\A0\F4\82\80\B0\00", align 1
  char b[10] = "\u1120\u0220\U00102030";

  // ITANIUM: private unnamed_addr constant [3 x i16] [i16 65, i16 66, i16 0]
  // MSABI: linkonce_odr dso_local unnamed_addr constant [3 x i16] [i16 65, i16 66, i16 0]
  const wchar_t *foo = L"AB";

  // This should convert to utf16.
  // ITANIUM: private unnamed_addr constant [5 x i16] [i16 4384, i16 544, i16 -9272, i16 -9168, i16 0]
  // MSABI: linkonce_odr dso_local unnamed_addr constant [5 x i16] [i16 4384, i16 544, i16 -9272, i16 -9168, i16 0]
  const wchar_t *bar = L"\u1120\u0220\U00102030";

  // Should pick second character.
  // CHECK: store i8 98
  char c = 'ab';

  // CHECK: store i16 97
  wchar_t wa = L'a';

  // -4085 == 0xf00b
  // CHECK: store i16 -4085
  wchar_t wc = L'\uF00B';
}
