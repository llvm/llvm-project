// RUN: %clangxx -fsanitize=enum %s -O3 -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-PLAIN
// RUN: %clangxx -fsanitize=enum -std=c++11 -DE="class E" %s -O3 -o %t && %run %t
// RUN: %clangxx -fsanitize=enum -std=c++11 -DE="class E : bool" %s -O3 -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-BOOL

// FIXME: UBSan fails to add the correct instrumentation code for some reason on
// Windows.
// XFAIL: windows-msvc

enum E { a = 1 } e;
#undef E

int main(int argc, char **argv) {
  // memset(&e, 0xff, sizeof(e));
  for (unsigned char *p = (unsigned char*)&e; p != (unsigned char*)(&e + 1); ++p)
    *p = 0xff;

  // CHECK-PLAIN: error: load of value 4294967295, which is not a valid value for type 'enum E'
  // FIXME: Support marshalling and display of enum class values.
  // CHECK-BOOL: error: load of value <unknown>, which is not a valid value for type 'enum E'
  return (int)e != -1;
}
