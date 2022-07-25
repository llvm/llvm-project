// RUN: %clangxx -fsanitize=enum %s -O3 -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=CHECK

// FIXME: UBSan fails to add the correct instrumentation code for some reason on
// Windows.
// XFAIL: windows-msvc

enum E { a = 1 };
enum class EClass { a = 1 };
enum class EBool : bool { a = 1 } e3;
enum EEmpty {};
enum EMinus { em = -1 };

int main(int argc, char **argv) {
  E e1 = static_cast<E>(0xFFFFFFFF);
  EClass e2 = static_cast<EClass>(0xFFFFFFFF);
  EEmpty e4 = static_cast<EEmpty>(1);
  EEmpty e5 = static_cast<EEmpty>(2);
  EMinus e6 = static_cast<EMinus>(1);
  EMinus e7 = static_cast<EMinus>(2);

  for (unsigned char *p = (unsigned char *)&e3; p != (unsigned char *)(&e3 + 1);
       ++p)
    *p = 0xff;

  return ((int)e1 != -1) & ((int)e2 != -1) &
         // CHECK: error: load of value 4294967295, which is not a valid value for type 'E'
         ((int)e3 != -1) & ((int)e4 == 1) &
         // CHECK: error: load of value <unknown>, which is not a valid value for type 'enum EBool'
         ((int)e5 == 2) & ((int)e6 == 1) &
         // CHECK: error: load of value 2, which is not a valid value for type 'EEmpty'
         ((int)e7 == 2);
  // CHECK: error: load of value 2, which is not a valid value for type 'EMinus'
}
