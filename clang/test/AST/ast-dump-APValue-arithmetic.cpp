// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN:            -ast-dump %s -ast-dump-filter Test \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN:           -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s

void Test() {
  constexpr int Int = 42;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} Int 'const int' constexpr cinit
  // CHECK-NEXT:  |   |-value: Int 42

  constexpr __int128 Int128 = (__int128)0xFFFFFFFFFFFFFFFF + (__int128)1;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} Int128 'const __int128' constexpr cinit
  // CHECK-NEXT:  |   |-value: Int 18446744073709551616

  constexpr float Float = 3.1415f;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} Float 'const float' constexpr cinit
  // CHECK-NEXT:  |   |-value: Float 3.141500e+00

  constexpr double Double = 3.1415f;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} Double 'const double' constexpr cinit
  // CHECK-NEXT:  |   |-value: Float 3.141500e+00

  constexpr _Complex int ComplexInt = 42 + 24i;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} referenced ComplexInt 'const _Complex int' constexpr cinit
  // CHECK-NEXT:  |   |-value: ComplexInt 42 + 24i

  constexpr _Complex float ComplexFloat = 3.1415f + 42i;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} referenced ComplexFloat 'const _Complex float' constexpr cinit
  // CHECK-NEXT:  |   |-value: ComplexFloat 3.141500e+00 + 4.200000e+01i

  constexpr _Complex int ArrayOfComplexInt[10] = {ComplexInt, ComplexInt, ComplexInt, ComplexInt};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} ArrayOfComplexInt '_Complex int const[10]' constexpr cinit
  // CHECK-NEXT:  |   |-value: Array size=10
  // CHECK-NEXT:  |   | |-elements: ComplexInt 42 + 24i, ComplexInt 42 + 24i, ComplexInt 42 + 24i, ComplexInt 42 + 24i
  // CHECK-NEXT:  |   | `-filler: 6 x ComplexInt 0 + 0i

  constexpr _Complex float ArrayOfComplexFloat[10] = {ComplexFloat, ComplexFloat, ComplexInt, ComplexInt};
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} ArrayOfComplexFloat '_Complex float const[10]' constexpr cinit
  // CHECK-NEXT:      |-value: Array size=10
  // CHECK-NEXT:      | |-elements: ComplexFloat 3.141500e+00 + 4.200000e+01i, ComplexFloat 3.141500e+00 + 4.200000e+01i, ComplexFloat 4.200000e+01 + 2.400000e+01i, ComplexFloat 4.200000e+01 + 2.400000e+01i
  // CHECK-NEXT:      | `-filler: 6 x ComplexFloat 0.000000e+00 + 0.000000e+00i
}
