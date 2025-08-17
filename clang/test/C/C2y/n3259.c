// RUN: %clang_cc1 -triple=x86_64 -std=c2y -Wall -pedantic -Wno-unused -Wpre-c2y-compat -verify=pre-c2y %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64 -std=c23 -Wall -pedantic -Wno-unused %s -verify -emit-llvm -o - | FileCheck %s

/* WG14 N3259: Yes
 * Support ++ and -- on complex values
 */

// CHECK-LABEL: define {{.*}} void @test()
void test() {
  // CHECK: %[[F:.+]] = alloca { float, float }
  // CHECK: store float 1
  // CHECK: store float 0
  _Complex float f = __builtin_complex(1.0f, 0.0f);

  // CHECK:      %[[F_REALP1:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 0
  // CHECK-NEXT: %[[F_REAL:.+]] = load float, ptr %[[F_REALP1]]
  // CHECK-NEXT: %[[F_IMAGP2:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 1
  // CHECK-NEXT: %[[F_IMAG:.+]] = load float, ptr %[[F_IMAGP2]]
  // CHECK-NEXT: %[[INC:.+]] = fadd float %[[F_REAL]], 1.000000e+00
  // CHECK-NEXT: %[[F_REALP3:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 0
  // CHECK-NEXT: %[[F_IMAGP4:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 1
  // CHECK-NEXT: store float %[[INC]], ptr %[[F_REALP3]]
  // CHECK-NEXT: store float %[[F_IMAG]], ptr %[[F_IMAGP4]]
  f++; /* expected-warning {{'++' on an object of complex type is a C2y extension}}
          pre-c2y-warning {{'++' on an object of complex type is incompatible with C standards before C2y}}
        */

  // CHECK:      %[[F_REALP5:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 0
  // CHECK-NEXT: %[[F_REAL6:.+]] = load float, ptr %[[F_REALP5]]
  // CHECK-NEXT: %[[F_IMAGP7:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 1
  // CHECK-NEXT: %[[F_IMAG8:.+]] = load float, ptr %[[F_IMAGP7]]
  // CHECK-NEXT: %[[INC9:.+]] = fadd float %[[F_REAL6]], 1.000000e+00
  // CHECK-NEXT: %[[F_REALP10:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 0
  // CHECK-NEXT: %[[F_IMAGP11:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 1
  // CHECK-NEXT: store float %[[INC9]], ptr %[[F_REALP10]]
  // CHECK-NEXT: store float %[[F_IMAG8]], ptr %[[F_IMAGP11]]
  ++f; /* expected-warning {{'++' on an object of complex type is a C2y extension}}
          pre-c2y-warning {{'++' on an object of complex type is incompatible with C standards before C2y}}
        */

  // CHECK:      %[[F_REALP12:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 0
  // CHECK-NEXT: %[[F_REAL13:.+]] = load float, ptr %[[F_REALP12]]
  // CHECK-NEXT: %[[F_IMAGP14:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 1
  // CHECK-NEXT: %[[F_IMAG15:.+]] = load float, ptr %[[F_IMAGP14]]
  // CHECK-NEXT: %[[DEC:.+]] = fadd float %[[F_REAL13]], -1.000000e+00
  // CHECK-NEXT: %[[F_REALP16:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 0
  // CHECK-NEXT: %[[F_IMAGP17:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 1
  // CHECK-NEXT: store float %[[DEC]], ptr %[[F_REALP16]]
  // CHECK-NEXT: store float %[[F_IMAG15]], ptr %[[F_IMAGP17]]
  f--; /* expected-warning {{'--' on an object of complex type is a C2y extension}}
          pre-c2y-warning {{'--' on an object of complex type is incompatible with C standards before C2y}}
        */

  // CHECK:      %[[F_REALP18:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 0
  // CHECK-NEXT: %[[F_REAL19:.+]] = load float, ptr %[[F_REALP18]]
  // CHECK-NEXT: %[[F_IMAGP20:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 1
  // CHECK-NEXT: %[[F_IMAG21:.+]] = load float, ptr %[[F_IMAGP20]]
  // CHECK-NEXT: %[[DEC22:.+]] = fadd float %[[F_REAL19]], -1.000000e+00
  // CHECK-NEXT: %[[F_REALP23:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 0
  // CHECK-NEXT: %[[F_IMAGP24:.+]] = getelementptr inbounds nuw { float, float }, ptr %[[F]], i32 0, i32 1
  // CHECK-NEXT: store float %[[DEC22]], ptr %[[F_REALP23]]
  // CHECK-NEXT: store float %[[F_IMAG21]], ptr %[[F_IMAGP24]]
  --f; /* expected-warning {{'--' on an object of complex type is a C2y extension}}
          pre-c2y-warning {{'--' on an object of complex type is incompatible with C standards before C2y}}
        */
}
