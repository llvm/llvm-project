// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffast-math -ffp-contract=fast -emit-llvm -o - %s | FileCheck %s

float test_default(float a, float b, float c) {
  float tmp = a;
  tmp += b;
  tmp += c;
  return tmp;
}

// CHECK: define float @_Z12test_defaultfff(float %a, float %b, float %c) [[FAST_ATTRS:#[0-9]+]]
// CHECK: fadd fast float {{%.+}}, {{%.+}}
// CHECK: fadd fast float {{%.+}}, {{%.+}}

float test_precise_on_pragma(float a, float b, float c) {
  float tmp = a;
  {
    #pragma float_control(precise, on)
    tmp += b;
  }
  tmp += c;
  return tmp;
}

// CHECK: define float @_Z22test_precise_on_pragmafff(float %a, float %b, float %c) [[PRECISE_ATTRS:#[0-9]+]]
// CHECK: fadd float {{%.+}}, {{%.+}}
// CHECK: fadd fast float {{%.+}}, {{%.+}}

float test_reassociate_off_pragma(float a, float b, float c) {
  float tmp = a;
  {
    #pragma clang fp reassociate(off)
    tmp += b;
  }
  tmp += c;
  return tmp;
}

// CHECK: define float @_Z27test_reassociate_off_pragmafff(float %a, float %b, float %c) [[NOREASSOC_ATTRS:#[0-9]+]]
// CHECK: fadd nnan ninf nsz arcp contract afn float {{%.+}}, {{%.+}}
// CHECK: fadd fast float {{%.+}}, {{%.+}}

// CHECK: attributes [[FAST_ATTRS]] = { {{.*}}"no-infs-fp-math"="true" {{.*}}"no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" {{.*}}"unsafe-fp-math"="true"{{.*}} }
// CHECK: attributes [[PRECISE_ATTRS]] = { {{.*}}"no-infs-fp-math"="false" {{.*}}"no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" {{.*}}"unsafe-fp-math"="false"{{.*}} }
// CHECK: attributes [[NOREASSOC_ATTRS]] = { {{.*}}"no-infs-fp-math"="true" {{.*}}"no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" {{.*}}"unsafe-fp-math"="false"{{.*}} }
