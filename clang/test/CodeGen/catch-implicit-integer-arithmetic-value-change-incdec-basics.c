// RUN: %clang_cc1 -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK

// CHECK-DAG: @[[INT:.*]] = {{.*}} c"'int'\00" }
// CHECK-DAG: @[[UNSIGNED_SHORT:.*]] = {{.*}} c"'unsigned short'\00" }
// CHECK-DAG: @[[LINE_100:.*]] = {{.*}}, i32 100, i32 4 }, {{.*}}* @[[INT]], {{.*}}* @[[UNSIGNED_SHORT]], i8 2 }
// CHECK-DAG: @[[LINE_200:.*]] = {{.*}}, i32 200, i32 4 }, {{.*}}* @[[INT]], {{.*}}* @[[UNSIGNED_SHORT]], i8 2 }
// CHECK-DAG: @[[LINE_300:.*]] = {{.*}}, i32 300, i32 3 }, {{.*}}* @[[INT]], {{.*}}* @[[UNSIGNED_SHORT]], i8 2 }
// CHECK-DAG: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 3 }, {{.*}}* @[[INT]], {{.*}}* @[[UNSIGNED_SHORT]], i8 2 }
// CHECK-DAG: @[[SHORT:.*]] = {{.*}} c"'short'\00" }
// CHECK-DAG: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 4 }, {{.*}}* @[[INT]], {{.*}}* @[[SHORT]], i8 2 }
// CHECK-DAG: @[[LINE_600:.*]] = {{.*}}, i32 600, i32 4 }, {{.*}}* @[[INT]], {{.*}}* @[[SHORT]], i8 2 }
// CHECK-DAG: @[[LINE_700:.*]] = {{.*}}, i32 700, i32 3 }, {{.*}}* @[[INT]], {{.*}}* @[[SHORT]], i8 2 }
// CHECK-DAG: @[[LINE_800:.*]] = {{.*}}, i32 800, i32 3 }, {{.*}}* @[[INT]], {{.*}}* @[[SHORT]], i8 2 }
// CHECK-DAG: @[[UNSIGNED_CHAR:.*]] = {{.*}} c"'unsigned char'\00" }
// CHECK-DAG: @[[LINE_900:.*]] = {{.*}}, i32 900, i32 4 }, {{.*}}* @[[INT]], {{.*}}* @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-DAG: @[[LINE_1000:.*]] = {{.*}}, i32 1000, i32 4 }, {{.*}}* @[[INT]], {{.*}}* @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-DAG: @[[LINE_1100:.*]] = {{.*}}, i32 1100, i32 3 }, {{.*}}* @[[INT]], {{.*}}* @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-DAG: @[[LINE_1200:.*]] = {{.*}}, i32 1200, i32 3 }, {{.*}}* @[[INT]], {{.*}}* @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-DAG: @[[SIGNED_CHAR:.*]] = {{.*}} c"'signed char'\00" }
// CHECK-DAG: @[[LINE_1300:.*]] = {{.*}}, i32 1300, i32 4 }, {{.*}}* @[[INT]], {{.*}}* @[[SIGNED_CHAR]], i8 2 }
// CHECK-DAG: @[[LINE_1400:.*]] = {{.*}}, i32 1400, i32 4 }, {{.*}}* @[[INT]], {{.*}}* @[[SIGNED_CHAR]], i8 2 }
// CHECK-DAG: @[[LINE_1500:.*]] = {{.*}}, i32 1500, i32 3 }, {{.*}}* @[[INT]], {{.*}}* @[[SIGNED_CHAR]], i8 2 }
// CHECK-DAG: @[[LINE_1600:.*]] = {{.*}}, i32 1600, i32 3 }, {{.*}}* @[[INT]], {{.*}}* @[[SIGNED_CHAR]], i8 2 }

// CHECK-LABEL: @t0(
unsigned short t0(unsigned short x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100]] to i8*)
#line 100
  x++;
  return x;
}
// CHECK-LABEL: @t1(
unsigned short t1(unsigned short x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_200]] to i8*)
#line 200
  x--;
  return x;
}
// CHECK-LABEL: @t2(
unsigned short t2(unsigned short x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_300]] to i8*)
#line 300
  ++x;
  return x;
}
// CHECK-LABEL: @t3(
unsigned short t3(unsigned short x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_400]] to i8*)
#line 400
  --x;
  return x;
}

// CHECK-LABEL: @t4(
signed short t4(signed short x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_500]] to i8*)
#line 500
  x++;
  return x;
}
// CHECK-LABEL: @t5(
signed short t5(signed short x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_600]] to i8*)
#line 600
  x--;
  return x;
}
// CHECK-LABEL: @t6(
signed short t6(signed short x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_700]] to i8*)
#line 700
  ++x;
  return x;
}
// CHECK-LABEL: @t7(
signed short t7(signed short x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_800]] to i8*)
#line 800
  --x;
  return x;
}

// CHECK-LABEL: @t8(
unsigned char t8(unsigned char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_900]] to i8*)
#line 900
  x++;
  return x;
}
// CHECK-LABEL: @t9(
unsigned char t9(unsigned char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1000]] to i8*)
#line 1000
  x--;
  return x;
}
// CHECK-LABEL: @t10(
unsigned char t10(unsigned char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1100]] to i8*)
#line 1100
  ++x;
  return x;
}
// CHECK-LABEL: @t11(
unsigned char t11(unsigned char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1200]] to i8*)
#line 1200
  --x;
  return x;
}

// CHECK-LABEL: @t12(
signed char t12(signed char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1300]] to i8*)
#line 1300
  x++;
  return x;
}
// CHECK-LABEL: @t13(
signed char t13(signed char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1400]] to i8*)
#line 1400
  x--;
  return x;
}
// CHECK-LABEL: @t14(
signed char t14(signed char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1500]] to i8*)
#line 1500
  ++x;
  return x;
}
// CHECK-LABEL: @t15(
signed char t15(signed char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1600]] to i8*)
#line 1600
  --x;
  return x;
}
