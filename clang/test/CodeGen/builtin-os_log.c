// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -emit-llvm -triple x86_64-darwin-apple -o - %s | FileCheck %s --check-prefixes=CHECK

 
// Removed from builtins.c as the behavior of __builtin_os_log differs between
// platforms, so we only test on X86 however having this embedded in builtins.c
// makes testing more obtuse for non-X86 dependent behaviours.

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log
// CHECK: (ptr noundef %[[BUF:.*]], i32 noundef %[[I:.*]], ptr noundef %[[DATA:.*]])
void test_builtin_os_log(void *buf, int i, const char *data) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[I_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[DATA_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store i32 %[[I]], ptr %[[I_ADDR]], align 4
  // CHECK: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8

  // CHECK: store volatile i32 34, ptr %[[LEN]]
  len = __builtin_os_log_format_buffer_size("%d %{public}s %{private}.16P", i, data, data);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]]
  // CHECK: %[[V2:.*]] = load i32, ptr %[[I_ADDR]]
  // CHECK: %[[V3:.*]] = load ptr, ptr %[[DATA_ADDR]]
  // CHECK: %[[V4:.*]] = ptrtoint ptr %[[V3]] to i64
  // CHECK: %[[V5:.*]] = load ptr, ptr %[[DATA_ADDR]]
  // CHECK: %[[V6:.*]] = ptrtoint ptr %[[V5]] to i64
  // CHECK: call void @__os_log_helper_1_3_4_4_0_8_34_4_17_8_49(ptr noundef %[[V1]], i32 noundef %[[V2]], i64 noundef %[[V4]], i32 noundef 16, i64 noundef %[[V6]])
  __builtin_os_log_format(buf, "%d %{public}s %{private}.16P", i, data, data);

  // privacy annotations aren't recognized when they are preceded or followed
  // by non-whitespace characters.

  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{xyz public}s", data);

  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ public xyz}s", data);

  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ public1}s", data);

  // Privacy annotations do not have to be in the first comma-delimited string.

  // CHECK: call void @__os_log_helper_1_2_1_8_34(
  __builtin_os_log_format(buf, "%{ xyz, public }s", "abc");

  // CHECK: call void @__os_log_helper_1_3_1_8_33(
  __builtin_os_log_format(buf, "%{ xyz, private }s", "abc");

  // CHECK: call void @__os_log_helper_1_3_1_8_37(
  __builtin_os_log_format(buf, "%{ xyz, sensitive }s", "abc");

  // The strictest privacy annotation in the string wins.

  // CHECK: call void @__os_log_helper_1_3_1_8_33(
  __builtin_os_log_format(buf, "%{ private, public, private, public}s", "abc");

  // CHECK: call void @__os_log_helper_1_3_1_8_37(
  __builtin_os_log_format(buf, "%{ private, sensitive, private, public}s",
                          "abc");

  // CHECK: store volatile i32 22, ptr %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("%{mask.xyz}s", "abc");

  // CHECK: call void @__os_log_helper_1_2_2_8_112_8_34(ptr noundef {{.*}}, i64 noundef 8026488
  __builtin_os_log_format(buf, "%{mask.xyz, public}s", "abc");

  // CHECK: call void @__os_log_helper_1_3_2_8_112_4_1(ptr noundef {{.*}}, i64 noundef 8026488
  __builtin_os_log_format(buf, "%{ mask.xyz, private }d", 11);

  // Mask type is silently ignored.
  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ mask. xyz }s", "abc");

  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ mask.xy z }s", "abc");
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_3_4_4_0_8_34_4_17_8_49
// CHECK: (ptr noundef %[[BUFFER:.*]], i32 noundef %[[ARG0:.*]], i64 noundef %[[ARG1:.*]], i32 noundef %[[ARG2:.*]], i64 noundef %[[ARG3:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG1_ADDR:.*]] = alloca i64, align 8
// CHECK: %[[ARG2_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG3_ADDR:.*]] = alloca i64, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i32 %[[ARG0]], ptr %[[ARG0_ADDR]], align 4
// CHECK: store i64 %[[ARG1]], ptr %[[ARG1_ADDR]], align 8
// CHECK: store i32 %[[ARG2]], ptr %[[ARG2_ADDR]], align 4
// CHECK: store i64 %[[ARG3]], ptr %[[ARG3_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 3, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 4, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 0, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 4, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i32, ptr %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[V0]], ptr %[[ARGDATA]], align 1
// CHECK: %[[ARGDESCRIPTOR1:.*]] = getelementptr i8, ptr %[[BUF]], i64 8
// CHECK: store i8 34, ptr %[[ARGDESCRIPTOR1]], align 1
// CHECK: %[[ARGSIZE2:.*]] = getelementptr i8, ptr %[[BUF]], i64 9
// CHECK: store i8 8, ptr %[[ARGSIZE2]], align 1
// CHECK: %[[ARGDATA3:.*]] = getelementptr i8, ptr %[[BUF]], i64 10
// CHECK: %[[V1:.*]] = load i64, ptr %[[ARG1_ADDR]], align 8
// CHECK: store i64 %[[V1]], ptr %[[ARGDATA3]], align 1
// CHECK: %[[ARGDESCRIPTOR5:.*]] = getelementptr i8, ptr %[[BUF]], i64 18
// CHECK: store i8 17, ptr %[[ARGDESCRIPTOR5]], align 1
// CHECK: %[[ARGSIZE6:.*]] = getelementptr i8, ptr %[[BUF]], i64 19
// CHECK: store i8 4, ptr %[[ARGSIZE6]], align 1
// CHECK: %[[ARGDATA7:.*]] = getelementptr i8, ptr %[[BUF]], i64 20
// CHECK: %[[V2:.*]] = load i32, ptr %[[ARG2_ADDR]], align 4
// CHECK: store i32 %[[V2]], ptr %[[ARGDATA7]], align 1
// CHECK: %[[ARGDESCRIPTOR9:.*]] = getelementptr i8, ptr %[[BUF]], i64 24
// CHECK: store i8 49, ptr %[[ARGDESCRIPTOR9]], align 1
// CHECK: %[[ARGSIZE10:.*]] = getelementptr i8, ptr %[[BUF]], i64 25
// CHECK: store i8 8, ptr %[[ARGSIZE10]], align 1
// CHECK: %[[ARGDATA11:.*]] = getelementptr i8, ptr %[[BUF]], i64 26
// CHECK: %[[V3:.*]] = load i64, ptr %[[ARG3_ADDR]], align 8
// CHECK: store i64 %[[V3]], ptr %[[ARGDATA11]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_wide
// CHECK: (ptr noundef %[[BUF:.*]], ptr noundef %[[DATA:.*]], ptr noundef %[[STR:.*]])
typedef int wchar_t;
void test_builtin_os_log_wide(void *buf, const char *data, wchar_t *str) {
  volatile int len;

  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[STR_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8
  // CHECK: store ptr %[[STR]], ptr %[[STR_ADDR]], align 8

  // CHECK: store volatile i32 12, ptr %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("%S", str);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load ptr, ptr %[[STR_ADDR]], align 8
  // CHECK: %[[V3:.*]] = ptrtoint ptr %[[V2]] to i64
  // CHECK: call void @__os_log_helper_1_2_1_8_80(ptr noundef %[[V1]], i64 noundef %[[V3]])

  __builtin_os_log_format(buf, "%S", str);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_1_8_80
// CHECK: (ptr noundef %[[BUFFER:.*]], i64 noundef %[[ARG0:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i64, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i64 %[[ARG0]], ptr %[[ARG0_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 2, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 1, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 80, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 8, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i64, ptr %[[ARG0_ADDR]], align 8
// CHECK: store i64 %[[V0]], ptr %[[ARGDATA]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_precision_width
// CHECK: (ptr noundef %[[BUF:.*]], ptr noundef %[[DATA:.*]], i32 noundef %[[PRECISION:.*]], i32 noundef %[[WIDTH:.*]])
void test_builtin_os_log_precision_width(void *buf, const char *data,
                                         int precision, int width) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[PRECISION_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[WIDTH_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8
  // CHECK: store i32 %[[PRECISION]], ptr %[[PRECISION_ADDR]], align 4
  // CHECK: store i32 %[[WIDTH]], ptr %[[WIDTH_ADDR]], align 4

  // CHECK: store volatile i32 24, ptr %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("Hello %*.*s World", precision, width, data);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load i32, ptr %[[PRECISION_ADDR]], align 4
  // CHECK: %[[V3:.*]] = load i32, ptr %[[WIDTH_ADDR]], align 4
  // CHECK: %[[V4:.*]] = load ptr, ptr %[[DATA_ADDR]], align 8
  // CHECK: %[[V5:.*]] = ptrtoint ptr %[[V4]] to i64
  // CHECK: call void @__os_log_helper_1_2_3_4_0_4_16_8_32(ptr noundef %[[V1]], i32 noundef %[[V2]], i32 noundef %[[V3]], i64 noundef %[[V5]])
  __builtin_os_log_format(buf, "Hello %*.*s World", precision, width, data);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_3_4_0_4_16_8_32
// CHECK: (ptr noundef %[[BUFFER:.*]], i32 noundef %[[ARG0:.*]], i32 noundef %[[ARG1:.*]], i64 noundef %[[ARG2:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG1_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG2_ADDR:.*]] = alloca i64, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i32 %[[ARG0]], ptr %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[ARG1]], ptr %[[ARG1_ADDR]], align 4
// CHECK: store i64 %[[ARG2]], ptr %[[ARG2_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 2, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 3, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 0, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 4, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i32, ptr %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[V0]], ptr %[[ARGDATA]], align 1
// CHECK: %[[ARGDESCRIPTOR1:.*]] = getelementptr i8, ptr %[[BUF]], i64 8
// CHECK: store i8 16, ptr %[[ARGDESCRIPTOR1]], align 1
// CHECK: %[[ARGSIZE2:.*]] = getelementptr i8, ptr %[[BUF]], i64 9
// CHECK: store i8 4, ptr %[[ARGSIZE2]], align 1
// CHECK: %[[ARGDATA3:.*]] = getelementptr i8, ptr %[[BUF]], i64 10
// CHECK: %[[V1:.*]] = load i32, ptr %[[ARG1_ADDR]], align 4
// CHECK: store i32 %[[V1]], ptr %[[ARGDATA3]], align 1
// CHECK: %[[ARGDESCRIPTOR5:.*]] = getelementptr i8, ptr %[[BUF]], i64 14
// CHECK: store i8 32, ptr %[[ARGDESCRIPTOR5]], align 1
// CHECK: %[[ARGSIZE6:.*]] = getelementptr i8, ptr %[[BUF]], i64 15
// CHECK: store i8 8, ptr %[[ARGSIZE6]], align 1
// CHECK: %[[ARGDATA7:.*]] = getelementptr i8, ptr %[[BUF]], i64 16
// CHECK: %[[V2:.*]] = load i64, ptr %[[ARG2_ADDR]], align 8
// CHECK: store i64 %[[V2]], ptr %[[ARGDATA7]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_invalid
// CHECK: (ptr noundef %[[BUF:.*]], i32 noundef %[[DATA:.*]])
void test_builtin_os_log_invalid(void *buf, int data) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store i32 %[[DATA]], ptr %[[DATA_ADDR]], align 4

  // CHECK: store volatile i32 8, ptr %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("invalid specifier %: %d even a trailing one%", data);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load i32, ptr %[[DATA_ADDR]], align 4
  // CHECK: call void @__os_log_helper_1_0_1_4_0(ptr noundef %[[V1]], i32 noundef %[[V2]])

  __builtin_os_log_format(buf, "invalid specifier %: %d even a trailing one%", data);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_0_1_4_0
// CHECK: (ptr noundef %[[BUFFER:.*]], i32 noundef %[[ARG0:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i32, align 4
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i32 %[[ARG0]], ptr %[[ARG0_ADDR]], align 4
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 0, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 1, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 0, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 4, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i32, ptr %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[V0]], ptr %[[ARGDATA]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_percent
// CHECK: (ptr noundef %[[BUF:.*]], ptr noundef %[[DATA1:.*]], ptr noundef %[[DATA2:.*]])
// Check that the %% which does not consume any argument is correctly handled
void test_builtin_os_log_percent(void *buf, const char *data1, const char *data2) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA1_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA2_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store ptr %[[DATA1]], ptr %[[DATA1_ADDR]], align 8
  // CHECK: store ptr %[[DATA2]], ptr %[[DATA2_ADDR]], align 8
  // CHECK: store volatile i32 22, ptr %[[LEN]], align 4

  len = __builtin_os_log_format_buffer_size("%s %% %s", data1, data2);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load ptr, ptr %[[DATA1_ADDR]], align 8
  // CHECK: %[[V3:.*]] = ptrtoint ptr %[[V2]] to i64
  // CHECK: %[[V4:.*]] = load ptr, ptr %[[DATA2_ADDR]], align 8
  // CHECK: %[[V5:.*]] = ptrtoint ptr %[[V4]] to i64
  // CHECK: call void @__os_log_helper_1_2_2_8_32_8_32(ptr noundef %[[V1]], i64 noundef %[[V3]], i64 noundef %[[V5]])

  __builtin_os_log_format(buf, "%s %% %s", data1, data2);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_2_8_32_8_32
// CHECK: (ptr noundef %[[BUFFER:.*]], i64 noundef %[[ARG0:.*]], i64 noundef %[[ARG1:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i64, align 8
// CHECK: %[[ARG1_ADDR:.*]] = alloca i64, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i64 %[[ARG0]], ptr %[[ARG0_ADDR]], align 8
// CHECK: store i64 %[[ARG1]], ptr %[[ARG1_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 2, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 2, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 32, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 8, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i64, ptr %[[ARG0_ADDR]], align 8
// CHECK: store i64 %[[V0]], ptr %[[ARGDATA]], align 1
// CHECK: %[[ARGDESCRIPTOR1:.*]] = getelementptr i8, ptr %[[BUF]], i64 12
// CHECK: store i8 32, ptr %[[ARGDESCRIPTOR1]], align 1
// CHECK: %[[ARGSIZE2:.*]] = getelementptr i8, ptr %[[BUF]], i64 13
// CHECK: store i8 8, ptr %[[ARGSIZE2]], align 1
// CHECK: %[[ARGDATA3:.*]] = getelementptr i8, ptr %[[BUF]], i64 14
// CHECK: %[[V1:.*]] = load i64, ptr %[[ARG1_ADDR]], align 8
// CHECK: store i64 %[[V1]], ptr %[[ARGDATA3]], align 1

// Check that the following two functions call the same helper function.

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_merge_helper0
// CHECK: call void @__os_log_helper_1_0_2_4_0_8_0(
void test_builtin_os_log_merge_helper0(void *buf, int i, double d) {
  __builtin_os_log_format(buf, "%d %f", i, d);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_0_2_4_0_8_0(

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_merge_helper1
// CHECK: call void @__os_log_helper_1_0_2_4_0_8_0(
void test_builtin_os_log_merge_helper1(void *buf, unsigned u, long long ll) {
  __builtin_os_log_format(buf, "%u %lld", u, ll);
}

// Check that this function doesn't write past the end of array 'buf'.

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_errno
void test_builtin_os_log_errno(void) {
  // CHECK-NOT: @stacksave
  // CHECK: %[[BUF:.*]] = alloca [4 x i8], align 1
  // CHECK: %[[DECAY:.*]] = getelementptr inbounds [4 x i8], ptr %[[BUF]], i64 0, i64 0
  // CHECK: call void @__os_log_helper_1_2_1_0_96(ptr noundef %[[DECAY]])
  // CHECK-NOT: @stackrestore

  char buf[__builtin_os_log_format_buffer_size("%m")];
  __builtin_os_log_format(buf, "%m");
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_1_0_96
// CHECK: (ptr noundef %[[BUFFER:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 2, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 1, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 96, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 0, ptr %[[ARGSIZE]], align 1
// CHECK-NEXT: ret void

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_long_double
// CHECK: (ptr noundef %[[BUF:.*]], x86_fp80 noundef %[[LD:.*]])
void test_builtin_os_log_long_double(void *buf, long double ld) {
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[LD_ADDR:.*]] = alloca x86_fp80, align 16
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store x86_fp80 %[[LD]], ptr %[[LD_ADDR]], align 16
  // CHECK: %[[V0:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V1:.*]] = load x86_fp80, ptr %[[LD_ADDR]], align 16
  // CHECK: %[[V2:.*]] = bitcast x86_fp80 %[[V1]] to i80
  // CHECK: %[[V3:.*]] = zext i80 %[[V2]] to i128
  // CHECK: call void @__os_log_helper_1_0_1_16_0(ptr noundef %[[V0]], i128 noundef %[[V3]])

  __builtin_os_log_format(buf, "%Lf", ld);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_0_1_16_0
// CHECK: (ptr noundef %[[BUFFER:.*]], i128 noundef %[[ARG0:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i128, align 16
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i128 %[[ARG0]], ptr %[[ARG0_ADDR]], align 16
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 0, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 1, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 0, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 16, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V3:.*]] = load i128, ptr %[[ARG0_ADDR]], align 16
// CHECK: store i128 %[[V3]], ptr %[[ARGDATA]], align 1
