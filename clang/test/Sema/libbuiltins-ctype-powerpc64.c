// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm < %s | FileCheck %s

int isalnum(int);
int isalpha(int);
int isblank(int);
int iscntrl(int);
int isdigit(int);
int isgraph(int);
int islower(int);
int isprint(int);
int ispunct(int);
int isspace(int);
int isupper(int);
int isxdigit(int);
int tolower(int);
int toupper(int);

void test(int x) {
  // CHECK: call signext i32 @isalnum(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)isalnum(x);
  // CHECK: call signext i32 @isalpha(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)isalpha(x);
  // CHECK: call signext i32 @isblank(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)isblank(x);
  // CHECK: call signext i32 @iscntrl(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)iscntrl(x);
  // CHECK: call signext i32 @isdigit(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)isdigit(x);
  // CHECK: call signext i32 @isgraph(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)isgraph(x);
  // CHECK: call signext i32 @islower(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)islower(x);
  // CHECK: call signext i32 @isprint(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)isprint(x);
  // CHECK: call signext i32 @ispunct(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)ispunct(x);
  // CHECK: call signext i32 @isspace(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)isspace(x);
  // CHECK: call signext i32 @isupper(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)isupper(x);
  // CHECK: call signext i32 @isxdigit(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)isxdigit(x);
  // CHECK: call signext i32 @tolower(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)tolower(x);
  // CHECK: call signext i32 @toupper(i32 signext {{%[0-9]+}}) [[NUW_RO_CALL:#[0-9]+]]
  (void)toupper(x);
}

// CHECK: declare signext i32 @isalnum(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @isalpha(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @isblank(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @iscntrl(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @isdigit(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @isgraph(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @islower(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @isprint(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @ispunct(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @isspace(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @isupper(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @isxdigit(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @tolower(i32 signext) [[NUW_RO:#[0-9]+]]
// CHECK: declare signext i32 @toupper(i32 signext) [[NUW_RO:#[0-9]+]]

// CHECK: attributes [[NUW_RO]] = { nounwind readonly{{.*}} }
// CHECK: attributes [[NUW_RO_CALL]] = { nounwind readonly willreturn }
