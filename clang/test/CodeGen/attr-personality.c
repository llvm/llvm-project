// RUN: %clang_cc1 -Oz -triple x86_64-unknown-linux-gnu -fexceptions -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -o - -emit-llvm %s | FileCheck %s -check-prefix CHECK-C23

typedef __UINT64_TYPE__ uint64_t;

void abort(void) __attribute__((__noreturn__));

typedef enum _Unwind_Reason_Code _Unwind_Reason_Code;
typedef enum _Unwind_Action _Unwind_Action;

struct _Unwind_Exception;
struct _Unwind_Context;

_Unwind_Reason_Code __swift_personality_v0(int version, _Unwind_Action action, uint64_t exception_class, struct _Unwind_Exception *exception, struct _Unwind_Context *context);

void function(void);

void __attribute__((__personality__(__swift_personality_v0))) function_with_custom_personality(void) {
}

static void cleanup_int(int *p) {
  *p = 0;
}

void function_without_custom_personality(void) {
  int variable __attribute__((__cleanup__(cleanup_int))) = 1;
  function();
}

// CHECK-DAG: define dso_local void @function_with_custom_personality(){{.*}}personality ptr @__swift_personality_v0
// CHECK-DAG: define dso_local void @function_without_custom_personality(){{.*}}personality ptr @__gcc_personality_v0

int __attribute__((__noinline__)) __invalid_personality_v0(void) { return 0; }
int __invalid_personality_caller_0(void) { return __invalid_personality_v0(); }
void __attribute__((__personality__(__invalid_personality_v0))) with_invalid_personality_0(void) {
}

int __attribute__((__noinline__)) __invalid_personality_v1(void) { return 0; }
void __attribute__((__personality__(__invalid_personality_v1))) with_invalid_personality_1(void) {
}
int __invalid_personality_caller_1(void) { return __invalid_personality_v1(); }

int __attribute__((__noinline__)) __invalid_personality_v2(void);
void __attribute__((__personality__(__invalid_personality_v2))) with_invalid_personality_2(void) {
}
int __attribute__((__noinline__)) __invalid_personality_v2(void) { return 0; }
int __invalid_personality_caller_2(void) { return __invalid_personality_v2(); }

// CHECK-C23:      define {{.*}}i32 @__invalid_personality_v0() {{.*}}{
// CHECK-C23-NEXT: entry:
// CHECK-C23-NEXT:   ret i32 0
// CHECK-C23-NEXT: }

// CHECK-C23:      define {{.*}}i32 @__invalid_personality_caller_0() {{.*}}{
// CHECK-C23-NEXT: entry:
// CHECK-C23-NEXT:   %call = call i32 @__invalid_personality_v0()
// CHECK-C23-NEXT:   ret i32 %call
// CHECK-C23-NEXT: }

// CHECK-C23:      define {{.*}}i32 @__invalid_personality_v1() {{.*}}{
// CHECK-C23-NEXT: entry:
// CHECK-C23-NEXT:   ret i32 0
// CHECK-C23-NEXT: }

// CHECK-C23:      define {{.*}}i32 @__invalid_personality_caller_1() {{.*}}{
// CHECK-C23-NEXT: entry:
// CHECK-C23-NEXT:   %call = call i32 @__invalid_personality_v1()
// CHECK-C23-NEXT:   ret i32 %call
// CHECK-C23-NEXT: }

// CHECK-C23:      define {{.*}}i32 @__invalid_personality_v2() {{.*}}{
// CHECK-C23-NEXT: entry:
// CHECK-C23-NEXT:   ret i32 0
// CHECK-C23-NEXT: }

// CHECK-C23:      define {{.*}}i32 @__invalid_personality_caller_2() {{.*}}{
// CHECK-C23-NEXT: entry:
// CHECK-C23-NEXT:   %call = call i32 @__invalid_personality_v2()
// CHECK-C23-NEXT:   ret i32 %call
// CHECK-C23-NEXT: }
