// RUN: %clang_cc1 -Oz -triple x86_64-unknown-linux-gnu -fexceptions -o - -emit-llvm %s | FileCheck %s

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
