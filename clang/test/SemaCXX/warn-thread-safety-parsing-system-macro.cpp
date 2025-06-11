// RUN: %clang_cc1 -fsyntax-only -I%S/Inputs -Wthread-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -I%S/Inputs -Wthread-safety -std=c++98 %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -I%S/Inputs -Wthread-safety -std=c++11 %s -D CPP11 2>&1 | FileCheck %s

#include <warn-thread-safety-parsing.h>

#define LOCKABLE            __attribute__ ((lockable))
#define SCOPED_LOCKABLE     __attribute__ ((scoped_lockable))
#define GUARDED_BY(x)       __attribute__ ((guarded_by(x)))
#define GUARDED_VAR         __attribute__ ((guarded_var))
#define PT_GUARDED_BY(x)    __attribute__ ((pt_guarded_by(x)))
#define PT_GUARDED_VAR      __attribute__ ((pt_guarded_var))
#define ACQUIRED_AFTER(...) __attribute__ ((acquired_after(__VA_ARGS__)))
#define ACQUIRED_BEFORE(...) __attribute__ ((acquired_before(__VA_ARGS__)))
#define EXCLUSIVE_LOCK_FUNCTION(...)    __attribute__ ((exclusive_lock_function(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)       __attribute__ ((shared_lock_function(__VA_ARGS__)))
#define ASSERT_EXCLUSIVE_LOCK(...)      __attribute__ ((assert_exclusive_lock(__VA_ARGS__)))
#define ASSERT_SHARED_LOCK(...)         __attribute__ ((assert_shared_lock(__VA_ARGS__)))
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) __attribute__ ((exclusive_trylock_function(__VA_ARGS__)))
#define SHARED_TRYLOCK_FUNCTION(...)    __attribute__ ((shared_trylock_function(__VA_ARGS__)))
#define UNLOCK_FUNCTION(...)            __attribute__ ((unlock_function(__VA_ARGS__)))
#define LOCK_RETURNED(x)    __attribute__ ((lock_returned(x)))
#define LOCKS_EXCLUDED(...) __attribute__ ((locks_excluded(__VA_ARGS__)))
#define EXCLUSIVE_LOCKS_REQUIRED(...) \
  __attribute__ ((exclusive_locks_required(__VA_ARGS__)))
#define SHARED_LOCKS_REQUIRED(...) \
  __attribute__ ((shared_locks_required(__VA_ARGS__)))
#define NO_THREAD_SAFETY_ANALYSIS  __attribute__ ((no_thread_safety_analysis))


class LOCKABLE Mutex {
  public:
  void Lock()          EXCLUSIVE_LOCK_FUNCTION();
  void ReaderLock()    SHARED_LOCK_FUNCTION();
  void Unlock()        UNLOCK_FUNCTION();

  bool TryLock()       EXCLUSIVE_TRYLOCK_FUNCTION(true);
  bool ReaderTryLock() SHARED_TRYLOCK_FUNCTION(true);

  void AssertHeld()       ASSERT_EXCLUSIVE_LOCK();
  void AssertReaderHeld() ASSERT_SHARED_LOCK();
};

Mutex mu1;

void gb_function_sys_macro() _SYS_GUARDED_BY(mu1);
// CHECK: :[[@LINE-1]]:{{.*}} warning: '_SYS_GUARDED_BY' attribute only applies to
// CHECK: {{.*}}warn-thread-safety-parsing.h:3:{{.*}}: note: expanded from macro '_SYS_GUARDED_BY'
// CHECK: 3 | #define _SYS_GUARDED_BY(x)       __attribute__ ((guarded_by(x)))
// CHECK:   |                                                  ^

void gb_function_params_sys_macro(int gv_lvar _SYS_GUARDED_BY(mu1));
// CHECK: :[[@LINE-1]]:{{.*}} warning: '_SYS_GUARDED_BY' attribute only applies to
// CHECK: {{.*}}warn-thread-safety-parsing.h:3:{{.*}}: note: expanded from macro '_SYS_GUARDED_BY'
// CHECK: 3 | #define _SYS_GUARDED_BY(x)       __attribute__ ((guarded_by(x)))
// CHECK:   |                                                  ^

int gb_testfn_sys_macro(int y){
  int x _SYS_GUARDED_BY(mu1) = y;
// CHECK: :[[@LINE-1]]:{{.*}} warning: '_SYS_GUARDED_BY' attribute only applies to
// CHECK: {{.*}}warn-thread-safety-parsing.h:3:{{.*}}: note: expanded from macro '_SYS_GUARDED_BY'
// CHECK: 3 | #define _SYS_GUARDED_BY(x)       __attribute__ ((guarded_by(x)))
// CHECK:   |                                                  ^
  return x;
}
