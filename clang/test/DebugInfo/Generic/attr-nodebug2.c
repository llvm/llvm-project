// RUN: %clang_cc1 -x c -debug-info-kind=limited -debugger-tuning=gdb -dwarf-version=4 -O -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c++ -debug-info-kind=limited -debugger-tuning=gdb -dwarf-version=4 -O -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#ifdef __cplusplus
extern "C" {
#endif

void t1(void);

void use(void) { t1(); }

__attribute__((nodebug)) void t1(void) {
  int a = 10;
  a++;
}

// A deferred caller is emitted after the nodebug definition, so its call site
// declaration would attach to a function that already has a body.
void t2(void);

__attribute__((nodebug)) void t2(void) {
  int a = 10;
  a++;
}

static inline void deferred_caller(void) { t2(); }

void use2(void) { deferred_caller(); }

#ifdef __cplusplus
}
#endif

// CHECK-LABEL: define{{.*}} void @use()
// CHECK-SAME:  !dbg
// CHECK-SAME:  {
// CHECK:       !dbg
// CHECK:       }

// PR50767 Function __attribute__((nodebug)) inconsistency causes crash
// illegal (non-distinct) !dbg metadata was being added to _Z2t1v definition

// CHECK-LABEL: define{{.*}} void @t1()
// CHECK-NOT:   !dbg
// CHECK-SAME:  {
// CHECK-NOT:   !dbg
// CHECK:       }

// CHECK-LABEL: define{{.*}} void @t2()
// CHECK-NOT:   !dbg
// CHECK-SAME:  {
// CHECK-NOT:   !dbg
// CHECK:       }
