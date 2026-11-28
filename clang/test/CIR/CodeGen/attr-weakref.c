// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Globals. A weakref's target is extern_weak unless the target is also declared
// or defined, without being weak, elsewhere in the translation unit.
// CIR-DAG: cir.global "private" extern_weak @var_target : !s32i
// LLVM-DAG: @var_target = extern_weak global i32
// CIR-DAG: cir.global external @var_init = #cir.global_view<@var_target> : !cir.ptr<!s32i> align(8)
// LLVM-DAG: @var_init = global ptr @var_target, align 8
// CIR-DAG: cir.global external @fn_init = #cir.global_view<@fn_target> : !cir.ptr<!cir.func<(!s32i)>> align(8)
// LLVM-DAG: @fn_init = global ptr @fn_target, align 8
// CIR-DAG: cir.global external @later_def = #cir.int<5> : !s32i align(4)
// LLVM-DAG: @later_def = global i32 5, align 4
// CIR-DAG: cir.global "private" external @decl_target : !s32i
// LLVM-DAG: @decl_target = external global i32
// CIR-DAG: cir.global "private" extern_weak @weak_target : !s32i
// LLVM-DAG: @weak_target = extern_weak global i32

// A weakref declaration that is never referenced should produce no output for
// either the alias or its target. Only the unrelated definition below should
// be emitted.
void weakref_target(void);
static void weakref_alias(void) __attribute__((weakref("weakref_target")));

void unrelated(void) {}

// The weakref alias produces no output by itself.
// CIR-NOT: @weakref_alias
// CIR-NOT: @weakref_target
// LLVM-NOT: @weakref_alias
// LLVM-NOT: @weakref_target

// CIR: cir.func{{.*}}@unrelated
// LLVM: define{{.*}}@unrelated

// A reference to a weakref is a reference to its target, which is declared
// extern_weak.
static int var_ref __attribute__((weakref("var_target")));
static void fn_ref(int) __attribute__((weakref("fn_target")));

int read_var(void) { return var_ref; }
// CIR: cir.func {{.*}}@read_var()
// CIR:   %[[VAR:.*]] = cir.get_global @var_target : !cir.ptr<!s32i>
// CIR:   cir.load align(4) %[[VAR]]
// LLVM: define {{.*}}@read_var()
// LLVM:   load i32, ptr @var_target, align 4

int *addr_var(void) { return &var_ref; }
// CIR: cir.func {{.*}}@addr_var()
// CIR:   cir.get_global @var_target : !cir.ptr<!s32i>
// LLVM: define {{.*}}@addr_var()
// LLVM:   ptr @var_target

void call_fn(void) { fn_ref(1); }
// CIR: cir.func {{.*}}@call_fn()
// CIR:   cir.call @fn_target(%{{.*}}) : (!s32i {llvm.noundef}) -> ()
// CIR: cir.func extern_weak private @fn_target(!s32i {llvm.noundef})
// LLVM: define {{.*}}@call_fn()
// LLVM:   call void @fn_target(i32 noundef 1)
// LLVM: declare extern_weak void @fn_target(i32 noundef)

int fn_not_null(void) { return fn_ref != 0; }
// CIR: cir.func {{.*}}@fn_not_null()
// CIR:   %[[FN:.*]] = cir.get_global @fn_target : !cir.ptr<!cir.func<(!s32i)>>
// CIR:   cir.cmp ne %[[FN]], %{{.*}} : !cir.ptr<!cir.func<(!s32i)>>
// LLVM: define {{.*}}@fn_not_null()
// LLVM:   icmp ne ptr @fn_target, null

void call_fn_through_pointer(void) {
  void (*p)(int) = fn_ref;
  p(2);
}
// CIR: cir.func {{.*}}@call_fn_through_pointer()
// CIR:   cir.get_global @fn_target
// LLVM: define {{.*}}@call_fn_through_pointer()
// LLVM:   store ptr @fn_target, ptr %{{.*}}
// LLVM:   call void %{{.*}}(i32 noundef 2)

// A weakref in a constant initializer is the address of its target.
int *var_init = &var_ref;
void (*fn_init)(int) = fn_ref;

// The target of a weakref is no longer weak once it is defined, or declared
// without 'weak', anywhere in the translation unit, even after the weakref is
// used.
static int later_ref __attribute__((weakref("later_def")));
int *addr_later(void) { return &later_ref; }
int later_def = 5;
// CIR: cir.func {{.*}}@addr_later()
// CIR:   cir.get_global @later_def : !cir.ptr<!s32i>
// LLVM: define {{.*}}@addr_later()
// LLVM:   ptr @later_def

static void later_fn_ref(void) __attribute__((weakref("later_fn")));
void call_later_fn(void) { later_fn_ref(); }
void later_fn(void) {}
// CIR: cir.func {{.*}}@call_later_fn()
// CIR:   cir.call @later_fn() : () -> ()
// CIR: cir.func no_inline dso_local @later_fn()
// LLVM: define {{.*}}@call_later_fn()
// LLVM:   call void @later_fn()
// LLVM: define dso_local void @later_fn()

static int decl_ref __attribute__((weakref("decl_target")));
int read_decl(void) { return decl_ref; }
extern int decl_target;
int read_decl_target(void) { return decl_target; }
// CIR: cir.func {{.*}}@read_decl()
// CIR:   cir.get_global @decl_target : !cir.ptr<!s32i>
// CIR: cir.func {{.*}}@read_decl_target()
// CIR:   cir.get_global @decl_target : !cir.ptr<!s32i>
// LLVM: define {{.*}}@read_decl()
// LLVM:   load i32, ptr @decl_target
// LLVM: define {{.*}}@read_decl_target()
// LLVM:   load i32, ptr @decl_target

// If the target is itself declared weak, it stays weak.
static int weak_ref __attribute__((weakref("weak_target")));
int read_weak(void) { return weak_ref; }
extern int weak_target __attribute__((weak));
int read_weak_target(void) { return weak_target; }
// CIR: cir.func {{.*}}@read_weak()
// CIR:   cir.get_global @weak_target : !cir.ptr<!s32i>
// CIR: cir.func {{.*}}@read_weak_target()
// CIR:   cir.get_global @weak_target : !cir.ptr<!s32i>
// LLVM: define {{.*}}@read_weak()
// LLVM:   load i32, ptr @weak_target
// LLVM: define {{.*}}@read_weak_target()
// LLVM:   load i32, ptr @weak_target
