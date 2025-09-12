// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s

void empty();

void check_noproto_ptr() {
  void (*fun)(void) = empty;
}

// CHECK:  cir.func no_proto dso_local @check_noproto_ptr()
// CHECK:    [[ALLOC:%.*]] = cir.alloca !cir.ptr<!cir.func<()>>, !cir.ptr<!cir.ptr<!cir.func<()>>>, ["fun", init] {alignment = 8 : i64}
// CHECK:    [[GGO:%.*]] = cir.get_global @empty : !cir.ptr<!cir.func<()>>
// CHECK:    cir.store{{.*}} [[GGO]], [[ALLOC]] : !cir.ptr<!cir.func<()>>, !cir.ptr<!cir.ptr<!cir.func<()>>>
// CHECK:    cir.return

void empty(void) {}

void buz() {
  void (*func)();
  (*func)();
}

// CHECK:  cir.func no_proto dso_local @buz()
// CHECK:    [[FNPTR_ALLOC:%.*]] = cir.alloca !cir.ptr<!cir.func<(...)>>, !cir.ptr<!cir.ptr<!cir.func<(...)>>>, ["func"] {alignment = 8 : i64}
// CHECK:    [[FNPTR:%.*]] = cir.load deref{{.*}} [[FNPTR_ALLOC]] : !cir.ptr<!cir.ptr<!cir.func<(...)>>>, !cir.ptr<!cir.func<(...)>>
// CHECK:    [[CAST:%.*]] = cir.cast(bitcast, %1 : !cir.ptr<!cir.func<(...)>>), !cir.ptr<!cir.func<()>>
// CHECK:    cir.call [[CAST]]() : (!cir.ptr<!cir.func<()>>) -> ()
// CHECK:    cir.return
