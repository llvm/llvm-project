// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s

void empty();

void check_noproto_ptr() {
  void (*fun)(void) = empty;
}

// CHECK:  cir.func no_proto @check_noproto_ptr()
// CHECK:    [[ALLOC:%.*]] = cir.alloca !cir.ptr<!cir.func<!void ()>>, cir.ptr <!cir.ptr<!cir.func<!void ()>>>, ["fun", init] {alignment = 8 : i64}
// CHECK:    [[GGO:%.*]] = cir.get_global @empty : cir.ptr <!cir.func<!void ()>>
// CHECK:    [[CAST:%.*]] = cir.cast(bitcast, [[GGO]] : !cir.ptr<!cir.func<!void ()>>), !cir.ptr<!cir.func<!void ()>>
// CHECK:    cir.store [[CAST]], [[ALLOC]] : !cir.ptr<!cir.func<!void ()>>, cir.ptr <!cir.ptr<!cir.func<!void ()>>>
// CHECK:    cir.return

void empty(void) {}

void buz() {
  void (*func)();
  (*func)();
}

// CHECK:  cir.func no_proto @buz()
// CHECK:    [[FNPTR_ALLOC:%.*]] = cir.alloca !cir.ptr<!cir.func<!void (...)>>, cir.ptr <!cir.ptr<!cir.func<!void (...)>>>, ["func"] {alignment = 8 : i64}
// CHECK:    [[FNPTR:%.*]] = cir.load deref [[FNPTR_ALLOC]] : cir.ptr <!cir.ptr<!cir.func<!void (...)>>>, !cir.ptr<!cir.func<!void (...)>>
// CHECK:    [[CAST:%.*]] = cir.cast(bitcast, %1 : !cir.ptr<!cir.func<!void (...)>>), !cir.ptr<!cir.func<!void ()>>
// CHECK:    cir.call [[CAST]]() : (!cir.ptr<!cir.func<!void ()>>) -> ()
// CHECK:    cir.return
