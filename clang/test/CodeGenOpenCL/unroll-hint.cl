// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL2.0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL1.2 -o - %s | FileCheck %s

/*** for ***/
void for_count()
{
// CHECK-LABEL: for_count
    __attribute__((opencl_unroll_hint(8)))
    for( int i = 0; i < 1000; ++i);
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_COUNT:.*]]
}

void for_disable()
{
// CHECK-LABEL: for_disable
    __attribute__((opencl_unroll_hint(1)))
    for( int i = 0; i < 1000; ++i);
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_DISABLE:.*]]
}

void for_enable()
{
// CHECK-LABEL: for_enable
    __attribute__((opencl_unroll_hint))
    for( int i = 0; i < 1000; ++i);
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_ENABLE:.*]]
}

/*** while ***/
void while_count()
{
// CHECK-LABEL: while_count
    int i = 1000;
    __attribute__((opencl_unroll_hint(8)))
    while(i-->0);
// CHECK: br label %{{.*}}, !llvm.loop ![[WHILE_COUNT:.*]]
}

void while_disable()
{
// CHECK-LABEL: while_disable
    int i = 1000;
    __attribute__((opencl_unroll_hint(1)))
    while(i-->0);
// CHECK: br label %{{.*}}, !llvm.loop ![[WHILE_DISABLE:.*]]
}

void while_enable()
{
// CHECK-LABEL: while_enable
    int i = 1000;
    __attribute__((opencl_unroll_hint))
    while(i-->0);
// CHECK: br label %{{.*}}, !llvm.loop ![[WHILE_ENABLE:.*]]
}

/*** do ***/
void do_count()
{
// CHECK-LABEL: do_count
    int i = 1000;
    __attribute__((opencl_unroll_hint(8)))
    do {} while(i--> 0);
// CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop ![[DO_COUNT:.*]]
}

void do_disable()
{
// CHECK-LABEL: do_disable
    int i = 1000;
    __attribute__((opencl_unroll_hint(1)))
    do {} while(i--> 0);
// CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop ![[DO_DISABLE:.*]]
}

void do_enable()
{
// CHECK-LABEL: do_enable
    int i = 1000;
    __attribute__((opencl_unroll_hint))
    do {} while(i--> 0);
// CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop ![[DO_ENABLE:.*]]
}


// CHECK: ![[FOR_COUNT]]     =  distinct !{![[FOR_COUNT]],  ![[COUNT:.*]]}
// CHECK: ![[COUNT]]         =  !{!"llvm.loop.unroll.count", i32 8}
// CHECK: ![[FOR_DISABLE]]   =  distinct !{![[FOR_DISABLE]],  ![[DISABLE:.*]]}
// CHECK: ![[DISABLE]]       =  !{!"llvm.loop.unroll.disable"}
// CHECK: ![[FOR_ENABLE]]      =  distinct !{![[FOR_ENABLE]],  ![[ENABLE:.*]]}
// CHECK: ![[ENABLE]]          =  !{!"llvm.loop.unroll.enable"}
// CHECK: ![[WHILE_COUNT]]   =  distinct !{![[WHILE_COUNT]],    ![[COUNT]]}
// CHECK: ![[WHILE_DISABLE]] =  distinct !{![[WHILE_DISABLE]],  ![[DISABLE]]}
// CHECK: ![[WHILE_ENABLE]]    =  distinct !{![[WHILE_ENABLE]],     ![[ENABLE]]}
// CHECK: ![[DO_COUNT]]      =  distinct !{![[DO_COUNT]],       ![[COUNT]]}
// CHECK: ![[DO_DISABLE]]    =  distinct !{![[DO_DISABLE]],     ![[DISABLE]]}
// CHECK: ![[DO_ENABLE]]       =  distinct !{![[DO_ENABLE]],        ![[ENABLE]]}
