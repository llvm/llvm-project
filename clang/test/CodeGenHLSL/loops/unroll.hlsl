// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN: dxil-pc-shadermodel6.3-library -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s

/*** for ***/
void for_count()
{
// CHECK-LABEL: for_count
    [unroll(8)]
    for( int i = 0; i < 1000; ++i);
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_DISTINCT:.*]]
}

void for_disable()
{
// CHECK-LABEL: for_disable
    [loop]
    for( int i = 0; i < 1000; ++i);
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_DISABLE:.*]]
}

void for_enable()
{
// CHECK-LABEL: for_enable
    [unroll]
    for( int i = 0; i < 1000; ++i);
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_ENABLE:.*]]
}

void for_nested_one_unroll_enable()
{
// CHECK-LABEL: for_nested_one_unroll_enable
    int s = 0;
    [unroll]
    for( int i = 0; i < 1000; ++i) {
        for( int j = 0; j < 10; ++j)
            s += i + j;
    }
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_NESTED_ENABLE:.*]]
// CHECK-NOT: br label %{{.*}}, !llvm.loop ![[FOR_NESTED_1_ENABLE:.*]]
}

void for_nested_two_unroll_enable()
{
// CHECK-LABEL: for_nested_two_unroll_enable
    int s = 0;
    [unroll]
    for( int i = 0; i < 1000; ++i) {
        [unroll]
        for( int j = 0; j < 10; ++j)
            s += i + j;
    }
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_NESTED2_ENABLE:.*]]
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_NESTED2_1_ENABLE:.*]]
}


/*** while ***/
void while_count()
{
// CHECK-LABEL: while_count
    int i = 1000;
    [unroll(4)]
    while(i-->0);
// CHECK: br label %{{.*}}, !llvm.loop ![[WHILE_DISTINCT:.*]]
}

void while_disable()
{
// CHECK-LABEL: while_disable
    int i = 1000;
    [loop]
    while(i-->0);
// CHECK: br label %{{.*}}, !llvm.loop ![[WHILE_DISABLE:.*]]
}

void while_enable()
{
// CHECK-LABEL: while_enable
    int i = 1000;
    [unroll]
    while(i-->0);
// CHECK: br label %{{.*}}, !llvm.loop ![[WHILE_ENABLE:.*]]
}

/*** do ***/
void do_count()
{
// CHECK-LABEL: do_count
    int i = 1000;
    [unroll(16)]
    do {} while(i--> 0);
// CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop ![[DO_DISTINCT:.*]]
}

void do_disable()
{
// CHECK-LABEL: do_disable
    int i = 1000;
    [loop]
    do {} while(i--> 0);
// CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop ![[DO_DISABLE:.*]]
}

void do_enable()
{
// CHECK-LABEL: do_enable
    int i = 1000;
    [unroll]
    do {} while(i--> 0);
// CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop ![[DO_ENABLE:.*]]
}


// CHECK: ![[FOR_DISTINCT]]     =  distinct !{![[FOR_DISTINCT]],  ![[FOR_COUNT:.*]]}
// CHECK: ![[FOR_COUNT]]         =  !{!"llvm.loop.unroll.count", i32 8}
// CHECK: ![[FOR_DISABLE]]   =  distinct !{![[FOR_DISABLE]],  ![[DISABLE:.*]]}
// CHECK: ![[DISABLE]]       =  !{!"llvm.loop.unroll.disable"}
// CHECK: ![[FOR_ENABLE]]      =  distinct !{![[FOR_ENABLE]],  ![[ENABLE:.*]]}
// CHECK: ![[ENABLE]]          =  !{!"llvm.loop.unroll.enable"}
// CHECK: ![[FOR_NESTED_ENABLE]] =  distinct !{![[FOR_NESTED_ENABLE]], ![[ENABLE]]}
// CHECK: ![[FOR_NESTED2_ENABLE]] =  distinct !{![[FOR_NESTED2_ENABLE]], ![[ENABLE]]}
// CHECK: ![[FOR_NESTED2_1_ENABLE]] =  distinct !{![[FOR_NESTED2_1_ENABLE]], ![[ENABLE]]}
// CHECK: ![[WHILE_DISTINCT]]   =  distinct !{![[WHILE_DISTINCT]],    ![[WHILE_COUNT:.*]]}
// CHECK: ![[WHILE_COUNT]]         =  !{!"llvm.loop.unroll.count", i32 4}
// CHECK: ![[WHILE_DISABLE]] =  distinct !{![[WHILE_DISABLE]],  ![[DISABLE]]}
// CHECK: ![[WHILE_ENABLE]]    =  distinct !{![[WHILE_ENABLE]],     ![[ENABLE]]}
// CHECK: ![[DO_DISTINCT]]      =  distinct !{![[DO_DISTINCT]],       ![[DO_COUNT:.*]]}
// CHECK: ![[DO_COUNT]]         =  !{!"llvm.loop.unroll.count", i32 16}
// CHECK: ![[DO_DISABLE]]    =  distinct !{![[DO_DISABLE]],     ![[DISABLE]]}
// CHECK: ![[DO_ENABLE]]       =  distinct !{![[DO_ENABLE]],        ![[ENABLE]]}
