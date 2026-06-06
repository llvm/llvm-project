// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
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
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_NESTED_ENABLE_INNER:.*]]
// CHECK: br label %{{.*}}, !llvm.loop ![[FOR_NESTED_ENABLE_OUTER:.*]]
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


// CHECK-DAG: [[MUST_PROGRESS:.*]] = !{!"llvm.loop.mustprogress"}
// CHECK-DAG: [[DISABLE:.*]] = !{!"llvm.loop.unroll.disable"}
// CHECK-DAG: [[FOR_COUNT:.*]] =  !{!"llvm.loop.unroll.count", i32 8}
// CHECK-DAG: [[ENABLE:.*]] = !{!"llvm.loop.unroll.enable"}
// CHECK-DAG: [[WHILE_COUNT:.*]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK-DAG: [[DO_COUNT:.*]] = !{!"llvm.loop.unroll.count", i32 16}

// CHECK-DAG: ![[FOR_DISTINCT]] =  distinct !{![[FOR_DISTINCT]], [[MUST_PROGRESS]], [[FOR_COUNT]]}
// CHECK-DAG: ![[FOR_DISABLE]] =  distinct !{![[FOR_DISABLE]], [[MUST_PROGRESS]], [[DISABLE]]}
// CHECK-DAG: ![[FOR_ENABLE]] =  distinct !{![[FOR_ENABLE]], [[MUST_PROGRESS]], [[ENABLE]]}

// CHECK-DAG: ![[FOR_NESTED_ENABLE_INNER]] = distinct !{![[FOR_NESTED_ENABLE_INNER]], [[MUST_PROGRESS]]}
// CHECK-DAG: ![[FOR_NESTED_ENABLE_OUTER]] = distinct !{![[FOR_NESTED_ENABLE_OUTER]], [[MUST_PROGRESS]], [[ENABLE]]}
// CHECK-DAG: ![[FOR_NESTED2_ENABLE]] =  distinct !{![[FOR_NESTED2_ENABLE]], [[MUST_PROGRESS]], [[ENABLE]]}
// CHECK-DAG: ![[FOR_NESTED2_1_ENABLE]] =  distinct !{![[FOR_NESTED2_1_ENABLE]], [[MUST_PROGRESS]], [[ENABLE]]}
// CHECK-DAG: ![[WHILE_DISTINCT]]   =  distinct !{![[WHILE_DISTINCT]], [[MUST_PROGRESS]], [[WHILE_COUNT]]}

// CHECK-DAG: ![[WHILE_DISABLE]] =  distinct !{![[WHILE_DISABLE]], [[MUST_PROGRESS]], [[DISABLE]]}
// CHECK-DAG: ![[WHILE_ENABLE]] =  distinct !{![[WHILE_ENABLE]], [[MUST_PROGRESS]], [[ENABLE]]}
// CHECK-DAG: ![[DO_DISTINCT]]  =  distinct !{![[DO_DISTINCT]], [[MUST_PROGRESS]], [[DO_COUNT]]}

// CHECK-DAG: ![[DO_DISABLE]] =  distinct !{![[DO_DISABLE]], [[MUST_PROGRESS]], [[DISABLE]]}
// CHECK-DAG: ![[DO_ENABLE]] =  distinct !{![[DO_ENABLE]], [[MUST_PROGRESS]], [[ENABLE]]}
