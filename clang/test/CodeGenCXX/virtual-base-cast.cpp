// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - -triple i686-pc-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - -triple i686-pc-win32 | FileCheck -check-prefix MSVC %s

struct A { int a; virtual int aa(); };
struct B { int b; virtual int bb(); };
struct C : virtual A, virtual B { int c; virtual int aa(); virtual int bb(); };
struct AA { int a; virtual int aa(); };
struct BB { int b; virtual int bb(); };
struct CC : AA, BB { virtual int aa(); virtual int bb(); virtual int cc(); };
struct D : virtual C, virtual CC { int e; };

D* x;

A* a() { return x; }
// CHECK: @_Z1av() [[NUW:#[0-9]+]]
// CHECK: [[VBASEOFFSETPTRA:%[a-zA-Z0-9\.]+]] = getelementptr i8, i8* {{.*}}, i64 -16
// CHECK: [[CASTVBASEOFFSETPTRA:%[a-zA-Z0-9\.]+]] = bitcast i8* [[VBASEOFFSETPTRA]] to i32*
// CHECK: load i32, i32* [[CASTVBASEOFFSETPTRA]]
// CHECK: }

// MSVC: @"?a@@YAPAUA@@XZ"() [[NUW:#[0-9]+]] {
// MSVC:   %[[vbptr_off:.*]] = getelementptr inbounds i8, i8* {{.*}}, i32 0
// MSVC:   %[[vbptr:.*]] = bitcast i8* %[[vbptr_off]] to i32**
// MSVC:   %[[vbtable:.*]] = load i32*, i32** %[[vbptr]]
// MSVC:   %[[entry:.*]] = getelementptr inbounds i32, i32* {{.*}}, i32 1
// MSVC:   %[[offset:.*]] = load i32, i32* %[[entry]]
// MSVC:   add nsw i32 0, %[[offset]]
// MSVC: }

B* b() { return x; }
// CHECK: @_Z1bv() [[NUW]]
// CHECK: [[VBASEOFFSETPTRA:%[a-zA-Z0-9\.]+]] = getelementptr i8, i8* {{.*}}, i64 -20
// CHECK: [[CASTVBASEOFFSETPTRA:%[a-zA-Z0-9\.]+]] = bitcast i8* [[VBASEOFFSETPTRA]] to i32*
// CHECK: load i32, i32* [[CASTVBASEOFFSETPTRA]]
// CHECK: }

// Same as 'a' except we use a different vbtable offset.
// MSVC: @"?b@@YAPAUB@@XZ"() [[NUW:#[0-9]+]] {
// MSVC:   %[[vbptr_off:.*]] = getelementptr inbounds i8, i8* {{.*}}, i32 0
// MSVC:   %[[vbptr:.*]] = bitcast i8* %[[vbptr_off]] to i32**
// MSVC:   %[[vbtable:.*]] = load i32*, i32** %[[vbptr]]
// MSVC:   %[[entry:.*]] = getelementptr inbounds i32, i32* {{.*}}, i32 2
// MSVC:   %[[offset:.*]] = load i32, i32* %[[entry]]
// MSVC:   add nsw i32 0, %[[offset]]
// MSVC: }


BB* c() { return x; }
// CHECK: @_Z1cv() [[NUW]]
// CHECK: [[VBASEOFFSETPTRC:%[a-zA-Z0-9\.]+]] = getelementptr i8, i8* {{.*}}, i64 -24
// CHECK: [[CASTVBASEOFFSETPTRC:%[a-zA-Z0-9\.]+]] = bitcast i8* [[VBASEOFFSETPTRC]] to i32*
// CHECK: [[VBASEOFFSETC:%[a-zA-Z0-9\.]+]] = load i32, i32* [[CASTVBASEOFFSETPTRC]]
// CHECK: add i32 [[VBASEOFFSETC]], 8
// CHECK: }

// Same as 'a' except we use a different vbtable offset.
// MSVC: @"?c@@YAPAUBB@@XZ"() [[NUW:#[0-9]+]] {
// MSVC:   %[[vbptr_off:.*]] = getelementptr inbounds i8, i8* {{.*}}, i32 0
// MSVC:   %[[vbptr:.*]] = bitcast i8* %[[vbptr_off]] to i32**
// MSVC:   %[[vbtable:.*]] = load i32*, i32** %[[vbptr]]
// MSVC:   %[[entry:.*]] = getelementptr inbounds i32, i32* {{.*}}, i32 4
// MSVC:   %[[offset:.*]] = load i32, i32* %[[entry]]
// MSVC:   add nsw i32 0, %[[offset]]
// MSVC: }

// Put the vbptr at a non-zero offset inside a non-virtual base.
struct E { int e; };
struct F : E, D { int f; };

F* y;

BB* d() { return y; }

// Same as 'c' except the vbptr offset is 4, changing the initial GEP and the
// final add.
// MSVC: @"?d@@YAPAUBB@@XZ"() [[NUW:#[0-9]+]] {
// MSVC:   %[[vbptr_off:.*]] = getelementptr inbounds i8, i8* {{.*}}, i32 4
// MSVC:   %[[vbptr:.*]] = bitcast i8* %[[vbptr_off]] to i32**
// MSVC:   %[[vbtable:.*]] = load i32*, i32** %[[vbptr]]
// MSVC:   %[[entry:.*]] = getelementptr inbounds i32, i32* {{.*}}, i32 4
// MSVC:   %[[offset:.*]] = load i32, i32* %[[entry]]
// MSVC:   add nsw i32 4, %[[offset]]
// MSVC: }

// CHECK: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }
