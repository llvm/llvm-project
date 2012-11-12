// RUN: %clang_cc1 -emit-llvm %s -o - -triple i686-pc-linux-gnu | FileCheck %s

struct A { int a; virtual int aa(); };
struct B { int b; virtual int bb(); };
struct C : virtual A, virtual B { int c; virtual int aa(); virtual int bb(); };
struct AA { int a; virtual int aa(); };
struct BB { int b; virtual int bb(); };
struct CC : AA, BB { virtual int aa(); virtual int bb(); virtual int cc(); };
struct D : virtual C, virtual CC { int e; };

D* x;

A* a() { return x; }
// CHECK: @_Z1av() nounwind
// CHECK: [[VBASEOFFSETPTRA:%[a-zA-Z0-9\.]+]] = getelementptr i8* {{.*}}, i64 -16
// CHECK: [[CASTVBASEOFFSETPTRA:%[a-zA-Z0-9\.]+]] = bitcast i8* [[VBASEOFFSETPTRA]] to i32*
// CHECK: load i32* [[CASTVBASEOFFSETPTRA]]
// CHECK: }

B* b() { return x; }
// CHECK: @_Z1bv() nounwind
// CHECK: [[VBASEOFFSETPTRA:%[a-zA-Z0-9\.]+]] = getelementptr i8* {{.*}}, i64 -20
// CHECK: [[CASTVBASEOFFSETPTRA:%[a-zA-Z0-9\.]+]] = bitcast i8* [[VBASEOFFSETPTRA]] to i32*
// CHECK: load i32* [[CASTVBASEOFFSETPTRA]]
// CHECK: }

BB* c() { return x; }
// CHECK: @_Z1cv() nounwind
// CHECK: [[VBASEOFFSETPTRC:%[a-zA-Z0-9\.]+]] = getelementptr i8* {{.*}}, i64 -24
// CHECK: [[CASTVBASEOFFSETPTRC:%[a-zA-Z0-9\.]+]] = bitcast i8* [[VBASEOFFSETPTRC]] to i32*
// CHECK: [[VBASEOFFSETC:%[a-zA-Z0-9\.]+]] = load i32* [[CASTVBASEOFFSETPTRC]]
// CHECK: add i32 [[VBASEOFFSETC]], 8
// CHECK: }
