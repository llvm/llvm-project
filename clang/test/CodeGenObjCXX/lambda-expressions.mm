// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -fblocks -fobjc-arc | FileCheck -check-prefix=ARC %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -fblocks | FileCheck -check-prefix=MRC %s

typedef int (^fp)();
fp f() { auto x = []{ return 3; }; return x; }

// MRC: @"\01L_OBJC_METH_VAR_NAME{{.*}}" = internal global [5 x i8] c"copy\00"
// MRC: @"\01L_OBJC_METH_VAR_NAME{{.*}}" = internal global [12 x i8] c"autorelease\00"
// MRC: define i32 ()* @_Z1fv(
// MRC: define internal i32 ()* @"_ZZ1fvENK3$_0cvU13block_pointerFivEEv"
// MRC: store i8* bitcast (i8** @_NSConcreteStackBlock to i8*)
// MRC: store i8* bitcast (i32 (i8*)* @"___ZZ1fvENK3$_0cvU13block_pointerFivEEv_block_invoke" to i8*)
// MRC: call i32 ()* (i8*, i8*)* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 ()* (i8*, i8*)*)
// MRC: call i32 ()* (i8*, i8*)* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 ()* (i8*, i8*)*)
// MRC: ret i32 ()*

// ARC: define i32 ()* @_Z1fv(
// ARC: define internal i32 ()* @"_ZZ1fvENK3$_0cvU13block_pointerFivEEv"
// ARC: store i8* bitcast (i8** @_NSConcreteStackBlock to i8*)
// ARC: store i8* bitcast (i32 (i8*)* @"___ZZ1fvENK3$_0cvU13block_pointerFivEEv_block_invoke" to i8*)
// ARC: call i8* @objc_retainBlock
// ARC: call i8* @objc_autoreleaseReturnValue

typedef int (^fp)();
fp global;
void f2() { global = []{ return 3; }; }

// MRC: define void @_Z2f2v() nounwind {
// MRC: store i8* bitcast (i32 (i8*)* @___Z2f2v_block_invoke to i8*),
// MRC-NOT: call
// MRC: ret void
// ("global" contains a dangling pointer after this function runs.)

// ARC: define void @_Z2f2v() nounwind {
// ARC: store i8* bitcast (i32 (i8*)* @___Z2f2v_block_invoke to i8*),
// ARC: call i8* @objc_retainBlock
// ARC: call void @objc_release
// ARC: define internal i32 @___Z2f2v_block_invoke
// ARC: call i32 @"_ZZ2f2vENK3$_1clEv
