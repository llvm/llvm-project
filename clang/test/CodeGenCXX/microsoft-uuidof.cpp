// RUN: %clang_cc1 -emit-llvm %s -o - -DDEFINE_GUID -triple=i386-pc-linux -fms-extensions | FileCheck %s --check-prefix=CHECK-DEFINE-GUID
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-linux -fms-extensions | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-linux -fms-extensions | FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 -emit-llvm %s -o - -DDEFINE_GUID -DWRONG_GUID -triple=i386-pc-linux -fms-extensions | FileCheck %s --check-prefix=CHECK-DEFINE-WRONG-GUID

#ifdef DEFINE_GUID
struct _GUID {
#ifdef WRONG_GUID
    unsigned int SomethingWentWrong;
#else
    unsigned long  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];
#endif
};
#endif
typedef struct _GUID GUID;

struct __declspec(uuid("12345678-1234-1234-1234-1234567890aB")) S1 { } s1;
struct __declspec(uuid("87654321-4321-4321-4321-ba0987654321")) S2 { };
struct __declspec(uuid("{12345678-1234-1234-1234-1234567890ac}")) Curly;
struct __declspec(uuid("{12345678-1234-1234-1234-1234567890ac}")) Curly;

#ifdef DEFINE_GUID
// Make sure we can properly generate code when the UUID has curly braces on it.
GUID thing = __uuidof(Curly);
// CHECK-DEFINE-GUID: @thing = global %struct._GUID zeroinitializer, align 4
// CHECK-DEFINE-WRONG-GUID: @thing = global %struct._GUID zeroinitializer, align 4

// This gets initialized in a static initializer.
// CHECK-DEFINE-GUID: @g = global %struct._GUID zeroinitializer, align 4
// CHECK-DEFINE-WRONG-GUID: @g = global %struct._GUID zeroinitializer, align 4
GUID g = __uuidof(S1);
#endif

// First global use of __uuidof(S1) forces the creation of the global.
// CHECK: @_GUID_12345678_1234_1234_1234_1234567890ab = linkonce_odr constant { i32, i16, i16, [8 x i8] } { i32 305419896, i16 4660, i16 4660, [8 x i8] c"\124\124Vx\90\AB" }, comdat
// CHECK: @gr = constant %struct._GUID* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to %struct._GUID*), align 4
// CHECK-64: @gr = constant %struct._GUID* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to %struct._GUID*), align 8
const GUID& gr = __uuidof(S1);

// CHECK: @gp = global %struct._GUID* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to %struct._GUID*), align 4
const GUID* gp = &__uuidof(S1);

// CHECK: @cp = global %struct._GUID* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ac to %struct._GUID*), align 4
const GUID* cp = &__uuidof(Curly);

// Special case: _uuidof(0)
// CHECK: @zeroiid = constant %struct._GUID* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_00000000_0000_0000_0000_000000000000 to %struct._GUID*), align 4
const GUID& zeroiid = __uuidof(0);

// __uuidof(S2) hasn't been used globally yet, so it's emitted when it's used
// in a function and is emitted at the end of the globals section.
// CHECK: @_GUID_87654321_4321_4321_4321_ba0987654321 = linkonce_odr constant { i32, i16, i16, [8 x i8] } { i32 -2023406815, i16 17185, i16 17185, [8 x i8] c"C!\BA\09\87eC!" }, comdat

// The static initializer for thing.
// CHECK-DEFINE-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast (%struct._GUID* @thing to i8*), i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ac to i8*), i32 16, i32 4, i1 false)
// CHECK-DEFINE-WRONG-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast (%struct._GUID* @thing to i8*), i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ac to i8*), i32 4, i32 4, i1 false)

// The static initializer for g.
// CHECK-DEFINE-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast (%struct._GUID* @g to i8*), i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to i8*), i32 16, i32 4, i1 false)
// CHECK-DEFINE-WRONG-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast (%struct._GUID* @g to i8*), i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to i8*), i32 4, i32 4, i1 false)

#ifdef DEFINE_GUID
void fun() {
  // CHECK-DEFINE-GUID: %s1_1 = alloca %struct._GUID, align 4
  // CHECK-DEFINE-WRONG-GUID: %s1_1 = alloca %struct._GUID, align 4
  // CHECK-DEFINE-GUID: %s1_2 = alloca %struct._GUID, align 4
  // CHECK-DEFINE-WRONG-GUID: %s1_2 = alloca %struct._GUID, align 4
  // CHECK-DEFINE-GUID: %s1_3 = alloca %struct._GUID, align 4
  // CHECK-DEFINE-WRONG-GUID: %s1_3 = alloca %struct._GUID, align 4

  // CHECK-DEFINE-GUID: [[U1:%.+]] = bitcast %struct._GUID* %s1_1 to i8*
  // CHECK-DEFINE-WRONG-GUID: [[U1:%.+]] = bitcast %struct._GUID* %s1_1 to i8*
  // CHECK-DEFINE-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[U1]], i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to i8*), i32 16, i32 4, i1 false)
  // CHECK-DEFINE-WRONG-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[U1]], i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to i8*), i32 4, i32 4, i1 false)
  GUID s1_1 = __uuidof(S1);

  // CHECK-DEFINE-GUID: [[U2:%.+]] = bitcast %struct._GUID* %s1_2 to i8*
  // CHECK-DEFINE-WRONG-GUID: [[U2:%.+]] = bitcast %struct._GUID* %s1_2 to i8*
  // CHECK-DEFINE-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[U2]], i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to i8*), i32 16, i32 4, i1 false)
  // CHECK-DEFINE-WRONG-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[U2]], i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to i8*), i32 4, i32 4, i1 false)
  GUID s1_2 = __uuidof(S1);

  // CHECK-DEFINE-GUID: [[U3:%.+]] = bitcast %struct._GUID* %s1_3 to i8*
  // CHECK-DEFINE-WRONG-GUID: [[U3:%.+]] = bitcast %struct._GUID* %s1_3 to i8*
  // CHECK-DEFINE-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[U3]], i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to i8*), i32 16, i32 4, i1 false)
  // CHECK-DEFINE-WRONG-GUID: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[U3]], i8* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab to i8*), i32 4, i32 4, i1 false)
  GUID s1_3 = __uuidof(s1);
}
#endif

void gun() {
#ifdef DEFINE_GUID
  // CHECK-DEFINE-GUID: %s2_1 = alloca %struct._GUID, align 4
  // CHECK-DEFINE-WRONG-GUID: %s2_1 = alloca %struct._GUID, align 4
  // CHECK-DEFINE-GUID: %s2_2 = alloca %struct._GUID, align 4
  // CHECK-DEFINE-WRONG-GUID: %s2_2 = alloca %struct._GUID, align 4
  GUID s2_1 = __uuidof(S2);
  GUID s2_2 = __uuidof(S2);
#endif
  // CHECK: %r = alloca %struct._GUID*, align 4
  // CHECK: %p = alloca %struct._GUID*, align 4
  // CHECK: %zeroiid = alloca %struct._GUID*, align 4

  // CHECK: store %struct._GUID* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_87654321_4321_4321_4321_ba0987654321 to %struct._GUID*), %struct._GUID** %r, align 4
  const GUID& r = __uuidof(S2);
  // CHECK: store %struct._GUID* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_87654321_4321_4321_4321_ba0987654321 to %struct._GUID*), %struct._GUID** %p, align 4
  const GUID* p = &__uuidof(S2);

  // Special case _uuidof(0), local scope version.
  // CHECK: store %struct._GUID* bitcast ({ i32, i16, i16, [8 x i8] }* @_GUID_00000000_0000_0000_0000_000000000000 to %struct._GUID*), %struct._GUID** %zeroiid, align 4
  const GUID& zeroiid = __uuidof(0);
}
