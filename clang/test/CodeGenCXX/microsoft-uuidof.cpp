// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -fms-extensions | FileCheck %s

typedef struct _GUID
{
    unsigned long  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];
} GUID;

struct __declspec(uuid("12345678-1234-1234-1234-1234567890ab")) S1 { } s1;
struct __declspec(uuid("87654321-4321-4321-4321-ba0987654321")) S2 { };

// This gets initialized in a static initializer.
// CHECK: @g = global %struct._GUID zeroinitializer, align 4
GUID g = __uuidof(S1);

// First global use of __uuidof(S1) forces the creation of the global.
// CHECK: @__uuid_12345678-1234-1234-1234-1234567890ab = private unnamed_addr constant %struct._GUID { i32 305419896, i16 4660, i16 4660, [8 x i8] c"\124\124Vx\90\AB" }
// CHECK: @gr = constant %struct._GUID* @__uuid_12345678-1234-1234-1234-1234567890ab, align 4
const GUID& gr = __uuidof(S1);

// CHECK: @gp = global %struct._GUID* @__uuid_12345678-1234-1234-1234-1234567890ab, align 4
const GUID* gp = &__uuidof(S1);

// Special case: _uuidof(0)
// CHECK: @zeroiid = constant %struct._GUID* @__uuid_00000000-0000-0000-0000-000000000000, align 4
const GUID& zeroiid = __uuidof(0);

// __uuidof(S2) hasn't been used globally yet, so it's emitted when it's used
// in a function and is emitted at the end of the globals section.
// CHECK: @__uuid_87654321-4321-4321-4321-ba0987654321 = private unnamed_addr constant %struct._GUID { i32 -2023406815, i16 17185, i16 17185, [8 x i8] c"C!\BA\09\87eC!" }

// The static initializer for g.
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast (%struct._GUID* @g to i8*), i8* bitcast (%struct._GUID* @__uuid_12345678-1234-1234-1234-1234567890ab to i8*), i32 16, i32 4, i1 false)

void fun() {
  // CHECK: %s1_1 = alloca %struct._GUID, align 4
  // CHECK: %s1_2 = alloca %struct._GUID, align 4
  // CHECK: %s1_3 = alloca %struct._GUID, align 4

  // CHECK: [[U1:%.+]] = bitcast %struct._GUID* %s1_1 to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[U1]], i8* bitcast (%struct._GUID* @__uuid_12345678-1234-1234-1234-1234567890ab to i8*), i32 16, i32 4, i1 false)
  GUID s1_1 = __uuidof(S1);

  // CHECK: [[U2:%.+]] = bitcast %struct._GUID* %s1_2 to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[U2]], i8* bitcast (%struct._GUID* @__uuid_12345678-1234-1234-1234-1234567890ab to i8*), i32 16, i32 4, i1 false)
  GUID s1_2 = __uuidof(S1);

  // CHECK: [[U3:%.+]] = bitcast %struct._GUID* %s1_3 to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[U3]], i8* bitcast (%struct._GUID* @__uuid_12345678-1234-1234-1234-1234567890ab to i8*), i32 16, i32 4, i1 false)
  GUID s1_3 = __uuidof(s1);
}

void gun() {
  // CHECK: %s2_1 = alloca %struct._GUID, align 4
  // CHECK: %s2_2 = alloca %struct._GUID, align 4
  // CHECK: %r = alloca %struct._GUID*, align 4
  // CHECK: %p = alloca %struct._GUID*, align 4
  // CHECK: %zeroiid = alloca %struct._GUID*, align 4
  GUID s2_1 = __uuidof(S2);
  GUID s2_2 = __uuidof(S2);

  // CHECK: store %struct._GUID* @__uuid_87654321-4321-4321-4321-ba0987654321, %struct._GUID** %r, align 4
  const GUID& r = __uuidof(S2);
  // CHECK: store %struct._GUID* @__uuid_87654321-4321-4321-4321-ba0987654321, %struct._GUID** %p, align 4
  const GUID* p = &__uuidof(S2);

  // Special case _uuidof(0), local scope version.
  // CHECK: store %struct._GUID* @__uuid_00000000-0000-0000-0000-000000000000, %struct._GUID** %zeroiid, align 4
  const GUID& zeroiid = __uuidof(0);
}
