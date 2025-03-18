// RUN: %clang_cc1 -DENABLE_TID=0 -I%S -std=c++11 -triple=arm64e-apple-darwin \
// RUN:   -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NODISC

// RUN: %clang_cc1 -DENABLE_TID=0 -I%S -std=c++11 -triple=aarch64-linux-gnu \
// RUN:   -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NODISC

// RUN: %clang_cc1 -DENABLE_TID=1 -I%S -std=c++11 -triple=arm64e-apple-darwin \
// RUN:   -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-type-info-vtable-pointer-discrimination \
// RUN:   %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,DISC

// RUN: %clang_cc1 -DENABLE_TID=1 -I%S -std=c++11 -triple=aarch64-linux-gnu \
// RUN:   -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-type-info-vtable-pointer-discrimination \
// RUN:   %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,DISC

// copied from typeinfo
namespace std {

#if __has_cpp_attribute(clang::ptrauth_vtable_pointer)
#  if __has_feature(ptrauth_type_info_vtable_pointer_discrimination)
#    define _LIBCPP_TYPE_INFO_VTABLE_POINTER_AUTH \
       [[clang::ptrauth_vtable_pointer(process_independent, address_discrimination, type_discrimination)]]
#  else
#    define _LIBCPP_TYPE_INFO_VTABLE_POINTER_AUTH \
       [[clang::ptrauth_vtable_pointer(process_independent, no_address_discrimination, no_extra_discrimination)]]
#  endif
#else
#  define _LIBCPP_TYPE_INFO_VTABLE_POINTER_AUTH
#endif

  class _LIBCPP_TYPE_INFO_VTABLE_POINTER_AUTH type_info
  {
    type_info& operator=(const type_info&);
    type_info(const type_info&);

  protected:
      explicit type_info(const char* __n);

  public:
      virtual ~type_info();

      virtual void test_method();
  };
} // namespace std

static_assert(__has_feature(ptrauth_type_info_vtable_pointer_discrimination) == ENABLE_TID, "incorrect feature state");

// CHECK: @disc_std_type_info = global i32 [[STDTYPEINFO_DISC:45546]]
extern "C" int disc_std_type_info = __builtin_ptrauth_string_discriminator("_ZTVSt9type_info");

// CHECK: @_ZTV10TestStruct = unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI10TestStruct, ptr ptrauth (ptr @_ZN10TestStructD1Ev, i32 0, i64 52216, ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV10TestStruct, i32 0, i32 0, i32 2)), ptr ptrauth (ptr @_ZN10TestStructD0Ev, i32 0, i64 39671, ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV10TestStruct, i32 0, i32 0, i32 3))] }, align 8

// NODISC: @_ZTI10TestStruct = constant { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2), ptr @_ZTS10TestStruct }, align 8

// DISC: @_ZTI10TestStruct = constant { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2, i64 [[STDTYPEINFO_DISC]], ptr @_ZTI10TestStruct), ptr @_ZTS10TestStruct }, align 8

// CHECK: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK: @_ZTS10TestStruct = constant [13 x i8] c"10TestStruct\00", align 1

struct TestStruct {
  virtual ~TestStruct();
  int a;
};

TestStruct::~TestStruct(){}

extern "C" void test_vtable(std::type_info* t) {
  t->test_method();
}
// NODISC: define{{.*}} void @test_vtable(ptr noundef %t)
// NODISC: [[T_ADDR:%.*]] = alloca ptr, align 8
// NODISC: store ptr %t, ptr [[T_ADDR]], align 8
// NODISC: [[T:%.*]] = load ptr, ptr [[T_ADDR]], align 8
// NODISC: [[VPTR:%.*]] = load ptr, ptr [[T]], align 8
// NODISC: [[CAST_VPTR:%.*]] = ptrtoint ptr [[VPTR]] to i64
// NODISC: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[CAST_VPTR]], i32 2, i64 0)

// DISC: define{{.*}} void @test_vtable(ptr noundef %t)
// DISC: [[T_ADDR:%.*]] = alloca ptr, align 8
// DISC: store ptr %t, ptr [[T_ADDR]], align 8
// DISC: [[T:%.*]] = load ptr, ptr [[T_ADDR]], align 8
// DISC: [[VPTR:%.*]] = load ptr, ptr [[T]], align 8
// DISC: [[ADDR:%.*]] = ptrtoint ptr [[T]] to i64
// DISC: [[DISCRIMINATOR:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[ADDR]], i64 [[STDTYPEINFO_DISC]])
// DISC: [[VPTRI:%.*]] = ptrtoint ptr [[VPTR]] to i64
// DISC: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VPTRI]], i32 2, i64 [[DISCRIMINATOR]])

extern "C" const void *ensure_typeinfo() {
  return new TestStruct;
}
