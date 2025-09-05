// RUN: %clang_cc1 %s -x c++ -std=c++20 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -emit-llvm -o - | FileCheck --check-prefixes=CHECK,NODISC %s

// RUN: %clang_cc1 %s -x c++ -std=c++20 -triple arm64-apple-ios   -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -emit-llvm -o - | FileCheck --check-prefixes=CHECK,TYPE %s

// RUN: %clang_cc1 %s -x c++ -std=c++20 -triple arm64-apple-ios   -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -emit-llvm -o - | FileCheck --check-prefixes=CHECK,ADDR %s

// RUN: %clang_cc1 %s -x c++ -std=c++20 -triple arm64-apple-ios   -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -emit-llvm -o - | FileCheck --check-prefixes=CHECK,BOTH %s

// RUN: %clang_cc1 %s -x c++ -std=c++20 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -emit-llvm -o - | FileCheck --check-prefixes=CHECK,NODISC %s

// RUN: %clang_cc1 %s -x c++ -std=c++20 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -emit-llvm -o - | FileCheck --check-prefixes=CHECK,TYPE %s

// RUN: %clang_cc1 %s -x c++ -std=c++20 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -emit-llvm -o - | FileCheck --check-prefixes=CHECK,ADDR %s

// RUN: %clang_cc1 %s -x c++ -std=c++20 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -emit-llvm -o - | FileCheck --check-prefixes=CHECK,BOTH %s

#include <ptrauth.h>

namespace test1 {

#define authenticated(a...) ptrauth_cxx_vtable_pointer(a)

struct NoExplicitAuth {
  virtual ~NoExplicitAuth();
  virtual void f();
  virtual void g();
};

struct authenticated(no_authentication, no_address_discrimination, no_extra_discrimination) ExplicitlyDisableAuth {
  virtual ~ExplicitlyDisableAuth();
  virtual void f();
  virtual void g();
};

struct authenticated(default_key, address_discrimination, default_extra_discrimination) ExplicitAddressDiscrimination {
  virtual ~ExplicitAddressDiscrimination();
  virtual void f();
  virtual void g();
};

struct authenticated(default_key, no_address_discrimination, default_extra_discrimination) ExplicitNoAddressDiscrimination {
  virtual ~ExplicitNoAddressDiscrimination();
  virtual void f();
  virtual void g();
};

struct authenticated(default_key, default_address_discrimination, no_extra_discrimination) ExplicitNoExtraDiscrimination {
  virtual ~ExplicitNoExtraDiscrimination();
  virtual void f();
  virtual void g();
};

struct authenticated(default_key, default_address_discrimination, type_discrimination) ExplicitTypeDiscrimination {
  virtual ~ExplicitTypeDiscrimination();
  virtual void f();
  virtual void g();
};

struct authenticated(default_key, default_address_discrimination, custom_discrimination, 42424) ExplicitCustomDiscrimination {
  virtual ~ExplicitCustomDiscrimination();
  virtual void f();
  virtual void g();
};

// CHECK: @_ZTVN5test19ConstEvalE = external unnamed_addr constant { [3 x ptr] }, align 8
// CHECK: @_ZN5test12ceE = global %{{.*}} { ptr ptrauth (ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTVN5test19ConstEvalE, i32 0, i32 0, i32 2), i32 2, i64 0, ptr @_ZN5test12ceE) }, align 8
// CHECK: @_ZTVN5test116ConstEvalDerivedE = linkonce_odr unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTIN5test116ConstEvalDerivedE, ptr ptrauth (ptr @_ZN5test19ConstEval1fEv, i32 0, i64 26259, ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTVN5test116ConstEvalDerivedE, i32 0, i32 0, i32 2))] },{{.*}}align 8
// CHECK: @_ZN5test13cedE = global { ptr } { ptr ptrauth (ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTVN5test116ConstEvalDerivedE, i32 0, i32 0, i32 2), i32 2, i64 0, ptr @_ZN5test13cedE) }, align 8

struct authenticated(default_key, address_discrimination, no_extra_discrimination) ConstEval {
  consteval ConstEval() {}
  virtual void f();
};

// clang used to bail out with error message "could not emit constant value abstractly".
ConstEval ce;

struct ConstEvalDerived : public ConstEval {
public:
  consteval ConstEvalDerived() {}
};

// clang used to emit an undef initializer.
ConstEvalDerived ced;

template <typename T>
struct SubClass : T {
  virtual void g();
  virtual T *h();
};

template <typename T>
SubClass<T> *make_subclass(T *);

struct authenticated(default_key, address_discrimination, type_discrimination) BasicStruct {
  virtual ~BasicStruct();
};

template <typename T>
struct PrimaryBasicStruct : BasicStruct, T {};
template <typename T>
struct PrimaryBasicStruct<T> *make_multiple_primary(T *);

template <typename T>
struct VirtualSubClass : virtual T {
  virtual void g();
  virtual T *h();
};
template <typename T>
struct VirtualPrimaryStruct : virtual T, VirtualSubClass<T> {};
template <typename T>
struct VirtualPrimaryStruct<T> *make_virtual_primary(T *);

extern "C" {

// CHECK: @TVDisc_NoExplicitAuth = global i32 [[DISC_DEFAULT:49565]], align 4
int TVDisc_NoExplicitAuth = ptrauth_string_discriminator("_ZTVN5test114NoExplicitAuthE");

// CHECK: @TVDisc_ExplicitlyDisableAuth = global i32 [[DISC_DISABLED:24369]], align 4
int TVDisc_ExplicitlyDisableAuth = ptrauth_string_discriminator("_ZTVN5test121ExplicitlyDisableAuthE");

// CHECK: @TVDisc_ExplicitAddressDiscrimination = global i32 [[DISC_ADDR:56943]], align 4
int TVDisc_ExplicitAddressDiscrimination = ptrauth_string_discriminator("_ZTVN5test129ExplicitAddressDiscriminationE");

// CHECK: @TVDisc_ExplicitNoAddressDiscrimination = global i32 [[DISC_NO_ADDR:6022]], align 4
int TVDisc_ExplicitNoAddressDiscrimination = ptrauth_string_discriminator("_ZTVN5test131ExplicitNoAddressDiscriminationE");

// CHECK: @TVDisc_ExplicitNoExtraDiscrimination = global i32 [[DISC_NO_EXTRA:9072]], align 4
int TVDisc_ExplicitNoExtraDiscrimination = ptrauth_string_discriminator("_ZTVN5test129ExplicitNoExtraDiscriminationE");

// CHECK: @TVDisc_ExplicitTypeDiscrimination = global i32 [[DISC_TYPE:6177]], align 4
int TVDisc_ExplicitTypeDiscrimination = ptrauth_string_discriminator("_ZTVN5test126ExplicitTypeDiscriminationE");


// CHECK-LABEL: define{{.*}} void @test_default(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = load ptr, ptr {{%.*}}, align 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_DEFAULT]])
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_DEFAULT]])
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_default(NoExplicitAuth *a) {
  a->f();
}

// CHECK-LABEL: define{{.*}} void @test_disabled(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = load ptr, ptr {{%.*}}, align 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
// CHECK-NOT:     call i64 @llvm.ptrauth.auth
void test_disabled(ExplicitlyDisableAuth *a) {
  a->f();
}

// CHECK-LABEL: define{{.*}} void @test_addr_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = load ptr, ptr {{%.*}}, align 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// TYPE:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// TYPE:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_ADDR]])
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_ADDR]])
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_addr_disc(ExplicitAddressDiscrimination *a) {
  a->f();
}

// CHECK-LABEL: define{{.*}} void @test_no_addr_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = load ptr, ptr {{%.*}}, align 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_NO_ADDR]])
//
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_NO_ADDR]])
void test_no_addr_disc(ExplicitNoAddressDiscrimination *a) {
  a->f();
}

// CHECK-LABEL: define{{.*}} void @test_no_extra_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = load ptr, ptr {{%.*}}, align 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
void test_no_extra_disc(ExplicitNoExtraDiscrimination *a) {
  a->f();
}

// CHECK-LABEL: define{{.*}} void @test_type_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = load ptr, ptr {{%.*}}, align 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_TYPE]])
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_TYPE]])
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_TYPE]])
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_TYPE]])
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_type_disc(ExplicitTypeDiscrimination *a) {
  a->f();
}

// CHECK-LABEL: define{{.*}} void @test_custom_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = load ptr, ptr {{%.*}}, align 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 42424)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 42424)
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 42424)
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 42424)
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_custom_disc(ExplicitCustomDiscrimination *a) {
  a->f();
}

//
// Test some simple single inheritance cases.
// Codegen should be the same as the simple cases above once we have a vtable.
//

// CHECK-LABEL: define{{.*}} void @test_subclass_default(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = call noundef ptr @_ZN5test113make_subclass
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_DEFAULT]])
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_DEFAULT]])
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_subclass_default(NoExplicitAuth *a) {
  make_subclass(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_subclass_disabled(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = call noundef ptr @_ZN5test113make_subclass
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
// CHECK-NOT:     call i64 @llvm.ptrauth.auth
void test_subclass_disabled(ExplicitlyDisableAuth *a) {
  make_subclass(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_subclass_addr_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = call noundef ptr @_ZN5test113make_subclass
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// TYPE:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// TYPE:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_ADDR]])
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_ADDR]])
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_subclass_addr_disc(ExplicitAddressDiscrimination *a) {
  make_subclass(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_subclass_no_addr_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = call noundef ptr @_ZN5test113make_subclass
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_NO_ADDR]])
//
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_NO_ADDR]])
void test_subclass_no_addr_disc(ExplicitNoAddressDiscrimination *a) {
  make_subclass(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_subclass_no_extra_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = call noundef ptr @_ZN5test113make_subclass
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
void test_subclass_no_extra_disc(ExplicitNoExtraDiscrimination *a) {
  make_subclass(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_subclass_type_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = call noundef ptr @_ZN5test113make_subclass
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_TYPE]])
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_TYPE]])
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_TYPE]])
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_TYPE]])
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_subclass_type_disc(ExplicitTypeDiscrimination *a) {
  make_subclass(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_subclass_custom_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTADDR:%.*]] = call noundef ptr @_ZN5test113make_subclass
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 42424)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 42424)
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 42424)
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 42424)
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_subclass_custom_disc(ExplicitCustomDiscrimination *a) {
  make_subclass(a)->f();
}


//
// Test some simple multiple inheritance cases.
// Codegen should be the same as the simple cases above once we have a vtable.
//

// CHECK-LABEL: define{{.*}} void @test_multiple_default(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[CALL:%.*]] = call noundef ptr @_ZN5test121make_multiple_primary
// CHECK:         [[VTADDR:%.*]] = getelementptr inbounds i8, ptr [[CALL]], i64 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_DEFAULT]])
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_DEFAULT]])
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_multiple_default(NoExplicitAuth *a) {
  make_multiple_primary(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_multiple_disabled(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[CALL:%.*]] = call noundef ptr @_ZN5test121make_multiple_primary
// CHECK:         [[VTADDR:%.*]] = getelementptr inbounds i8, ptr [[CALL]], i64 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
// CHECK-NOT:     call i64 @llvm.ptrauth.auth
void test_multiple_disabled(ExplicitlyDisableAuth *a) {
  make_multiple_primary(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_multiple_custom_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[CALL:%.*]] = call noundef ptr @_ZN5test121make_multiple_primary
// CHECK:         [[VTADDR:%.*]] = getelementptr inbounds i8, ptr [[CALL]], i64 8
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 42424)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 42424)
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 42424)
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 42424)
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_multiple_custom_disc(ExplicitCustomDiscrimination *a) {
  make_multiple_primary(a)->f();
}

//
// Test some virtual inheritance cases.
// Codegen should be the same as the simple cases above once we have a vtable,
// but twice for vtt/vtable.  The names in the vtt version have "VTT" prefixes.
//

// CHECK-LABEL: define{{.*}} void @test_virtual_default(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTTADDR:%.*]] = call noundef ptr @_ZN5test120make_virtual_primary
// CHECK:         [[VTTABLE:%.*]] = load ptr, ptr [[VTTADDR]], align 8
//
// NODISC:        [[VTTABLEI64:%.*]] = ptrtoint ptr [[VTTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTTABLEI64]], i32 2, i64 0)
//
// TYPE:          [[VTTABLEI64:%.*]] = ptrtoint ptr [[VTTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTTABLEI64]], i32 2, i64 [[DISC_DEFAULT]])
//
// ADDR:          [[VTTADDRI64:%.*]] = ptrtoint ptr [[VTTADDR]] to i64
// ADDR:          [[VTTABLEI64:%.*]] = ptrtoint ptr [[VTTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTTABLEI64]], i32 2, i64 [[VTTADDRI64]])
//
// BOTH:          [[VTTADDRI64:%.*]] = ptrtoint ptr [[VTTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTTADDRI64]], i64 [[DISC_DEFAULT]])
// BOTH:          [[VTTABLEI64:%.*]] = ptrtoint ptr [[VTTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTTABLEI64]], i32 2, i64 [[BLEND]])

// CHECK:         [[AUTHEDPTR:%.*]] = inttoptr i64 [[AUTHED]] to ptr
// CHECK:         [[VBOFFPTR:%.*]] = getelementptr i8, ptr [[AUTHEDPTR]], i64 -48
// CHECK:         [[VBOFFSET:%.*]] = load i64, ptr [[VBOFFPTR]]
// CHECK:         [[VTADDR:%.*]] = getelementptr inbounds i8, ptr [[VTTADDR]], i64 [[VBOFFSET]]
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 0)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[DISC_DEFAULT]])
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[VTADDRI64]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 [[DISC_DEFAULT]])
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_virtual_default(NoExplicitAuth *a) {
  make_virtual_primary(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_virtual_disabled(ptr noundef {{%.*}}) {{#.*}} {
// CHECK-NOT:     call i64 @llvm.ptrauth.auth
void test_virtual_disabled(ExplicitlyDisableAuth *a) {
  make_virtual_primary(a)->f();
}

// CHECK-LABEL: define{{.*}} void @test_virtual_custom_disc(ptr noundef {{%.*}}) {{#.*}} {
// CHECK:         [[VTTADDR:%.*]] = call noundef ptr @_ZN5test120make_virtual_primary
// CHECK:         [[VTTABLE:%.*]] = load ptr, ptr [[VTTADDR]], align 8
//
// NODISC:        [[VTTABLEI64:%.*]] = ptrtoint ptr [[VTTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTTABLEI64]], i32 2, i64 42424)
//
// TYPE:          [[VTTABLEI64:%.*]] = ptrtoint ptr [[VTTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTTABLEI64]], i32 2, i64 42424)
//
// ADDR:          [[VTTADDRI64:%.*]] = ptrtoint ptr [[VTTADDR]] to i64
// ADDR:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTTADDRI64]], i64 42424)
// ADDR:          [[VTTABLEI64:%.*]] = ptrtoint ptr [[VTTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTTABLEI64]], i32 2, i64 [[BLEND]])
//
// BOTH:          [[VTTADDRI64:%.*]] = ptrtoint ptr [[VTTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTTADDRI64]], i64 42424)
// BOTH:          [[VTTABLEI64:%.*]] = ptrtoint ptr [[VTTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTTABLEI64]], i32 2, i64 [[BLEND]])

// CHECK:         [[AUTHEDPTR:%.*]] = inttoptr i64 [[AUTHED]] to ptr
// CHECK:         [[VBOFFPTR:%.*]] = getelementptr i8, ptr [[AUTHEDPTR]], i64 -48
// CHECK:         [[VBOFFSET:%.*]] = load i64, ptr [[VBOFFPTR]]
// CHECK:         [[VTADDR:%.*]] = getelementptr inbounds i8, ptr [[VTTADDR]], i64 [[VBOFFSET]]
// CHECK:         [[VTABLE:%.*]] = load ptr, ptr [[VTADDR]], align 8
//
// NODISC:        [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// NODISC:        [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 42424)
//
// TYPE:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// TYPE:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 42424)
//
// ADDR:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// ADDR:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 42424)
// ADDR:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// ADDR:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
//
// BOTH:          [[VTADDRI64:%.*]] = ptrtoint ptr [[VTADDR]] to i64
// BOTH:          [[BLEND:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTADDRI64]], i64 42424)
// BOTH:          [[VTABLEI64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// BOTH:          [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI64]], i32 2, i64 [[BLEND]])
void test_virtual_custom_disc(ExplicitCustomDiscrimination *a) {
  make_virtual_primary(a)->f();
}

} // extern "C"
} // namespace test1
