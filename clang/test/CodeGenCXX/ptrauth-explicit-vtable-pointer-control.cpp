// RUN: %clang_cc1 %s -x c++ -std=c++11  -triple arm64-apple-ios -fptrauth-intrinsics -fptrauth-calls -emit-llvm -O1 -disable-llvm-passes  -o - | FileCheck  --check-prefix=CHECK-DEFAULT-NONE %s
// RUN: %clang_cc1 %s -x c++ -std=c++11  -triple arm64-apple-ios -fptrauth-intrinsics -fptrauth-calls -fptrauth-vtable-pointer-type-discrimination -emit-llvm -O1 -disable-llvm-passes -o - | FileCheck --check-prefix=CHECK-DEFAULT-TYPE %s
// RUN: %clang_cc1 %s -x c++ -std=c++11  -triple arm64-apple-ios -fptrauth-intrinsics -fptrauth-calls -fptrauth-vtable-pointer-address-discrimination -emit-llvm -O1 -disable-llvm-passes -o - | FileCheck --check-prefix=CHECK-DEFAULT-ADDRESS %s
// RUN: %clang_cc1 %s -x c++ -std=c++11  -triple arm64-apple-ios -fptrauth-intrinsics -fptrauth-calls -fptrauth-vtable-pointer-type-discrimination -fptrauth-vtable-pointer-address-discrimination -emit-llvm -O1 -disable-llvm-passes -o - | FileCheck --check-prefix=CHECK-DEFAULT-BOTH %s
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

struct authenticated(default_key, default_address_discrimination, custom_discrimination, 0xf00d) ExplicitCustomDiscrimination {
  virtual ~ExplicitCustomDiscrimination();
  virtual void f();
  virtual void g();
};

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
struct PrimaryBasicStruct<T> *make_primary_basic(T *);
template <typename T>
struct SecondaryBasicStruct : T, BasicStruct {};
template <typename T>
struct SecondaryBasicStruct<T> *make_secondary_basic(T *);
template <typename T>
struct VirtualSubClass : virtual T {
  virtual void g();
  virtual T *h();
};
template <typename T>
struct VirtualPrimaryStruct : virtual T, VirtualSubClass<T> {};
template <typename T>
struct VirtualPrimaryStruct<T> *make_virtual_primary(T *);
template <typename T>
struct VirtualSecondaryStruct : VirtualSubClass<T>, virtual T {};
template <typename T>
struct VirtualSecondaryStruct<T> *make_virtual_secondary(T *);

// CHECK-DEFAULT-NONE: _ZN5test14testEPNS_14NoExplicitAuthEPNS_21ExplicitlyDisableAuthEPNS_29ExplicitAddressDiscriminationEPNS_31ExplicitNoAddressDiscriminationEPNS_29ExplicitNoExtraDiscriminationEPNS_26ExplicitTypeDiscriminationEPNS_28ExplicitCustomDiscriminationE
void test(NoExplicitAuth *a, ExplicitlyDisableAuth *b, ExplicitAddressDiscrimination *c,
          ExplicitNoAddressDiscrimination *d, ExplicitNoExtraDiscrimination *e,
          ExplicitTypeDiscrimination *f, ExplicitCustomDiscrimination *g) {
  a->f();
  // CHECK-DEFAULT-NONE: %0 = load ptr, ptr %a.addr, align 8, !tbaa !2
  // CHECK-DEFAULT-NONE: %vtable = load ptr, ptr %0, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %1 = ptrtoint ptr %vtable to i64
  // CHECK-DEFAULT-NONE: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 0)
  // CHECK-DEFAULT-NONE: %3 = inttoptr i64 %2 to ptr
  // CHECK-DEFAULT-NONE: %vfn = getelementptr inbounds ptr, ptr %3, i64 2

  b->f();
  // CHECK-DEFAULT-NONE: %7 = load ptr, ptr %b.addr, align 8, !tbaa !2
  // CHECK-DEFAULT-NONE: %vtable1 = load ptr, ptr %7, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %vfn2 = getelementptr inbounds ptr, ptr %vtable1, i64 2

  c->f();
  // CHECK-DEFAULT-NONE: %11 = load ptr, ptr %c.addr, align 8, !tbaa !2
  // CHECK-DEFAULT-NONE: %vtable3 = load ptr, ptr %11, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %12 = ptrtoint ptr %11 to i64
  // CHECK-DEFAULT-NONE: %13 = ptrtoint ptr %vtable3 to i64
  // CHECK-DEFAULT-NONE: %14 = call i64 @llvm.ptrauth.auth(i64 %13, i32 2, i64 %12)
  // CHECK-DEFAULT-NONE: %15 = inttoptr i64 %14 to ptr
  // CHECK-DEFAULT-NONE: %vfn4 = getelementptr inbounds ptr, ptr %15, i64 2

  d->f();
  // CHECK-DEFAULT-NONE: %19 = load ptr, ptr %d.addr, align 8, !tbaa !2
  // CHECK-DEFAULT-NONE: %vtable5 = load ptr, ptr %19, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %20 = ptrtoint ptr %vtable5 to i64
  // CHECK-DEFAULT-NONE: %21 = call i64 @llvm.ptrauth.auth(i64 %20, i32 2, i64 0)
  // CHECK-DEFAULT-NONE: %22 = inttoptr i64 %21 to ptr
  // CHECK-DEFAULT-NONE: %vfn6 = getelementptr inbounds ptr, ptr %22, i64 2

  e->f();
  // CHECK-DEFAULT-NONE: %26 = load ptr, ptr %e.addr, align 8, !tbaa !2
  // CHECK-DEFAULT-NONE: %vtable7 = load ptr, ptr %26, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %27 = ptrtoint ptr %vtable7 to i64
  // CHECK-DEFAULT-NONE: %28 = call i64 @llvm.ptrauth.auth(i64 %27, i32 2, i64 0)
  // CHECK-DEFAULT-NONE: %29 = inttoptr i64 %28 to ptr
  // CHECK-DEFAULT-NONE: %vfn8 = getelementptr inbounds ptr, ptr %29, i64 2

  f->f();
  // CHECK-DEFAULT-NONE: %33 = load ptr, ptr %f.addr, align 8, !tbaa !2
  // CHECK-DEFAULT-NONE: %vtable9 = load ptr, ptr %33, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %34 = ptrtoint ptr %vtable9 to i64
  // CHECK-DEFAULT-NONE: %35 = call i64 @llvm.ptrauth.auth(i64 %34, i32 2, i64 6177)
  // CHECK-DEFAULT-NONE: %36 = inttoptr i64 %35 to ptr
  // CHECK-DEFAULT-NONE: %vfn10 = getelementptr inbounds ptr, ptr %36, i64 2
 

  g->f();
  // CHECK-DEFAULT-NONE: %40 = load ptr, ptr %g.addr, align 8, !tbaa !2
  // CHECK-DEFAULT-NONE: %vtable11 = load ptr, ptr %40, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %41 = ptrtoint ptr %vtable11 to i64
  // CHECK-DEFAULT-NONE: %42 = call i64 @llvm.ptrauth.auth(i64 %41, i32 2, i64 61453)
  // CHECK-DEFAULT-NONE: %43 = inttoptr i64 %42 to ptr
  // CHECK-DEFAULT-NONE: %vfn12 = getelementptr inbounds ptr, ptr %43, i64 2

  // basic subclass
  make_subclass(a)->f();
  // CHECK-DEFAULT-NONE: %vtable13 = load ptr, ptr %call, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %48 = ptrtoint ptr %vtable13 to i64
  // CHECK-DEFAULT-NONE: %49 = call i64 @llvm.ptrauth.auth(i64 %48, i32 2, i64 0)
  // CHECK-DEFAULT-NONE: %50 = inttoptr i64 %49 to ptr
  // CHECK-DEFAULT-NONE: %vfn14 = getelementptr inbounds ptr, ptr %50, i64 2
 

  make_subclass(a)->g();
  // CHECK-DEFAULT-NONE: %vtable16 = load ptr, ptr %call15, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %55 = ptrtoint ptr %vtable16 to i64
  // CHECK-DEFAULT-NONE: %56 = call i64 @llvm.ptrauth.auth(i64 %55, i32 2, i64 0)
  // CHECK-DEFAULT-NONE: %57 = inttoptr i64 %56 to ptr
  // CHECK-DEFAULT-NONE: %vfn17 = getelementptr inbounds ptr, ptr %57, i64 3

  make_subclass(a)->h();
  // CHECK-DEFAULT-NONE: %vtable19 = load ptr, ptr %call18, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %62 = ptrtoint ptr %vtable19 to i64
  // CHECK-DEFAULT-NONE: %63 = call i64 @llvm.ptrauth.auth(i64 %62, i32 2, i64 0)
  // CHECK-DEFAULT-NONE: %64 = inttoptr i64 %63 to ptr
  // CHECK-DEFAULT-NONE: %vfn20 = getelementptr inbounds ptr, ptr %64, i64 4

  make_subclass(b)->f();
  // CHECK-DEFAULT-NONE: %vtable23 = load ptr, ptr %call22, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %vfn24 = getelementptr inbounds ptr, ptr %vtable23, i64 2

  make_subclass(b)->g();
  // CHECK-DEFAULT-NONE: %vtable26 = load ptr, ptr %call25, align 8, !tbaa !6
  // CHECK-DEFAULT-NONE: %vfn27 = getelementptr inbounds ptr, ptr %vtable26, i64 3

  make_subclass(b)->h();

  make_subclass(c)->f();
  make_subclass(c)->g();
  make_subclass(c)->h();

  make_subclass(d)->f();
  make_subclass(d)->g();
  make_subclass(d)->h();

  make_subclass(e)->f();
  make_subclass(e)->g();
  make_subclass(e)->h();

  make_subclass(f)->f();
  make_subclass(f)->g();
  make_subclass(f)->h();

  make_subclass(g)->f();
  make_subclass(g)->g();
  make_subclass(g)->h();

  // Basic multiple inheritance
  make_primary_basic(a)->f();
  make_primary_basic(b)->f();
  make_primary_basic(c)->f();
  make_primary_basic(d)->f();
  make_primary_basic(e)->f();
  make_primary_basic(f)->f();
  make_primary_basic(g)->f();
  make_secondary_basic(a)->f();
  make_secondary_basic(b)->f();
  make_secondary_basic(c)->f();
  make_secondary_basic(d)->f();
  make_secondary_basic(e)->f();
  make_secondary_basic(f)->f();
  make_secondary_basic(g)->f();

  // virtual inheritance
  make_virtual_primary(a)->f();
  make_virtual_primary(b)->f();
  make_virtual_primary(c)->f();
  make_virtual_primary(d)->f();
  make_virtual_primary(e)->f();
  make_virtual_primary(f)->f();
  make_virtual_primary(g)->f();
  make_virtual_secondary(a)->f();
  make_virtual_secondary(b)->f();
  make_virtual_secondary(c)->f();
  make_virtual_secondary(d)->f();
  make_virtual_secondary(e)->f();
  make_virtual_secondary(f)->f();
  make_virtual_secondary(g)->f();
}
} // namespace test1

// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 37831)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 2191)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 44989)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 63209)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43275)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 19073)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 25182)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 23051)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 3267)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 57764)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 8498)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61320)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 7682)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 53776)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 49565)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6177)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 61453)
// CHECK-DEFAULT-TYPE:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)

// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 37831)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 2191)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 44989)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 63209)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43275)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 19073)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 25182)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 23051)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 3267)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 57764)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 8498)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61320)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 7682)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 53776)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 0)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-ADDRESS:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)

// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 37831)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 2191)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 44989)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 63209)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43275)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 19073)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 25182)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 23051)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 3267)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 57764)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 8498)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61320)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 7682)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 53776)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 49565)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 27707)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 31119)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 56943)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 5268)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 6022)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 34147)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 39413)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 6177)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 29468)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 61453)
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T:%.*]], i32 2, i64 [[T:%.*]])
// CHECK-DEFAULT-BOTH:   [[T:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T:%.*]], i64 43175)
