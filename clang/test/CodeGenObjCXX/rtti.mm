// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// PR7864.  This all follows GCC's lead.

namespace std { class type_info; }

// CHECK: @_ZTI1A = linkonce_odr constant {{.*}}@_ZTVN10__cxxabiv117__class_type_infoE{{.*}}@_ZTS1A
@interface A
@end

// CHECK: @_ZTI1B = linkonce_odr constant {{.*}}@_ZTVN10__cxxabiv120__si_class_type_infoE{{.*}}@_ZTS1B{{.*}}@_ZTI1A
@interface B : A
@end

// CHECK: @_ZTIP1B = linkonce_odr constant {{.*}}@_ZTVN10__cxxabiv119__pointer_type_infoE{{.*}}@_ZTSP1B{{.*}}, i32 0, {{.*}}@_ZTI1B
// CHECK: @_ZTI11objc_object = linkonce_odr constant {{.*}}@_ZTVN10__cxxabiv117__class_type_infoE{{.*}}@_ZTS11objc_object
// CHECK: @_ZTIP11objc_object = linkonce_odr constant {{.*}}@_ZTVN10__cxxabiv119__pointer_type_infoE{{.*}}@_ZTSP11objc_object{{.*}}@_ZTI11objc_object
// CHECK: @_ZTI10objc_class = linkonce_odr constant {{.*}}@_ZTVN10__cxxabiv117__class_type_infoE{{.*}}@_ZTS10objc_class
// CHECK: @_ZTIP10objc_class = linkonce_odr constant {{.*}}@_ZTVN10__cxxabiv119__pointer_type_infoE{{.*}}@_ZTSP10objc_class{{.*}}@_ZTI10objc_class

@protocol P;

int main() {
  // CHECK: store {{.*}} @_ZTIP1B
  // CHECK: store {{.*}} @_ZTI1B
  const std::type_info &t1 = typeid(B*);
  const std::type_info &t2 = typeid(B);

  // CHECK: store {{.*}} @_ZTIP11objc_object
  // CHECK: store {{.*}} @_ZTI11objc_object
  id i = 0;
  const std::type_info &t3 = typeid(i);
  const std::type_info &t4 = typeid(*i);

  // CHECK: store {{.*}} @_ZTIP10objc_class
  // CHECK: store {{.*}} @_ZTI10objc_class
  Class c = 0;
  const std::type_info &t5 = typeid(c);
  const std::type_info &t6 = typeid(*c);

  // CHECK: store {{.*}} @_ZTIPU11objcproto1P11objc_object
  // CHECK: store {{.*}} @_ZTIU11objcproto1P11objc_object
  id<P> i2 = 0;
  const std::type_info &t7 = typeid(i2);
  const std::type_info &t8 = typeid(*i2);

  // CHECK: store {{.*}} @_ZTIPU11objcproto1P10objc_class
  // CHECK: store {{.*}} @_ZTIU11objcproto1P10objc_class
  Class<P> c2 = 0;
  const std::type_info &t9 = typeid(c2);
  const std::type_info &t10 = typeid(*c2);
}
