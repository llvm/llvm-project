// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin10 -o - %s | FileCheck %s

// PR6024
extern int i;

// CHECK: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i32* @_Z16lvalue_noop_castv() [[NUW:#[0-9]+]]
const int &lvalue_noop_cast() {
  if (i == 0)
    // CHECK: store i32 17, i32*
    return (const int&)17;
  else if (i == 1)
    // CHECK: store i32 17, i32*
    return static_cast<const int&>(17);
    // CHECK: store i32 17, i32*
  return 17;
}

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i16* @_Z20lvalue_integral_castv() 
const short &lvalue_integral_cast() {
  if (i == 0)
    // CHECK: store i16 17, i16*
    return (const short&)17;
  else if (i == 1)
    // CHECK: store i16 17, i16*
    return static_cast<const short&>(17);
  // CHECK: store i16 17, i16*
  return 17;
}

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i16* @_Z29lvalue_floating_integral_castv()
const short &lvalue_floating_integral_cast() {
  if (i == 0)
    // CHECK: store i16 17, i16*
    return (const short&)17.5;
  else if (i == 1)
    // CHECK: store i16 17, i16*
    return static_cast<const short&>(17.5);
  // CHECK: store i16 17, i16*
  return 17.5;
}

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) float* @_Z29lvalue_integral_floating_castv()
const float &lvalue_integral_floating_cast() {
  if (i == 0)
    // CHECK: store float 1.700000e+{{0*}}1, float*
    return (const float&)17;
  else if (i == 1)
    // CHECK: store float 1.700000e+{{0*}}1, float*
    return static_cast<const float&>(17);
  // CHECK: store float 1.700000e+{{0*}}1, float*
  return 17;
}

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) float* @_Z20lvalue_floating_castv()
const float &lvalue_floating_cast() {
  if (i == 0)
    // CHECK: store float 1.700000e+{{0*}}1, float*
    return (const float&)17.0;
  else if (i == 1)
    // CHECK: store float 1.700000e+{{0*}}1, float*
    return static_cast<const float&>(17.0);
  // CHECK: store float 1.700000e+{{0*}}1, float*
  return 17.0;
}

int get_int();

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i8* @_Z24lvalue_integer_bool_castv()
const bool &lvalue_integer_bool_cast() {
  if (i == 0)
    // CHECK: call i32 @_Z7get_intv()
    // CHECK: store i8
    return (const bool&)get_int();
  else if (i == 1)
    // CHECK: call i32 @_Z7get_intv()
    // CHECK: store i8
    return static_cast<const bool&>(get_int());
  // CHECK: call i32 @_Z7get_intv()
  // CHECK: store i8
  return get_int();
}

float get_float();

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i8* @_Z25lvalue_floating_bool_castv()
const bool &lvalue_floating_bool_cast() {
  if (i == 0)
    // CHECK: call float @_Z9get_floatv()
    // CHECK: fcmp une float
    // CHECK: store i8
    return (const bool&)get_float();
  else if (i == 1)
    // CHECK: call float @_Z9get_floatv()
    // CHECK: fcmp une float
    // CHECK: store i8
    return static_cast<const bool&>(get_float());
  // CHECK: call float @_Z9get_floatv()
  // CHECK: fcmp une float
  // CHECK: store i8
  return get_float();
}

struct X { };
typedef int X::*pm;
typedef int (X::*pmf)(int);

pm get_pointer_to_member_data();
pmf get_pointer_to_member_function();

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i8* @_Z26lvalue_ptrmem_to_bool_castv()
const bool &lvalue_ptrmem_to_bool_cast() {
  if (i == 0)
    // CHECK: call i64 @_Z26get_pointer_to_member_datav()
    // CHECK: store i8
    // CHECK: store i8*
    return (const bool&)get_pointer_to_member_data();
  else if (i == 1)
    // CHECK: call i64 @_Z26get_pointer_to_member_datav()
    // CHECK: store i8
    // CHECK: store i8*
    return static_cast<const bool&>(get_pointer_to_member_data());
  // CHECK: call i64 @_Z26get_pointer_to_member_datav()
  // CHECK: store i8
  // CHECK: store i8*
  return get_pointer_to_member_data();
}

// CHECK-LABEL: define{{.*}} nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) i8* @_Z27lvalue_ptrmem_to_bool_cast2v
const bool &lvalue_ptrmem_to_bool_cast2() {
  if (i == 0)
    // CHECK: {{call.*_Z30get_pointer_to_member_functionv}}
    // CHECK: store i8
    // CHECK: store i8*
    return (const bool&)get_pointer_to_member_function();
  else if (i == 1)
    // CHECK: {{call.*_Z30get_pointer_to_member_functionv}}
    // CHECK: store i8
    // CHECK: store i8*
    return static_cast<const bool&>(get_pointer_to_member_function());
  // CHECK: {{call.*_Z30get_pointer_to_member_functionv}}
  // CHECK: store i8
  // CHECK: store i8*
  return get_pointer_to_member_function();
}

_Complex double get_complex_double();

// CHECK: {{define.*_Z2f1v}}
const _Complex float &f1() {
  if (i == 0)
    // CHECK: {{call.*_Z18get_complex_doublev}}
    // CHECK: fptrunc
    // CHECK: fptrunc
    // CHECK: store float
    // CHECK: store float
    return (const _Complex float&)get_complex_double();
  else if (i == 1)
    // CHECK: {{call.*_Z18get_complex_doublev}}
    // CHECK: fptrunc
    // CHECK: fptrunc
    // CHECK: store float
    // CHECK: store float
    return static_cast<const _Complex float&>(get_complex_double());
  // CHECK: {{call.*_Z18get_complex_doublev}}
  // CHECK: fptrunc
  // CHECK: fptrunc
  // CHECK: store float
  // CHECK: store float
  return get_complex_double();
}

// CHECK-LABEL: define{{.*}} i32 @_Z7pr10592RKi(i32*
unsigned pr10592(const int &v) {
  // CHECK: [[VADDR:%[a-zA-Z0-9.]+]] = alloca i32*
  // CHECK-NEXT: [[REFTMP:%[a-zA-Z0-9.]+]] = alloca i32
  // CHECK-NEXT: store i32* [[V:%[a-zA-Z0-9.]+]], i32** [[VADDR]]
  // CHECK-NEXT: [[VADDR_1:%[a-zA-Z0-9.]+]] = load i32*, i32** [[VADDR]]
  // CHECK-NEXT: [[VVAL:%[a-zA-Z0-9.]+]] = load i32, i32* [[VADDR_1]]
  // CHECK-NEXT: store i32 [[VVAL]], i32* [[REFTMP]]
  // CHECK-NEXT: [[VVAL_I:%[a-zA-Z0-9.]+]] = load i32, i32* [[REFTMP]]
  // CHECK-NEXT: ret i32 [[VVAL_I]]
  return static_cast<const unsigned &>(v);
}

namespace PR10650 {
  struct Helper {
    unsigned long long id();
  };
  unsigned long long test(Helper *obj) {
    return static_cast<const unsigned long long&>(obj->id());
  }
  // CHECK-LABEL: define{{.*}} i64 @_ZN7PR106504testEPNS_6HelperE
  // CHECK: store i64
}

// CHECK: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }
