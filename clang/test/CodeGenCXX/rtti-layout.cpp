// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -O3 -o - | FileCheck %s
#include <typeinfo>

// vtables.
extern "C" {
  const void *_ZTVN10__cxxabiv123__fundamental_type_infoE;
  const void *_ZTVN10__cxxabiv117__class_type_infoE;
  const void *_ZTVN10__cxxabiv120__si_class_type_infoE;
  const void *_ZTVN10__cxxabiv121__vmi_class_type_infoE;
  const void *_ZTVN10__cxxabiv119__pointer_type_infoE;
  const void *_ZTVN10__cxxabiv129__pointer_to_member_type_infoE;
};
#define fundamental_type_info_vtable _ZTVN10__cxxabiv123__fundamental_type_infoE
#define class_type_info_vtable _ZTVN10__cxxabiv117__class_type_infoE
#define si_class_type_info_vtable _ZTVN10__cxxabiv120__si_class_type_infoE
#define vmi_class_type_info_vtable _ZTVN10__cxxabiv121__vmi_class_type_infoE
#define pointer_type_info_vtable _ZTVN10__cxxabiv119__pointer_type_infoE
#define pointer_to_member_type_info_vtable _ZTVN10__cxxabiv129__pointer_to_member_type_infoE

class __pbase_type_info : public std::type_info {
public:
  unsigned int __flags;
  const std::type_info *__pointee;

  enum __masks {
    __const_mask = 0x1,
    __volatile_mask = 0x2,
    __restrict_mask = 0x4,
    __incomplete_mask = 0x8,
    __incomplete_class_mask = 0x10
  };
};

class __class_type_info : public std::type_info { };

class __si_class_type_info : public __class_type_info {
public:
  const __class_type_info *__base_type;
};

struct __base_class_type_info {
public:
 const __class_type_info *__base_type;
 long __offset_flags;

 enum __offset_flags_masks {
   __virtual_mask = 0x1,
   __public_mask = 0x2,
   __offset_shift = 8
 };
};

class __vmi_class_type_info : public __class_type_info {
public:
  unsigned int __flags;
  unsigned int __base_count;
  __base_class_type_info __base_info[1];

  enum __flags_masks {
    __non_diamond_repeat_mask = 0x1,
    __diamond_shaped_mask = 0x2
  };
};

template<typename T> const T& to(const std::type_info &info) {
return static_cast<const T&>(info);
}
struct Incomplete;

struct A { int a; };
struct Empty { };

struct SI1 : A { };
struct SI2 : Empty { };
struct SI3 : Empty { virtual void f() { } };

struct VMI1 : private A { };
struct VMI2 : virtual A { };
struct VMI3 : A { virtual void f() { } };
struct VMI4 : A, Empty { };

struct VMIBase1 { int a; };
struct VMIBase2 : VMIBase1 { int a; };
struct VMI5 : VMIBase1, VMIBase2 { int a; };

struct VMIBase3 : virtual VMIBase1 { int a; };
struct VMI6 : virtual VMIBase1, VMIBase3 { int a; };

struct VMI7 : VMIBase1, VMI5, private VMI6 { };

#define CHECK(x) if (!(x)) return __LINE__
#define CHECK_VTABLE(type, vtable) CHECK(&vtable##_type_info_vtable + 2 == (((void **)&(typeid(type)))[0]))
#define CHECK_BASE_INFO_TYPE(type, index, base) CHECK(to<__vmi_class_type_info>(typeid(type)).__base_info[(index)].__base_type == &typeid(base))
#define CHECK_BASE_INFO_OFFSET_FLAGS(type, index, offset, flags) CHECK(to<__vmi_class_type_info>(typeid(type)).__base_info[(index)].__offset_flags == (((offset) << 8) | (flags)))

struct B {
  static int const volatile (*a)[10];
  static int (*b)[10];
  
  static int const volatile (B::*c)[10];
  static int (B::*d)[10];
};

// CHECK-LABEL: define{{.*}} i32 @_Z1fv()
int f() {
  // Vectors should be treated as fundamental types.
  typedef short __v4hi __attribute__ ((__vector_size__ (8)));
  CHECK_VTABLE(__v4hi, fundamental);

  // A does not have any bases.
  CHECK_VTABLE(A, class);
  
  // SI1 has a single public base.
  CHECK_VTABLE(SI1, si_class);
  CHECK(to<__si_class_type_info>(typeid(SI1)).__base_type == &typeid(A));
  
  // SI2 has a single public empty base.
  CHECK_VTABLE(SI2, si_class);
  CHECK(to<__si_class_type_info>(typeid(SI2)).__base_type == &typeid(Empty));

  // SI3 has a single public empty base. SI3 is dynamic whereas Empty is not, but since Empty is
  // an empty class, it will still be at offset zero.
  CHECK_VTABLE(SI3, si_class);
  CHECK(to<__si_class_type_info>(typeid(SI3)).__base_type == &typeid(Empty));

  // VMI1 has a single base, but it is private.
  CHECK_VTABLE(VMI1, vmi_class);

  // VMI2 has a single base, but it is virtual.
  CHECK_VTABLE(VMI2, vmi_class);

  // VMI3 has a single base, but VMI3 is dynamic whereas A is not, and A is not empty.
  CHECK_VTABLE(VMI3, vmi_class);

  // VMI4 has two bases.
  CHECK_VTABLE(VMI4, vmi_class);

  // VMI5 has non-diamond shaped inheritance.
  CHECK_VTABLE(VMI5, vmi_class);
  CHECK(to<__vmi_class_type_info>(typeid(VMI5)).__flags == __vmi_class_type_info::__non_diamond_repeat_mask);
  CHECK(to<__vmi_class_type_info>(typeid(VMI5)).__base_count == 2);
  CHECK_BASE_INFO_TYPE(VMI5, 0, VMIBase1);
  CHECK_BASE_INFO_OFFSET_FLAGS(VMI5, 0, 0, __base_class_type_info::__public_mask);
  CHECK_BASE_INFO_TYPE(VMI5, 1, VMIBase2);
  CHECK_BASE_INFO_OFFSET_FLAGS(VMI5, 1, 4, __base_class_type_info::__public_mask);
  
  // VMI6 has diamond shaped inheritance.
  CHECK_VTABLE(VMI6, vmi_class);
  CHECK(to<__vmi_class_type_info>(typeid(VMI6)).__flags == __vmi_class_type_info::__diamond_shaped_mask);
  CHECK(to<__vmi_class_type_info>(typeid(VMI6)).__base_count == 2);
  CHECK_BASE_INFO_TYPE(VMI6, 0, VMIBase1);
  CHECK_BASE_INFO_OFFSET_FLAGS(VMI6, 0, -24, __base_class_type_info::__public_mask | __base_class_type_info::__virtual_mask);
  CHECK_BASE_INFO_TYPE(VMI6, 1, VMIBase3);
  CHECK_BASE_INFO_OFFSET_FLAGS(VMI6, 1, 0, __base_class_type_info::__public_mask);
  
  // VMI7 has both non-diamond and diamond shaped inheritance.
  CHECK_VTABLE(VMI7, vmi_class);
  CHECK(to<__vmi_class_type_info>(typeid(VMI7)).__flags == (__vmi_class_type_info::__non_diamond_repeat_mask | __vmi_class_type_info::__diamond_shaped_mask));
  CHECK(to<__vmi_class_type_info>(typeid(VMI7)).__base_count == 3);
  CHECK_BASE_INFO_TYPE(VMI7, 0, VMIBase1);
  CHECK_BASE_INFO_OFFSET_FLAGS(VMI7, 0, 16, __base_class_type_info::__public_mask);
  CHECK_BASE_INFO_TYPE(VMI7, 1, VMI5);
  CHECK_BASE_INFO_OFFSET_FLAGS(VMI7, 1, 20, __base_class_type_info::__public_mask);
  CHECK_BASE_INFO_TYPE(VMI7, 2, VMI6);
  CHECK_BASE_INFO_OFFSET_FLAGS(VMI7, 2, 0, 0);
  
  // Pointers to incomplete classes.
  CHECK_VTABLE(Incomplete *, pointer);
  CHECK(to<__pbase_type_info>(typeid(Incomplete *)).__flags == __pbase_type_info::__incomplete_mask);
  CHECK(to<__pbase_type_info>(typeid(Incomplete **)).__flags == __pbase_type_info::__incomplete_mask);
  CHECK(to<__pbase_type_info>(typeid(Incomplete ***)).__flags == __pbase_type_info::__incomplete_mask);

  // Member pointers.
  CHECK_VTABLE(int Incomplete::*, pointer_to_member);
  CHECK(to<__pbase_type_info>(typeid(int Incomplete::*)).__flags == __pbase_type_info::__incomplete_class_mask);
  CHECK(to<__pbase_type_info>(typeid(Incomplete Incomplete::*)).__flags == (__pbase_type_info::__incomplete_class_mask | __pbase_type_info::__incomplete_mask));
  CHECK(to<__pbase_type_info>(typeid(Incomplete A::*)).__flags == (__pbase_type_info::__incomplete_mask));

  // Check that when stripping qualifiers off the pointee type, we correctly handle arrays.
  CHECK(to<__pbase_type_info>(typeid(B::a)).__flags == (__pbase_type_info::__const_mask | __pbase_type_info::__volatile_mask));
  CHECK(to<__pbase_type_info>(typeid(B::a)).__pointee == to<__pbase_type_info>(typeid(B::b)).__pointee);
  CHECK(to<__pbase_type_info>(typeid(B::c)).__flags == (__pbase_type_info::__const_mask | __pbase_type_info::__volatile_mask));
  CHECK(to<__pbase_type_info>(typeid(B::c)).__pointee == to<__pbase_type_info>(typeid(B::d)).__pointee);

  // Success!
  // CHECK: ret i32 0
  return 0;
}

#ifdef HARNESS
extern "C" void printf(const char *, ...);

int main() {
  int result = f();
  
  if (result == 0)
    printf("success!\n");
  else
    printf("test on line %d failed!\n", result);

  return result;
}
#endif


