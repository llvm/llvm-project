// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -emit-llvm -pedantic -verify -O0 -o - | FileCheck %s
// expected-no-diagnostics

// Test that the 'this' pointer is in the __generic address space.

// FIXME: Add support for __constant address space.

class C {
public:
  int v;
  C() { v = 2; }
  C(C &&c) { v = c.v; }
  C(const C &c) { v = c.v; }
  C &operator=(const C &c) {
    v = c.v;
    return *this;
  }
  C &operator=(C &&c) & {
    v = c.v;
    return *this;
  }

  C operator+(const C& c) {
    v += c.v;
    return *this;
  }

  int get() { return v; }

  int outside();
};

int C::outside() {
  return v;
}

extern C&& foo();

__global C c;

__kernel void test__global() {
  int i = c.get();
  int i2 = c.outside();
  C c1(c);
  C c2;
  c2 = c1;
  C c3 = c1 + c2;
  C c4(foo());
  C c5 = foo();
}

// CHECK-LABEL: @__cxx_global_var_init()
// CHECK: call void @_ZNU3AS41CC1Ev(%class.C addrspace(4)* addrspacecast (%class.C addrspace(1)* @c to %class.C addrspace(4)*))

// Test that the address space is __generic for the constructor
// CHECK-LABEL: @_ZNU3AS41CC1Ev(%class.C addrspace(4)* %this)
// CHECK: entry:
// CHECK:   %this.addr = alloca %class.C addrspace(4)*, align 4
// CHECK:   store %class.C addrspace(4)* %this, %class.C addrspace(4)** %this.addr, align 4
// CHECK:   %this1 = load %class.C addrspace(4)*, %class.C addrspace(4)** %this.addr, align 4
// CHECK:   call void @_ZNU3AS41CC2Ev(%class.C addrspace(4)* %this1)
// CHECK:   ret void

// CHECK-LABEL: @_Z12test__globalv()

// Test the address space of 'this' when invoking a method.
// CHECK: %call = call i32 @_ZNU3AS41C3getEv(%class.C addrspace(4)* addrspacecast (%class.C addrspace(1)* @c to %class.C addrspace(4)*))

// Test the address space of 'this' when invoking a method that is declared in the file contex.
// CHECK: %call1 = call i32 @_ZNU3AS41C7outsideEv(%class.C addrspace(4)* addrspacecast (%class.C addrspace(1)* @c to %class.C addrspace(4)*))

// Test the address space of 'this' when invoking copy-constructor.
// CHECK: [[C1GEN:%[0-9]+]] = addrspacecast %class.C* %c1 to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1ERU3AS4KS_(%class.C addrspace(4)* [[C1GEN]], %class.C addrspace(4)* dereferenceable(4) addrspacecast (%class.C addrspace(1)* @c to %class.C addrspace(4)*))

// Test the address space of 'this' when invoking a constructor.
// CHECK:   [[C2GEN:%[0-9]+]] = addrspacecast %class.C* %c2 to %class.C addrspace(4)*
// CHECK:   call void @_ZNU3AS41CC1Ev(%class.C addrspace(4)* [[C2GEN]])

// Test the address space of 'this' when invoking assignment operator.
// CHECK:  [[C1GEN:%[0-9]+]] = addrspacecast %class.C* %c1 to %class.C addrspace(4)*
// CHECK:  [[C2GEN:%[0-9]+]] = addrspacecast %class.C* %c2 to %class.C addrspace(4)*
// CHECK:  %call2 = call dereferenceable(4) %class.C addrspace(4)* @_ZNU3AS41CaSERU3AS4KS_(%class.C addrspace(4)* [[C2GEN]], %class.C addrspace(4)* dereferenceable(4) [[C1GEN]])

// Test the address space of 'this' when invoking the operator+
// CHECK: [[C3GEN:%[0-9]+]] = addrspacecast %class.C* %c3 to %class.C addrspace(4)*
// CHECK: [[C1GEN:%[0-9]+]] = addrspacecast %class.C* %c1 to %class.C addrspace(4)*
// CHECK: [[C2GEN:%[0-9]+]] = addrspacecast %class.C* %c2 to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CplERU3AS4KS_(%class.C* sret %ref.tmp, %class.C addrspace(4)* [[C1GEN]], %class.C addrspace(4)* dereferenceable(4) [[C2GEN]])
// CHECK: [[REFGEN:%[0-9]+]] = addrspacecast %class.C* %ref.tmp to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1EOU3AS4S_(%class.C addrspace(4)* [[C3GEN]], %class.C addrspace(4)* dereferenceable(4) [[REFGEN]])

// Test the address space of 'this' when invoking the move constructor
// CHECK: [[C4GEN:%[0-9]+]] = addrspacecast %class.C* %c4 to %class.C addrspace(4)*
// CHECK: %call3 = call spir_func dereferenceable(4) %class.C addrspace(4)* @_Z3foov()
// CHECK: call void @_ZNU3AS41CC1EOU3AS4S_(%class.C addrspace(4)* [[C4GEN]], %class.C addrspace(4)* dereferenceable(4) %call3)

// Test the address space of 'this' when invoking the move assignment
// CHECK: [[C5GEN:%[0-9]+]] = addrspacecast %class.C* %c5 to %class.C addrspace(4)*
// CHECK: %call4 = call spir_func dereferenceable(4) %class.C addrspace(4)* @_Z3foov() #5
// CHECK: call void @_ZNU3AS41CC1EOU3AS4S_(%class.C addrspace(4)* [[C5GEN:%[0-9]+]], %class.C addrspace(4)* dereferenceable(4) %call4)


#define TEST(AS)             \
  __kernel void test##AS() { \
    AS C c;                  \
    int i = c.get();         \
    C c1(c);                 \
    C c2;                    \
    c2 = c1;                 \
  }

TEST(__local)

// CHECK-LABEL: _Z11test__localv
// CHECK: @__cxa_guard_acquire

// Test the address space of 'this' when invoking a constructor for an object in non-default address space
// CHECK: call void @_ZNU3AS41CC1Ev(%class.C addrspace(4)* addrspacecast (%class.C addrspace(3)* @_ZZ11test__localvE1c to %class.C addrspace(4)*))

// Test the address space of 'this' when invoking a method.
// CHECK: %call = call i32 @_ZNU3AS41C3getEv(%class.C addrspace(4)* addrspacecast (%class.C addrspace(3)* @_ZZ11test__localvE1c to %class.C addrspace(4)*))


// Test the address space of 'this' when invoking copy-constructor.
// CHECK: [[C1GEN:%[0-9]+]] = addrspacecast %class.C* %c1 to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1ERU3AS4KS_(%class.C addrspace(4)* [[C1GEN]], %class.C addrspace(4)* dereferenceable(4) addrspacecast (%class.C addrspace(3)* @_ZZ11test__localvE1c to %class.C addrspace(4)*))

// Test the address space of 'this' when invoking a constructor.
// CHECK: [[C2GEN:%[0-9]+]] = addrspacecast %class.C* %c2 to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1Ev(%class.C addrspace(4)* [[C2GEN]])

// Test the address space of 'this' when invoking assignment operator.
// CHECK: [[C1GEN:%[0-9]+]] = addrspacecast %class.C* %c1 to %class.C addrspace(4)*
// CHECK: [[C2GEN:%[0-9]+]] = addrspacecast %class.C* %c2 to %class.C addrspace(4)*
// CHECK: %call1 = call dereferenceable(4) %class.C addrspace(4)* @_ZNU3AS41CaSERU3AS4KS_(%class.C addrspace(4)* [[C2GEN]], %class.C addrspace(4)* dereferenceable(4) [[C1GEN]])

TEST(__private)

// CHECK-LABEL: @_Z13test__privatev

// Test the address space of 'this' when invoking a constructor for an object in non-default address space
// CHECK: [[CGEN:%[0-9]+]] = addrspacecast %class.C* %c to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1Ev(%class.C addrspace(4)* [[CGEN]])

// Test the address space of 'this' when invoking a method.
// CHECK: [[CGEN:%[0-9]+]] = addrspacecast %class.C* %c to %class.C addrspace(4)*
// CHECK: %call = call i32 @_ZNU3AS41C3getEv(%class.C addrspace(4)* [[CGEN]])

// Test the address space of 'this' when invoking a copy-constructor.
// CHECK: [[C1GEN:%[0-9]+]] = addrspacecast %class.C* %c1 to %class.C addrspace(4)*
// CHECK: [[CGEN:%[0-9]+]] = addrspacecast %class.C* %c to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1ERU3AS4KS_(%class.C addrspace(4)* [[C1GEN]], %class.C addrspace(4)* dereferenceable(4) [[CGEN]])

// Test the address space of 'this' when invoking a constructor.
// CHECK: [[C2GEN:%[0-9]+]] = addrspacecast %class.C* %c2 to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1Ev(%class.C addrspace(4)* [[C2GEN]])

// Test the address space of 'this' when invoking a copy-assignment.
// CHECK: [[C1GEN:%[0-9]+]] = addrspacecast %class.C* %c1 to %class.C addrspace(4)*
// CHECK: [[C2GEN:%[0-9]+]] = addrspacecast %class.C* %c2 to %class.C addrspace(4)*
// CHECK: %call1 = call dereferenceable(4) %class.C addrspace(4)* @_ZNU3AS41CaSERU3AS4KS_(%class.C addrspace(4)* [[C2GEN]], %class.C addrspace(4)* dereferenceable(4) [[C1GEN]])

TEST()

// CHECK-LABEL: @_Z4testv()

// Test the address space of 'this' when invoking a constructor for an object in non-default address space
// CHECK: [[CGEN:%[0-9]+]] = addrspacecast %class.C* %c to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1Ev(%class.C addrspace(4)* [[CGEN]])

// Test the address space of 'this' when invoking a method.
// CHECK: [[CGEN:%[0-9]+]] = addrspacecast %class.C* %c to %class.C addrspace(4)*
// CHECK: %call = call i32 @_ZNU3AS41C3getEv(%class.C addrspace(4)* [[CGEN]])

// Test the address space of 'this' when invoking a copy-constructor.
// CHECK: [[C1GEN:%[0-9]+]] = addrspacecast %class.C* %c1 to %class.C addrspace(4)*
// CHECK: [[CGEN:%[0-9]+]] = addrspacecast %class.C* %c to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1ERU3AS4KS_(%class.C addrspace(4)* [[C1GEN]], %class.C addrspace(4)* dereferenceable(4) [[CGEN]])

// Test the address space of 'this' when invoking a constructor.
// CHECK: [[C2GEN:%[0-9]+]] = addrspacecast %class.C* %c2 to %class.C addrspace(4)*
// CHECK: call void @_ZNU3AS41CC1Ev(%class.C addrspace(4)* [[C2GEN]])

// Test the address space of 'this' when invoking a copy-assignment.
// CHECK: [[C1GEN:%[0-9]+]] = addrspacecast %class.C* %c1 to %class.C addrspace(4)*
// CHECK: [[C2GEN:%[0-9]+]] = addrspacecast %class.C* %c2 to %class.C addrspace(4)*
// CHECK: %call1 = call dereferenceable(4) %class.C addrspace(4)* @_ZNU3AS41CaSERU3AS4KS_(%class.C addrspace(4)* [[C2GEN]], %class.C addrspace(4)* dereferenceable(4) [[C1GEN]])
