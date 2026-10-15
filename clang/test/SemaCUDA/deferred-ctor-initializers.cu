// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -std=c++17 -fsyntax-only -verify=host %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -std=c++17 -fsyntax-only -verify=host %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -x cuda -std=c++17 -fsyntax-only -verify=device %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -x hip -std=c++17 -fsyntax-only -verify=device %s

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__host__ int host_function(); // device-note 4 {{'host_function' declared here}}

struct Base {
  __host__ Base(int) {} // device-note {{'Base' declared here}}
  __host__ __device__ Base(int, int) {
    host_function(); // device-error {{reference to __host__ function 'host_function' in __host__ __device__ function}}
  }
  __host__ __device__ Base(int, int, int) {}
};

// Reachability must follow implicit member initialization into base initializers.
struct CallsHostConstructor : Base {
  __host__ __device__ CallsHostConstructor() : Base(1) {} // device-error {{reference to __host__ function 'Base' in __host__ __device__ function}}
};
struct CallsBadConstructor : Base {
  __host__ __device__ CallsBadConstructor() : Base(1, 2) {} // device-note {{called by 'CallsBadConstructor'}}
};
struct CallsHostInArgument : Base {
  __host__ __device__ CallsHostInArgument() : Base(1, 2, host_function()) {} // device-error {{reference to __host__ function 'host_function' in __host__ __device__ function}}
};
struct ImplicitMembers { // device-note 3 {{called by 'ImplicitMembers'}}
  CallsHostConstructor first;
  CallsBadConstructor second;
  CallsHostInArgument third;
};
__global__ void implicit_members() { ImplicitMembers members; } // device-note 3 {{which is called by 'implicit_members'}}

// Use initializers from the definition, even if the construct expression refers
// to an earlier declaration.
struct OutOfLineBase {
  __host__ __device__ OutOfLineBase() {
    host_function(); // device-error {{reference to __host__ function 'host_function' in __host__ __device__ function}}
  }
};
struct OutOfLine : OutOfLineBase {
  __host__ __device__ OutOfLine();
};
inline __host__ __device__ OutOfLine::OutOfLine() : OutOfLineBase() {} // device-note {{called by 'OutOfLine'}}
__global__ void out_of_line() { OutOfLine value; } // device-note {{which is called by 'out_of_line'}}

// Delegation and template instantiation also reach constructor initializers.
template <class T> struct Delegating {
  __host__ __device__ Delegating() : Delegating(0) {} // device-note {{called by 'Delegating'}}
  __host__ __device__ Delegating(T) {
    host_function(); // device-error {{reference to __host__ function 'host_function' in __host__ __device__ function}}
  }
};
__global__ void delegating() { Delegating<int> value; } // device-note {{which is called by 'delegating'}}

// Deferred errors in constructors which are not used on the device stay silent.
struct Unused : Base {
  __host__ __device__ Unused() : Base(1, 2, host_function()) {}
};
__host__ void host_only() { Unused unused; }
__global__ void discarded() {
  if constexpr (false) {
    Unused unused;
  }
}

namespace gh118415 {
__host__ int host_func(); // device-note 2 {{'host_func' declared here}}

struct A {
  int x;
  __host__ A(int) { x = host_func(); }; // Must never be called on the device.
  // device-note@-1 {{'A' declared here}}
  __host__ __device__ A(int, int) { x = host_func(); }; // host_func() should trigger deferred diags.
  // device-error@-1 {{reference to __host__ function 'host_func' in __host__ __device__ function}}
  __host__ __device__ A(int a, int b, int c) { x = a + b + c; }; // OK to call.
};
struct B : A {
  __host__ __device__ B()
      : A(1) {} // A(1) should trigger deferred diag.
  // device-error@-1 {{reference to __host__ function 'A' in __host__ __device__ function}}
};

struct B1 : A {
  __host__ __device__ B1() : A(1, 2) {} // device-note {{called by 'B1'}}
};

struct B2 : A {
  __host__ __device__ B2()
      : A(1, 2, host_func()) {} // host_func() call should trigger deferred diag.
  // device-error@-1 {{reference to __host__ function 'host_func' in __host__ __device__ function}}
};

struct C { // device-note 3 {{called by 'C'}}
  B b;
  B1 b1;
  B2 b2;
};

__attribute__((global)) void kernel() {
  C c; // device-note 3 {{which is called by 'kernel'}}
}
} // namespace gh118415

namespace virtual_bases {
__host__ int host_function(); // device-note 3 {{'host_function' declared here}}

struct Base {
  __host__ Base(int) {} // device-note {{'Base' declared here}}
  __host__ __device__ Base(int, int) {
    host_function(); // device-error {{reference to __host__ function 'host_function' in __host__ __device__ function}}
  }
  __host__ __device__ Base(int, int, int) {}
};
struct CallsHostConstructor : virtual Base {
  __host__ __device__ CallsHostConstructor() : Base(1) {} // device-error {{reference to __host__ function 'Base' in __host__ __device__ function}}
};
struct CallsBadConstructor : virtual Base {
  __host__ __device__ CallsBadConstructor() : Base(1, 2) {} // device-note {{called by 'CallsBadConstructor'}}
};
struct CallsHostInArgument : virtual Base {
  __host__ __device__ CallsHostInArgument() : Base(1, 2, host_function()) {} // device-error {{reference to __host__ function 'host_function' in __host__ __device__ function}}
};

// The most-derived constructor implicitly initializes an indirect virtual base.
struct ImplicitBase {
  __host__ __device__ ImplicitBase() {
    host_function(); // device-error {{reference to __host__ function 'host_function' in __host__ __device__ function}}
  }
  __host__ __device__ ImplicitBase(int) {}
};
struct Left : virtual ImplicitBase {
  __host__ __device__ Left() : ImplicitBase(1) {}
};
struct Right : virtual ImplicitBase {
  __host__ __device__ Right() : ImplicitBase(2) {}
};
struct Diamond : Left, Right {
  __host__ __device__ Diamond() {} // device-note {{called by 'Diamond'}}
};
// Explicitly initializing the shared virtual base with a valid constructor works.
struct ValidDiamond : Left, Right {
  __host__ __device__ ValidDiamond() : ImplicitBase(3) {}
};

__global__ void kernel() {
  CallsHostConstructor first; // device-note {{called by 'kernel'}}
  CallsBadConstructor second; // device-note {{which is called by 'kernel'}}
  CallsHostInArgument third;  // device-note {{called by 'kernel'}}
  Diamond fourth;             // device-note {{which is called by 'kernel'}}
  ValidDiamond valid;
}
} // namespace virtual_bases

// Host reachability must likewise diagnose device-only calls in initializers.
namespace host_calls {
__device__ int device_function(); // host-note 6 {{'device_function' declared here}}

struct Base {
  __device__ Base(int) {} // host-note 2 {{'Base' declared here}}
  __host__ __device__ Base(int, int) {
    device_function(); // host-error {{reference to __device__ function 'device_function' in __host__ __device__ function}}
  }
  __host__ __device__ Base(int, int, int) {}
};
struct CallsDeviceConstructor : Base {
  __host__ __device__ CallsDeviceConstructor() : Base(1) {} // host-error {{reference to __device__ function 'Base' in __host__ __device__ function}}
};
struct CallsBadConstructor : Base {
  __host__ __device__ CallsBadConstructor() : Base(1, 2) {} // host-note {{called by 'CallsBadConstructor'}}
};
struct CallsDeviceInArgument : Base {
  __host__ __device__ CallsDeviceInArgument() : Base(1, 2, device_function()) {} // host-error {{reference to __device__ function 'device_function' in __host__ __device__ function}}
};
struct ImplicitMembers { // host-note 3 {{called by 'ImplicitMembers'}}
  CallsDeviceConstructor first;
  CallsBadConstructor second;
  CallsDeviceInArgument third;
};
__host__ void implicit_members() { ImplicitMembers members; } // host-note 3 {{which is called by 'implicit_members'}}

struct OutOfLineBase {
  __host__ __device__ OutOfLineBase() {
    device_function(); // host-error {{reference to __device__ function 'device_function' in __host__ __device__ function}}
  }
};
struct OutOfLine : OutOfLineBase {
  __host__ __device__ OutOfLine();
};
inline __host__ __device__ OutOfLine::OutOfLine() : OutOfLineBase() {} // host-note {{called by 'OutOfLine'}}
__host__ void out_of_line() { OutOfLine value; }                       // host-note {{which is called by 'out_of_line'}}

template <class T>
struct Delegating {
  __host__ __device__ Delegating() : Delegating(0) {} // host-note {{called by 'Delegating'}}
  __host__ __device__ Delegating(T) {
    device_function(); // host-error {{reference to __device__ function 'device_function' in __host__ __device__ function}}
  }
};
__host__ void delegating() { Delegating<int> value; } // host-note {{which is called by 'delegating'}}

// Device-only uses and discarded host statements must not trigger host errors.
struct Unused : Base {
  __host__ __device__ Unused() : Base(1, 2, device_function()) {}
};
__device__ void device_only() { Unused unused; }
__host__ void discarded() {
  if constexpr (false) {
    Unused unused;
  }
}

namespace virtual_bases {
struct CallsDeviceConstructor : virtual Base {
  __host__ __device__ CallsDeviceConstructor() : Base(1) {} // host-error {{reference to __device__ function 'Base' in __host__ __device__ function}}
};
struct CallsBadConstructor : virtual Base {
  __host__ __device__ CallsBadConstructor() : Base(1, 2) {} // host-note {{called by 'CallsBadConstructor'}}
};
struct CallsDeviceInArgument : virtual Base {
  __host__ __device__ CallsDeviceInArgument() : Base(1, 2, device_function()) {} // host-error {{reference to __device__ function 'device_function' in __host__ __device__ function}}
};
struct ImplicitBase {
  __host__ __device__ ImplicitBase() {
    device_function(); // host-error {{reference to __device__ function 'device_function' in __host__ __device__ function}}
  }
  __host__ __device__ ImplicitBase(int) {}
};
struct Left : virtual ImplicitBase {
  __host__ __device__ Left() : ImplicitBase(1) {}
};
struct Right : virtual ImplicitBase {
  __host__ __device__ Right() : ImplicitBase(2) {}
};
struct Diamond : Left, Right {
  __host__ __device__ Diamond() {} // host-note {{called by 'Diamond'}}
};
struct ValidDiamond : Left, Right {
  __host__ __device__ ValidDiamond() : ImplicitBase(3) {}
};
__host__ void host_function() {
  CallsDeviceConstructor first; // host-note {{called by 'host_function'}}
  CallsBadConstructor second;   // host-note {{which is called by 'host_function'}}
  CallsDeviceInArgument third;  // host-note {{called by 'host_function'}}
  Diamond fourth;               // host-note {{which is called by 'host_function'}}
  ValidDiamond valid;
}
} // namespace virtual_bases
} // namespace host_calls

namespace inherited_constructors {
__host__ int host_function(); // device-note 3 {{'host_function' declared here}}
struct Base {
  __host__ __device__ Base(int) {
    host_function(); // device-error {{reference to __host__ function 'host_function' in __host__ __device__ function}}
  }
};
struct Derived : Base {
  using Base::Base; // device-note {{called by 'Base'}}
};
__global__ void kernel() { Derived value(1); } // device-note {{which is called by 'kernel'}}

template <class T>
struct TemplateBase {
  __host__ __device__ TemplateBase(T) {
    host_function(); // device-error 2 {{reference to __host__ function 'host_function' in __host__ __device__ function}}
  }
};
template <class T>
struct TemplateDerived : TemplateBase<T> {
  using TemplateBase<T>::TemplateBase; // device-note {{called by 'TemplateBase'}}
};
struct Indirect : TemplateDerived<int> {
  using TemplateDerived<int>::TemplateDerived; // device-note {{which is called by 'TemplateBase'}}
};
__global__ void indirect() { Indirect value(1); } // device-note {{which is called by 'indirect'}}

struct VirtualDerived : virtual TemplateBase<long> {
  using TemplateBase<long>::TemplateBase; // device-note {{called by 'TemplateBase'}}
};
__global__ void virtual_base() { VirtualDerived value(1L); } // device-note {{which is called by 'virtual_base'}}

__host__ void host_only() { TemplateDerived<char> value('x'); }
__global__ void discarded() {
  if constexpr (false) {
    TemplateDerived<short> value(1);
  }
}
} // namespace inherited_constructors

namespace host_inherited_constructors {
__device__ int device_function(); // host-note 3 {{'device_function' declared here}}
struct Base {
  __host__ __device__ Base(int) {
    device_function(); // host-error {{reference to __device__ function 'device_function' in __host__ __device__ function}}
  }
};
struct Derived : Base {
  using Base::Base; // host-note {{called by 'Base'}}
};
__host__ void host_function() { Derived value(1); } // host-note {{which is called by 'host_function'}}

template <class T>
struct TemplateBase {
  __host__ __device__ TemplateBase(T) {
    device_function(); // host-error 2 {{reference to __device__ function 'device_function' in __host__ __device__ function}}
  }
};
template <class T>
struct TemplateDerived : TemplateBase<T> {
  using TemplateBase<T>::TemplateBase; // host-note {{called by 'TemplateBase'}}
};
struct Indirect : TemplateDerived<int> {
  using TemplateDerived<int>::TemplateDerived; // host-note {{which is called by 'TemplateBase'}}
};
__host__ void indirect() { Indirect value(1); } // host-note {{which is called by 'indirect'}}

struct VirtualDerived : virtual TemplateBase<long> {
  using TemplateBase<long>::TemplateBase; // host-note {{called by 'TemplateBase'}}
};
__host__ void virtual_base() { VirtualDerived value(1L); } // host-note {{which is called by 'virtual_base'}}

__device__ void device_only() { TemplateDerived<char> value('x'); }
__host__ void discarded() {
  if constexpr (false) {
    TemplateDerived<short> value(1);
  }
}
} // namespace host_inherited_constructors
