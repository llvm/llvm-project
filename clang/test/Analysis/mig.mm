// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,alpha.osx.MIG -verify %s

// XNU APIs.

typedef int kern_return_t;
#define KERN_SUCCESS 0
#define KERN_ERROR 1

typedef unsigned mach_port_name_t;
typedef unsigned vm_address_t;
typedef unsigned vm_size_t;

kern_return_t vm_deallocate(mach_port_name_t, vm_address_t, vm_size_t);

// Tests.

kern_return_t basic_test(mach_port_name_t port, vm_address_t address, vm_size_t size) {
  vm_deallocate(port, address, size);
  if (size > 10) {
    return KERN_ERROR; // expected-warning{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
  }
  return KERN_SUCCESS;
}

kern_return_t test_unknown_return_value(mach_port_name_t port, vm_address_t address, vm_size_t size) {
  extern kern_return_t foo();

  vm_deallocate(port, address, size);
  // We don't know if it's a success or a failure.
  return foo(); // no-warning
}

// Make sure we don't crash when they forgot to write the return statement.
kern_return_t no_crash(mach_port_name_t port, vm_address_t address, vm_size_t size) {
  vm_deallocate(port, address, size);
}
