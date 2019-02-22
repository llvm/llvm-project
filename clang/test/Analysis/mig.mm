// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,alpha.osx.MIG\
// RUN:                       -analyzer-output=text -fblocks -verify %s

// XNU APIs.

typedef int kern_return_t;
#define KERN_SUCCESS 0
#define KERN_ERROR 1

typedef unsigned mach_port_name_t;
typedef unsigned vm_address_t;
typedef unsigned vm_size_t;

kern_return_t vm_deallocate(mach_port_name_t, vm_address_t, vm_size_t);

#define MIG_SERVER_ROUTINE __attribute__((mig_server_routine))


// Tests.

MIG_SERVER_ROUTINE
kern_return_t basic_test(mach_port_name_t port, vm_address_t address, vm_size_t size) {
  vm_deallocate(port, address, size); // expected-note{{Value passed through parameter 'address' is deallocated}}
  if (size > 10) { // expected-note{{Assuming 'size' is > 10}}
                   // expected-note@-1{{Taking true branch}}
    return KERN_ERROR; // expected-warning{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
											 // expected-note@-1{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
  }
  return KERN_SUCCESS;
}

MIG_SERVER_ROUTINE
kern_return_t test_unknown_return_value(mach_port_name_t port, vm_address_t address, vm_size_t size) {
  extern kern_return_t foo();

  vm_deallocate(port, address, size);
  // We don't know if it's a success or a failure.
  return foo(); // no-warning
}

// Make sure we don't crash when they forgot to write the return statement.
MIG_SERVER_ROUTINE
kern_return_t no_crash(mach_port_name_t port, vm_address_t address, vm_size_t size) {
  vm_deallocate(port, address, size);
}

// When releasing two parameters, add a note for both of them.
// Also when returning a variable, explain why do we think that it contains
// a non-success code.
MIG_SERVER_ROUTINE
kern_return_t release_twice(mach_port_name_t port, vm_address_t addr1, vm_address_t addr2, vm_size_t size) {
  kern_return_t ret = KERN_ERROR; // expected-note{{'ret' initialized to 1}}
  vm_deallocate(port, addr1, size); // expected-note{{Value passed through parameter 'addr1' is deallocated}}
  vm_deallocate(port, addr2, size); // expected-note{{Value passed through parameter 'addr2' is deallocated}}
  return ret; // expected-warning{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
                     // expected-note@-1{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
}

// Make sure we find the bug when the object is destroyed within an
// automatic destructor.
MIG_SERVER_ROUTINE
kern_return_t test_vm_deallocate_in_automatic_dtor(mach_port_name_t port, vm_address_t address, vm_size_t size) {
  struct WillDeallocate {
    mach_port_name_t port;
    vm_address_t address;
    vm_size_t size;
    ~WillDeallocate() {
      vm_deallocate(port, address, size); // expected-note{{Value passed through parameter 'address' is deallocated}}
    }
  } will_deallocate{port, address, size};

 if (size > 10) {
    // expected-note@-1{{Assuming 'size' is > 10}}
    // expected-note@-2{{Taking true branch}}
    return KERN_ERROR;
    // expected-note@-1{{Calling '~WillDeallocate'}}
    // expected-note@-2{{Returning from '~WillDeallocate'}}
    // expected-warning@-3{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
    // expected-note@-4   {{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
  }
  return KERN_SUCCESS;
}

// Check that we work on Objective-C messages and blocks.
@interface I
- (kern_return_t)fooAtPort:(mach_port_name_t)port withAddress:(vm_address_t)address ofSize:(vm_size_t)size;
@end

@implementation I
- (kern_return_t)fooAtPort:(mach_port_name_t)port
               withAddress:(vm_address_t)address
                    ofSize:(vm_size_t)size MIG_SERVER_ROUTINE {
  vm_deallocate(port, address, size); // expected-note{{Value passed through parameter 'address' is deallocated}}
  return KERN_ERROR; // expected-warning{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
                     // expected-note@-1{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
}
@end

void test_block() {
  kern_return_t (^block)(mach_port_name_t, vm_address_t, vm_size_t) =
      ^MIG_SERVER_ROUTINE (mach_port_name_t port,
                           vm_address_t address, vm_size_t size) {
        vm_deallocate(port, address, size); // expected-note{{Value passed through parameter 'address' is deallocated}}
        return KERN_ERROR; // expected-warning{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
                           // expected-note@-1{{MIG callback fails with error after deallocating argument value. This is a use-after-free vulnerability because the caller will try to deallocate it again}}
      };
}

void test_block_with_weird_return_type() {
  struct Empty {};

  // The block is written within a function so that it was actually analyzed as
  // a top-level function during analysis. If we were to write it as a global
  // variable of block type instead, it would not have been analyzed, because
  // ASTConsumer won't find the block's code body within the VarDecl.
  // At the same time, we shouldn't call it from the function, because otherwise
  // it will be analyzed as an inlined function rather than as a top-level
  // function.
  Empty (^block)(mach_port_name_t, vm_address_t, vm_size_t) =
      ^MIG_SERVER_ROUTINE(mach_port_name_t port,
                          vm_address_t address, vm_size_t size) {
        vm_deallocate(port, address, size);
        return Empty{}; // no-crash
      };
}
