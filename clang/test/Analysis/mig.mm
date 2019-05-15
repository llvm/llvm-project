// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,osx.MIG\
// RUN:                       -analyzer-output=text -fblocks -verify %s

typedef unsigned uint32_t;

// XNU APIs.

typedef int kern_return_t;
#define KERN_SUCCESS 0
#define KERN_ERROR 1
#define MIG_NO_REPLY (-305)

typedef unsigned mach_port_name_t;
typedef unsigned vm_address_t;
typedef unsigned vm_size_t;
typedef void *ipc_space_t;
typedef unsigned long io_user_reference_t;
typedef struct ipc_port *ipc_port_t;
typedef unsigned mach_port_t;
typedef uint32_t UInt32;

struct os_refcnt {};
typedef struct os_refcnt os_refcnt_t;

struct thread {
  os_refcnt_t ref_count;
};
typedef struct thread *thread_t;

kern_return_t vm_deallocate(mach_port_name_t, vm_address_t, vm_size_t);
kern_return_t mach_vm_deallocate(mach_port_name_t, vm_address_t, vm_size_t);
void mig_deallocate(vm_address_t, vm_size_t);
kern_return_t mach_port_deallocate(ipc_space_t, mach_port_name_t);
void ipc_port_release(ipc_port_t);
void thread_deallocate(thread_t);

static void os_ref_retain(struct os_refcnt *rc);

#define thread_reference_internal(thread) os_ref_retain(&(thread)->ref_count);

#define MIG_SERVER_ROUTINE __attribute__((mig_server_routine))

// IOKit wrappers.

class OSObject;
typedef kern_return_t IOReturn;
#define kIOReturnError 1

enum {
  kOSAsyncRef64Count = 8,
};

typedef io_user_reference_t OSAsyncReference64[kOSAsyncRef64Count];

struct IOExternalMethodArguments {
  io_user_reference_t *asyncReference;
};

struct IOExternalMethodDispatch {};

class IOUserClient {
public:
  static IOReturn releaseAsyncReference64(OSAsyncReference64);
  static IOReturn releaseNotificationPort(mach_port_t port);

  MIG_SERVER_ROUTINE
  virtual IOReturn externalMethod(
      uint32_t selector, IOExternalMethodArguments *arguments,
      IOExternalMethodDispatch *dispatch = 0, OSObject *target = 0,
      void *reference = 0);

  MIG_SERVER_ROUTINE
  virtual IOReturn registerNotificationPort(mach_port_t, UInt32, UInt32);
};

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

MIG_SERVER_ROUTINE
kern_return_t no_unrelated_notes(mach_port_name_t port, vm_address_t address, vm_size_t size) {
  vm_deallocate(port, address, size); // no-note
  1 / 0; // expected-warning{{Division by zero}}
         // expected-note@-1{{Division by zero}}
  return KERN_SUCCESS;
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

// Test various APIs.
MIG_SERVER_ROUTINE
kern_return_t test_mach_vm_deallocate(mach_port_name_t port, vm_address_t address, vm_size_t size) {
  mach_vm_deallocate(port, address, size); // expected-note{{Value passed through parameter 'address' is deallocated}}
  return KERN_ERROR;                 // expected-warning{{MIG callback fails with error after deallocating argument value}}
                                     // expected-note@-1{{MIG callback fails with error after deallocating argument value}}
}

MIG_SERVER_ROUTINE
kern_return_t test_mach_port_deallocate(ipc_space_t space,
                                        mach_port_name_t port) {
  mach_port_deallocate(space, port); // expected-note{{Value passed through parameter 'port' is deallocated}}
  return KERN_ERROR;                 // expected-warning{{MIG callback fails with error after deallocating argument value}}
                                     // expected-note@-1{{MIG callback fails with error after deallocating argument value}}
}

MIG_SERVER_ROUTINE
kern_return_t test_mig_deallocate(vm_address_t address, vm_size_t size) {
  mig_deallocate(address, size); // expected-note{{Value passed through parameter 'address' is deallocated}}
  return KERN_ERROR;             // expected-warning{{MIG callback fails with error after deallocating argument value}}
                                 // expected-note@-1{{MIG callback fails with error after deallocating argument value}}
}

MIG_SERVER_ROUTINE
kern_return_t test_ipc_port_release(ipc_port_t port) {
  ipc_port_release(port); // expected-note{{Value passed through parameter 'port' is deallocated}}
  return KERN_ERROR; // expected-warning{{MIG callback fails with error after deallocating argument value}}
							       // expected-note@-1{{MIG callback fails with error after deallocating argument value}}
}

// Let's try the C++11 attribute spelling syntax as well.
[[clang::mig_server_routine]]
IOReturn test_releaseAsyncReference64(IOExternalMethodArguments *arguments) {
  IOUserClient::releaseAsyncReference64(arguments->asyncReference); // expected-note{{Value passed through parameter 'arguments' is deallocated}}
  return kIOReturnError;                                            // expected-warning{{MIG callback fails with error after deallocating argument value}}
                                                                    // expected-note@-1{{MIG callback fails with error after deallocating argument value}}
}

MIG_SERVER_ROUTINE
kern_return_t test_no_reply(ipc_space_t space, mach_port_name_t port) {
  mach_port_deallocate(space, port);
  return MIG_NO_REPLY; // no-warning
}

class MyClient: public IOUserClient {
  // The MIG_SERVER_ROUTINE annotation is intentionally skipped.
  // It should be picked up from the superclass.
  IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *arguments,
                          IOExternalMethodDispatch *dispatch = 0, OSObject *target = 0, void *reference = 0) override {

    releaseAsyncReference64(arguments->asyncReference); // expected-note{{Value passed through parameter 'arguments' is deallocated}}
    return kIOReturnError;                              // expected-warning{{MIG callback fails with error after deallocating argument value}}
                                                        // expected-note@-1{{MIG callback fails with error after deallocating argument value}}
  }

  IOReturn registerNotificationPort(mach_port_t port, UInt32 x, UInt32 y) {
    releaseNotificationPort(port); // expected-note{{Value passed through parameter 'port' is deallocated}}
    return kIOReturnError; // expected-warning{{MIG callback fails with error after deallocating argument value}}
                           // expected-note@-1{{MIG callback fails with error after deallocating argument value}}
  }
};

MIG_SERVER_ROUTINE
kern_return_t test_os_ref_retain(thread_t thread) {
  thread_reference_internal(thread);
  thread_deallocate(thread);
  return KERN_ERROR; // no-warning
}
