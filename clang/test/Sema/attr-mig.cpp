// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef int kern_return_t;
typedef kern_return_t IOReturn;
#define KERN_SUCCESS 0
#define kIOReturnSuccess KERN_SUCCESS

class MyServer {
public:
  virtual __attribute__((mig_server_routine)) IOReturn externalMethod();
  virtual __attribute__((mig_server_routine)) void anotherMethod(); // expected-warning{{'mig_server_routine' attribute only applies to routines that return a kern_return_t}}
  virtual __attribute__((mig_server_routine)) int yetAnotherMethod(); // expected-warning{{'mig_server_routine' attribute only applies to routines that return a kern_return_t}}
  [[clang::mig_server_routine]] virtual IOReturn cppAnnotatedMethod();
  [[clang::mig_server_routine("arg")]] virtual IOReturn cppAnnotatedMethodWithInvalidArgs(); // expected-error{{'mig_server_routine' attribute takes no arguments}}
  [[clang::mig_server_routine]] virtual int cppInvalidAnnotatedMethod(); // expected-warning{{'mig_server_routine' attribute only applies to routines that return a kern_return_t}}
};

IOReturn MyServer::externalMethod() {
  return kIOReturnSuccess;
}
