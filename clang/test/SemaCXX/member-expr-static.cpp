// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
typedef void (*thread_continue_t)();

extern "C" {
  extern void kernel_thread_start(thread_continue_t continuation);
  extern void pure_c(void);
}

class _IOConfigThread {
public:
  static void main( void );
};


void foo( void ) {
  kernel_thread_start(&_IOConfigThread::main);
  kernel_thread_start((thread_continue_t)&_IOConfigThread::main);
  kernel_thread_start(&pure_c);
}
