// RUN: %clangxx_tsan %s -o %t -framework Foundation -std=c++11
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import <iostream>
#import <thread>

long my_global;
std::once_flag once_token;

void thread_func() {
  std::call_once(once_token, [] {
    my_global = 17;
  });

  long val = my_global;
  fprintf(stderr, "my_global = %ld\n", val);
}

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  std::thread t1(thread_func);
  std::thread t2(thread_func);
  t1.join();
  t2.join();

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
