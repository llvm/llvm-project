// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#import <xpc/xpc.h>

long global;

int main(int argc, const char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  dispatch_queue_t server_q = dispatch_queue_create("server.queue", DISPATCH_QUEUE_CONCURRENT);
  xpc_connection_t server_conn = xpc_connection_create(NULL, server_q);

  xpc_connection_set_event_handler(server_conn, ^(xpc_object_t client) {
    if (client == XPC_ERROR_CONNECTION_INTERRUPTED || client == XPC_ERROR_CONNECTION_INVALID) {
      global = 43;
      
      dispatch_async(dispatch_get_main_queue(), ^{
        CFRunLoopStop(CFRunLoopGetCurrent());
      });
    }
  });
  xpc_connection_resume(server_conn);
  
  global = 42;
  
  xpc_connection_cancel(server_conn);
  
  CFRunLoopRun();

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
