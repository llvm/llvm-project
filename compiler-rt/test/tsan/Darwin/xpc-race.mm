// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %env_tsan_opts=ignore_interceptors_accesses=1 %deflake %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>

#import "../test.h"

long global;

long received_msgs;
xpc_connection_t server_conn;
xpc_connection_t client_conns[2];

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    NSLog(@"Hello world.");
    barrier_init(&barrier, 2);

    dispatch_queue_t server_q = dispatch_queue_create("server.queue", DISPATCH_QUEUE_CONCURRENT);

    server_conn = xpc_connection_create(NULL, server_q);

    xpc_connection_set_event_handler(server_conn, ^(xpc_object_t client) {
      NSLog(@"server event handler, client = %@", client);

      if (client == XPC_ERROR_CONNECTION_INTERRUPTED || client == XPC_ERROR_CONNECTION_INVALID) {
        return;
      }
      xpc_connection_set_event_handler(client, ^(xpc_object_t object) {
        NSLog(@"received message: %@", object);

        barrier_wait(&barrier);
        global = 42;

        dispatch_sync(dispatch_get_main_queue(), ^{
          received_msgs++;

          if (received_msgs >= 2) {
            xpc_connection_cancel(client_conns[0]);
            xpc_connection_cancel(client_conns[1]);
            xpc_connection_cancel(server_conn);
            CFRunLoopStop(CFRunLoopGetCurrent());
          }
        });
      });

      xpc_connection_resume(client);
    });
    xpc_connection_resume(server_conn);
    xpc_endpoint_t endpoint = xpc_endpoint_create(server_conn);

    for (int i = 0; i < 2; i++) {
      client_conns[i] = xpc_connection_create_from_endpoint(endpoint);
      xpc_connection_set_event_handler(client_conns[i], ^(xpc_object_t event) {
        NSLog(@"client event handler, event = %@", event);
      });

      xpc_object_t msg = xpc_dictionary_create(NULL, NULL, 0);
      xpc_dictionary_set_string(msg, "hello", "world");
      NSLog(@"sending message: %@", msg);

      xpc_connection_send_message(client_conns[i], msg);
      xpc_connection_resume(client_conns[i]);
    }

    CFRunLoopRun();

    NSLog(@"Done.");
  }
  return 0;
}

// CHECK: Hello world.
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8
// CHECK:     #0 {{.*}}xpc-race.mm:33
// CHECK:   Previous write of size 8
// CHECK:     #0 {{.*}}xpc-race.mm:33
// CHECK: Location is global 'global'
// CHECK: Done.
