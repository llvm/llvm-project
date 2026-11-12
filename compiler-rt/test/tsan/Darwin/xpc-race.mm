// RUN: %clangxx_tsan %s -o %t -framework Foundation
// RUN: %deflake %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: ios

#import <Foundation/Foundation.h>
#import <xpc/xpc.h>
#import <stdatomic.h>

#import "../test.h"

long global;

_Atomic(long) msg_counter;
_Atomic(long) processed_msgs;
xpc_connection_t server_conn;
xpc_connection_t client_conns[2];

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    fprintf(stderr, "Hello world.\n");
    // CHECK: Hello world.

    barrier_init(&barrier, 2);

    dispatch_queue_t server_q = dispatch_queue_create("server.queue", DISPATCH_QUEUE_CONCURRENT);

    server_conn = xpc_connection_create(NULL, server_q);

    xpc_connection_set_event_handler(server_conn, ^(xpc_object_t client) {
      fprintf(stderr, "server event handler, client = %p\n", client);

      if (client == XPC_ERROR_CONNECTION_INTERRUPTED || client == XPC_ERROR_CONNECTION_INVALID) {
        return;
      }
      xpc_connection_set_event_handler(client, ^(xpc_object_t object) {
        fprintf(stderr, "received message: %p\n", object);

        long msg_number = atomic_fetch_add_explicit(&msg_counter, 1, memory_order_relaxed);

        if (msg_number == 0)
          barrier_wait(&barrier);

        global++;
        // CHECK: WARNING: ThreadSanitizer: data race
        // CHECK:   Write of size 8
        // CHECK:     #0 {{.*}}xpc-race.mm:[[@LINE-3]]
        // CHECK:   Previous write of size 8
        // CHECK:     #0 {{.*}}xpc-race.mm:[[@LINE-5]]
        // CHECK: Location is global 'global'

        if (msg_number == 1)
          barrier_wait(&barrier);

        atomic_fetch_add(&processed_msgs, 1);

        dispatch_sync(dispatch_get_main_queue(), ^{
          if (processed_msgs >= 2) {
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
        fprintf(stderr, "client event handler, event = %p\n", event);
      });

      xpc_object_t msg = xpc_dictionary_create(NULL, NULL, 0);
      xpc_dictionary_set_string(msg, "hello", "world");
      fprintf(stderr, "sending message: %p\n", msg);

      xpc_connection_send_message(client_conns[i], msg);
      xpc_connection_resume(client_conns[i]);
    }

    CFRunLoopRun();

    fprintf(stderr, "Done.\n");
    // CHECK: Done.
  }
  return 0;
}
