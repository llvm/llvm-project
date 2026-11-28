// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'
// RUN: %clangxx_tsan %s -o %t.arc -fobjc-arc -framework Foundation
// RUN: %run %t.arc 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

// UNSUPPORTED: ios

#import <Foundation/Foundation.h>
#import <xpc/xpc.h>

#import "../test.h"

long global;

int main(int argc, const char *argv[]) {
  barrier_init(&barrier, 2);
  @autoreleasepool {
    NSLog(@"Hello world.");

    dispatch_queue_t server_q = dispatch_queue_create("server.queue", DISPATCH_QUEUE_CONCURRENT);
    dispatch_queue_t client_q = dispatch_queue_create("client.queue", DISPATCH_QUEUE_CONCURRENT);

    xpc_connection_t server_conn = xpc_connection_create(NULL, server_q);
    xpc_endpoint_t endpoint = xpc_endpoint_create(server_conn);
    xpc_connection_t client_conn =
        xpc_connection_create_from_endpoint(endpoint);

    xpc_object_t msg = xpc_dictionary_create(NULL, NULL, 0);
    xpc_dictionary_set_string(msg, "hello", "world");
    NSLog(@"sending message: %@", msg);

    global = 42;

    NSString *server_captured =
        [[NSMutableString alloc] initWithString:@"server_val"];
    xpc_connection_set_event_handler(server_conn, ^(xpc_object_t client) {
      NSLog(@"global = %ld, captured = %@", global, server_captured);
      NSLog(@"server event handler, client = %@", client);

      if (client == XPC_ERROR_CONNECTION_INTERRUPTED || client == XPC_ERROR_CONNECTION_INVALID) {
        return;
      }
      NSString *peer_captured =
          [[NSMutableString alloc] initWithString:@"peer_val"];
      xpc_connection_set_event_handler(client, ^(xpc_object_t object) {
        NSLog(@"received message: %@, captured = %@", object, peer_captured);

        xpc_object_t reply = xpc_dictionary_create_reply(object);
        if (!reply)
          return;
        xpc_dictionary_set_string(reply, "reply", "value");

        xpc_connection_t remote = xpc_dictionary_get_remote_connection(object);
        NSString *barrier_captured =
            [[NSMutableString alloc] initWithString:@"barrier_val"];
        xpc_connection_send_barrier(remote, ^{
          NSLog(@"barrier, captured = %@", barrier_captured);
          xpc_connection_send_message(remote, reply);
        });
      });

      xpc_connection_resume(client);
    });

    NSString *client_captured =
        [[NSMutableString alloc] initWithString:@"client_val"];
    xpc_connection_set_event_handler(client_conn, ^(xpc_object_t event) {
      NSLog(@"client event handler, event = %@, captured = %@", event, client_captured);
      barrier_wait(&barrier);
    });

    NSString *reply_captured =
        [[NSMutableString alloc] initWithString:@"reply_val"];
    xpc_connection_send_message_with_reply(
        client_conn, msg, client_q, ^(xpc_object_t object) {
          NSLog(@"received reply: %@, captured = %@", object, reply_captured);

          xpc_connection_cancel(client_conn);
          xpc_connection_cancel(server_conn);
        });
    xpc_connection_resume(server_conn);
    xpc_connection_resume(client_conn);
    barrier_wait(&barrier);

    NSLog(@"Done.");
  }
  return 0;
}

// CHECK: Hello world.
// CHECK: global = 42, captured = server_val
// CHECK: captured = peer_val
// CHECK: barrier, captured = barrier_val
// CHECK: captured = reply_val
// CHECK: captured = client_val
// CHECK: Done.
