// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_tsan %s -o %t.arc -fobjc-arc -framework Foundation
// RUN: %run %t.arc 2>&1 | FileCheck %s

// UNSUPPORTED: ios

#import <Foundation/Foundation.h>
#import <xpc/xpc.h>

long global;

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    NSLog(@"Hello world.");

    dispatch_queue_t server_q = dispatch_queue_create("server.queue", DISPATCH_QUEUE_CONCURRENT);
    dispatch_queue_t client_q = dispatch_queue_create("client.queue", DISPATCH_QUEUE_CONCURRENT);

    xpc_connection_t server_conn = xpc_connection_create(NULL, server_q);
    NSObject *server_captured = [[NSObject alloc] init];

    global = 42;

    xpc_connection_set_event_handler(server_conn, ^(xpc_object_t client) {
      NSLog(@"global = %ld, captured = %@", global, server_captured);
      NSLog(@"server event handler, client = %@", client);

      if (client == XPC_ERROR_CONNECTION_INTERRUPTED || client == XPC_ERROR_CONNECTION_INVALID) {
        return;
      }
      NSObject *peer_captured = [[NSObject alloc] init];
      xpc_connection_set_event_handler(client, ^(xpc_object_t object) {
        NSLog(@"received message: %@, captured = %@", object, peer_captured);

        xpc_object_t reply = xpc_dictionary_create_reply(object);
        if (!reply)
          return;
        xpc_dictionary_set_string(reply, "reply", "value");

        xpc_connection_t remote = xpc_dictionary_get_remote_connection(object);
        xpc_connection_send_message(remote, reply);
        NSObject *barrier_captured = [[NSObject alloc] init];
        xpc_connection_send_barrier(remote, ^{
          NSLog(@"barrier, captured = %@", barrier_captured);
        });
      });

      xpc_connection_resume(client);
    });
    xpc_connection_resume(server_conn);
    xpc_endpoint_t endpoint = xpc_endpoint_create(server_conn);

    xpc_connection_t client_conn = xpc_connection_create_from_endpoint(endpoint);
    NSObject *client_captured = [[NSObject alloc] init];
    xpc_connection_set_event_handler(client_conn, ^(xpc_object_t event) {
      NSLog(@"client event handler, event = %@, captured = %@", event, client_captured);
    });

    xpc_object_t msg = xpc_dictionary_create(NULL, NULL, 0);
    xpc_dictionary_set_string(msg, "hello", "world");
    NSLog(@"sending message: %@", msg);

    NSObject *reply_captured = [[NSObject alloc] init];
    xpc_connection_send_message_with_reply(
        client_conn, msg, client_q, ^(xpc_object_t object) {
          NSLog(@"received reply: %@, captured = %@", object, reply_captured);

          xpc_connection_cancel(client_conn);
          xpc_connection_cancel(server_conn);

          dispatch_sync(dispatch_get_main_queue(), ^{
            CFRunLoopStop(CFRunLoopGetCurrent());
          });
        });
    xpc_connection_resume(client_conn);

    CFRunLoopRun();

    NSLog(@"Done.");
  }
  return 0;
}

// CHECK: Done.
// CHECK-NOT: WARNING: ThreadSanitizer
