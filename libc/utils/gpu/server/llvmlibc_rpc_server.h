//===-- Shared memory RPC server instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_SERVER_RPC_SERVER_H
#define LLVM_LIBC_UTILS_GPU_SERVER_RPC_SERVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The maximum number of ports that can be opened for any server.
const uint64_t RPC_MAXIMUM_PORT_COUNT = 4096;

/// The symbol name associated with the client for use with the LLVM C library
/// implementation.
const char *const rpc_client_symbol_name = "__llvm_libc_rpc_client";

/// status codes.
typedef enum {
  RPC_STATUS_SUCCESS = 0x0,
  RPC_STATUS_CONTINUE = 0x1,
  RPC_STATUS_ERROR = 0x1000,
  RPC_STATUS_UNHANDLED_OPCODE = 0x1001,
  RPC_STATUS_INVALID_LANE_SIZE = 0x1002,
} rpc_status_t;

/// A struct containing an opaque handle to an RPC port. This is what allows the
/// server to communicate with the client.
typedef struct rpc_port_s {
  uint64_t handle;
  uint32_t lane_size;
} rpc_port_t;

/// A fixed-size buffer containing the payload sent from the client.
typedef struct rpc_buffer_s {
  uint64_t data[8];
} rpc_buffer_t;

/// An opaque handle to an RPC server that can be attached to a device.
typedef struct rpc_device_s {
  uintptr_t handle;
} rpc_device_t;

/// A function used to allocate \p bytes for use by the RPC server and client.
/// The memory should support asynchronous and atomic access from both the
/// client and server.
typedef void *(*rpc_alloc_ty)(uint64_t size, void *data);

/// A function used to free the \p ptr previously allocated.
typedef void (*rpc_free_ty)(void *ptr, void *data);

/// A callback function provided with a \p port to communicate with the RPC
/// client. This will be called by the server to handle an opcode.
typedef void (*rpc_opcode_callback_ty)(rpc_port_t port, void *data);

/// A callback function to use the port to receive or send a \p buffer.
typedef void (*rpc_port_callback_ty)(rpc_buffer_t *buffer, void *data);

/// Initialize the server for a given device and return it in \p device.
rpc_status_t rpc_server_init(rpc_device_t *rpc_device, uint64_t num_ports,
                             uint32_t lane_size, rpc_alloc_ty alloc,
                             void *data);

/// Shut down the server for a given device.
rpc_status_t rpc_server_shutdown(rpc_device_t rpc_device, rpc_free_ty dealloc,
                                 void *data);

/// Queries the RPC clients at least once and performs server-side work if there
/// are any active requests. Runs until all work on the server is completed.
rpc_status_t rpc_handle_server(rpc_device_t rpc_device);

/// Register a callback to handle an opcode from the RPC client. The associated
/// data must remain accessible as long as the user intends to handle the server
/// with this callback.
rpc_status_t rpc_register_callback(rpc_device_t rpc_device, uint16_t opcode,
                                   rpc_opcode_callback_ty callback, void *data);

/// Obtain a pointer to a local client buffer that can be copied directly to the
/// other process using the address stored at the rpc client symbol name.
const void *rpc_get_client_buffer(rpc_device_t device);

/// Returns the size of the client in bytes to be used for a memory copy.
uint64_t rpc_get_client_size();

/// Use the \p port to send a buffer using the \p callback.
void rpc_send(rpc_port_t port, rpc_port_callback_ty callback, void *data);

/// Use the \p port to send \p bytes using the \p callback. The input is an
/// array of at least the configured lane size.
void rpc_send_n(rpc_port_t port, const void *const *src, uint64_t *size);

/// Use the \p port to recieve a buffer using the \p callback.
void rpc_recv(rpc_port_t port, rpc_port_callback_ty callback, void *data);

/// Use the \p port to recieve \p bytes using the \p callback. The inputs is an
/// array of at least the configured lane size. The \p alloc function allocates
/// memory for the recieved bytes.
void rpc_recv_n(rpc_port_t port, void **dst, uint64_t *size, rpc_alloc_ty alloc,
                void *data);

/// Use the \p port to receive and send a buffer using the \p callback.
void rpc_recv_and_send(rpc_port_t port, rpc_port_callback_ty callback,
                       void *data);

#ifdef __cplusplus
}
#endif

#endif
