//===-- Shared memory RPC server instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Server.h"

#include "src/__support/RPC/rpc.h"
#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>

using namespace __llvm_libc;

static_assert(sizeof(rpc_buffer_t) == sizeof(rpc::Buffer),
              "Buffer size mismatch");

static_assert(RPC_MAXIMUM_PORT_COUNT == rpc::DEFAULT_PORT_COUNT,
              "Incorrect maximum port count");
struct Device {
  rpc::Server server;
  std::unordered_map<rpc_opcode_t, rpc_opcode_callback_ty> callbacks;
  std::unordered_map<rpc_opcode_t, void *> callback_data;
};

// A struct containing all the runtime state required to run the RPC server.
struct State {
  State(uint32_t num_devices)
      : num_devices(num_devices),
        devices(std::unique_ptr<Device[]>(new Device[num_devices])),
        reference_count(0u) {}
  uint32_t num_devices;
  std::unique_ptr<Device[]> devices;
  std::atomic_uint32_t reference_count;
};

static std::mutex startup_mutex;

static State *state;

rpc_status_t rpc_init(uint32_t num_devices) {
  std::scoped_lock<decltype(startup_mutex)> lock(startup_mutex);
  if (!state)
    state = new State(num_devices);

  if (state->reference_count == std::numeric_limits<uint32_t>::max())
    return RPC_STATUS_ERROR;

  state->reference_count++;

  return RPC_STATUS_SUCCESS;
}

rpc_status_t rpc_shutdown(void) {
  if (state->reference_count-- == 1)
    delete state;

  return RPC_STATUS_SUCCESS;
}

rpc_status_t rpc_server_init(uint32_t device_id, uint64_t num_ports,
                             uint32_t lane_size, rpc_alloc_ty alloc,
                             void *data) {
  if (device_id >= state->num_devices)
    return RPC_STATUS_OUT_OF_RANGE;

  uint64_t buffer_size =
      __llvm_libc::rpc::Server::allocation_size(num_ports, lane_size);
  void *buffer = alloc(buffer_size, data);

  if (!buffer)
    return RPC_STATUS_ERROR;

  state->devices[device_id].server.reset(num_ports, lane_size, buffer);

  return RPC_STATUS_SUCCESS;
}

rpc_status_t rpc_server_shutdown(uint32_t device_id, rpc_free_ty dealloc,
                                 void *data) {
  if (device_id >= state->num_devices)
    return RPC_STATUS_OUT_OF_RANGE;

  dealloc(rpc_get_buffer(device_id), data);

  return RPC_STATUS_SUCCESS;
}

rpc_status_t rpc_handle_server(uint32_t device_id) {
  if (device_id >= state->num_devices)
    return RPC_STATUS_OUT_OF_RANGE;

  for (;;) {
    auto port = state->devices[device_id].server.try_open();
    if (!port)
      return RPC_STATUS_SUCCESS;

    switch (port->get_opcode()) {
    case rpc::Opcode::WRITE_TO_STREAM:
    case rpc::Opcode::WRITE_TO_STDERR:
    case rpc::Opcode::WRITE_TO_STDOUT: {
      uint64_t sizes[rpc::MAX_LANE_SIZE] = {0};
      void *strs[rpc::MAX_LANE_SIZE] = {nullptr};
      FILE *files[rpc::MAX_LANE_SIZE] = {nullptr};
      if (port->get_opcode() == rpc::Opcode::WRITE_TO_STREAM)
        port->recv([&](rpc::Buffer *buffer, uint32_t id) {
          files[id] = reinterpret_cast<FILE *>(buffer->data[0]);
        });
      port->recv_n(strs, sizes, [&](uint64_t size) { return new char[size]; });
      port->send([&](rpc::Buffer *buffer, uint32_t id) {
        FILE *file = port->get_opcode() == rpc::Opcode::WRITE_TO_STDOUT
                         ? stdout
                         : (port->get_opcode() == rpc::Opcode::WRITE_TO_STDERR
                                ? stderr
                                : files[id]);
        int ret = fwrite(strs[id], sizes[id], 1, file);
        reinterpret_cast<int *>(buffer->data)[0] = ret >= 0 ? sizes[id] : ret;
      });
      for (uint64_t i = 0; i < rpc::MAX_LANE_SIZE; ++i) {
        if (strs[i])
          delete[] reinterpret_cast<uint8_t *>(strs[i]);
      }
      break;
    }
    case rpc::Opcode::EXIT: {
      port->recv([](rpc::Buffer *buffer) {
        exit(reinterpret_cast<uint32_t *>(buffer->data)[0]);
      });
      break;
    }
    // TODO: Move handling of these  test cases to the loader implementation.
    case rpc::Opcode::TEST_INCREMENT: {
      port->recv_and_send([](rpc::Buffer *buffer) {
        reinterpret_cast<uint64_t *>(buffer->data)[0] += 1;
      });
      break;
    }
    case rpc::Opcode::TEST_INTERFACE: {
      uint64_t cnt = 0;
      bool end_with_recv;
      port->recv([&](rpc::Buffer *buffer) { end_with_recv = buffer->data[0]; });
      port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      port->send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      port->send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      port->send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      port->send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      if (end_with_recv)
        port->recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
      else
        port->send(
            [&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
      break;
    }
    case rpc::Opcode::TEST_STREAM: {
      uint64_t sizes[rpc::MAX_LANE_SIZE] = {0};
      void *dst[rpc::MAX_LANE_SIZE] = {nullptr};
      port->recv_n(dst, sizes, [](uint64_t size) { return new char[size]; });
      port->send_n(dst, sizes);
      for (uint64_t i = 0; i < rpc::MAX_LANE_SIZE; ++i) {
        if (dst[i])
          delete[] reinterpret_cast<uint8_t *>(dst[i]);
      }
      break;
    }
    case rpc::Opcode::NOOP: {
      port->recv([](rpc::Buffer *buffer) {});
      break;
    }
    default: {
      auto handler = state->devices[device_id].callbacks.find(
          static_cast<rpc_opcode_t>(port->get_opcode()));

      // We error out on an unhandled opcode.
      if (handler == state->devices[device_id].callbacks.end())
        return RPC_STATUS_UNHANDLED_OPCODE;

      // Invoke the registered callback with a reference to the port.
      void *data = state->devices[device_id].callback_data.at(
          static_cast<rpc_opcode_t>(port->get_opcode()));
      rpc_port_t port_ref{reinterpret_cast<uint64_t>(&*port)};
      (handler->second)(port_ref, data);
    }
    }
    port->close();
  }
}

rpc_status_t rpc_register_callback(uint32_t device_id, rpc_opcode_t opcode,
                                   rpc_opcode_callback_ty callback,
                                   void *data) {
  if (device_id >= state->num_devices)
    return RPC_STATUS_OUT_OF_RANGE;

  state->devices[device_id].callbacks[opcode] = callback;
  state->devices[device_id].callback_data[opcode] = data;
  return RPC_STATUS_SUCCESS;
}

void *rpc_get_buffer(uint32_t device_id) {
  if (device_id >= state->num_devices)
    return nullptr;
  return state->devices[device_id].server.get_buffer_start();
}

void rpc_recv_and_send(rpc_port_t ref, rpc_port_callback_ty callback,
                       void *data) {
  rpc::Server::Port *port = reinterpret_cast<rpc::Server::Port *>(ref.handle);
  port->recv_and_send([=](rpc::Buffer *buffer) {
    callback(reinterpret_cast<rpc_buffer_t *>(buffer), data);
  });
}
