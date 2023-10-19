//===-- Shared memory RPC server instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rpc_server.h"

#include "src/__support/RPC/rpc.h"
#include "src/stdio/gpu/file.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <variant>
#include <vector>

using namespace LIBC_NAMESPACE;

static_assert(sizeof(rpc_buffer_t) == sizeof(rpc::Buffer),
              "Buffer size mismatch");

static_assert(RPC_MAXIMUM_PORT_COUNT == rpc::MAX_PORT_COUNT,
              "Incorrect maximum port count");

// The client needs to support different lane sizes for the SIMT model. Because
// of this we need to select between the possible sizes that the client can use.
struct Server {
  template <uint32_t lane_size>
  Server(std::unique_ptr<rpc::Server<lane_size>> &&server)
      : server(std::move(server)) {}

  rpc_status_t handle_server(
      const std::unordered_map<rpc_opcode_t, rpc_opcode_callback_ty> &callbacks,
      const std::unordered_map<rpc_opcode_t, void *> &callback_data,
      uint32_t &index) {
    rpc_status_t ret = RPC_STATUS_SUCCESS;
    std::visit(
        [&](auto &server) {
          ret = handle_server(*server, callbacks, callback_data, index);
        },
        server);
    return ret;
  }

private:
  template <uint32_t lane_size>
  rpc_status_t handle_server(
      rpc::Server<lane_size> &server,
      const std::unordered_map<rpc_opcode_t, rpc_opcode_callback_ty> &callbacks,
      const std::unordered_map<rpc_opcode_t, void *> &callback_data,
      uint32_t &index) {
    auto port = server.try_open(index);
    if (!port)
      return RPC_STATUS_SUCCESS;

    switch (port->get_opcode()) {
    case RPC_WRITE_TO_STREAM:
    case RPC_WRITE_TO_STDERR:
    case RPC_WRITE_TO_STDOUT:
    case RPC_WRITE_TO_STDOUT_NEWLINE: {
      uint64_t sizes[lane_size] = {0};
      void *strs[lane_size] = {nullptr};
      FILE *files[lane_size] = {nullptr};
      if (port->get_opcode() == RPC_WRITE_TO_STREAM) {
        port->recv([&](rpc::Buffer *buffer, uint32_t id) {
          files[id] = reinterpret_cast<FILE *>(buffer->data[0]);
        });
      } else if (port->get_opcode() == RPC_WRITE_TO_STDERR) {
        std::fill(files, files + lane_size, stderr);
      } else {
        std::fill(files, files + lane_size, stdout);
      }

      port->recv_n(strs, sizes, [&](uint64_t size) { return new char[size]; });
      port->send([&](rpc::Buffer *buffer, uint32_t id) {
        buffer->data[0] = fwrite(strs[id], 1, sizes[id], files[id]);
        if (port->get_opcode() == RPC_WRITE_TO_STDOUT_NEWLINE &&
            buffer->data[0] == sizes[id])
          buffer->data[0] += fwrite("\n", 1, 1, files[id]);
        delete[] reinterpret_cast<uint8_t *>(strs[id]);
      });
      break;
    }
    case RPC_READ_FROM_STREAM: {
      uint64_t sizes[lane_size] = {0};
      void *data[lane_size] = {nullptr};
      port->recv([&](rpc::Buffer *buffer, uint32_t id) {
        data[id] = new char[buffer->data[0]];
        sizes[id] = fread(data[id], 1, buffer->data[0],
                          file::to_stream(buffer->data[1]));
      });
      port->send_n(data, sizes);
      port->send([&](rpc::Buffer *buffer, uint32_t id) {
        delete[] reinterpret_cast<uint8_t *>(data[id]);
        std::memcpy(buffer->data, &sizes[id], sizeof(uint64_t));
      });
      break;
    }
    case RPC_READ_FGETS: {
      uint64_t sizes[lane_size] = {0};
      void *data[lane_size] = {nullptr};
      port->recv([&](rpc::Buffer *buffer, uint32_t id) {
        data[id] = new char[buffer->data[0]];
        const char *str =
            fgets(reinterpret_cast<char *>(data[id]), buffer->data[0],
                  file::to_stream(buffer->data[1]));
        sizes[id] = !str ? 0 : std::strlen(str) + 1;
      });
      port->send_n(data, sizes);
      for (uint32_t id = 0; id < lane_size; ++id)
        if (data[id])
          delete[] reinterpret_cast<uint8_t *>(data[id]);
      break;
    }
    case RPC_OPEN_FILE: {
      uint64_t sizes[lane_size] = {0};
      void *paths[lane_size] = {nullptr};
      port->recv_n(paths, sizes, [&](uint64_t size) { return new char[size]; });
      port->recv_and_send([&](rpc::Buffer *buffer, uint32_t id) {
        FILE *file = fopen(reinterpret_cast<char *>(paths[id]),
                           reinterpret_cast<char *>(buffer->data));
        buffer->data[0] = reinterpret_cast<uintptr_t>(file);
      });
      break;
    }
    case RPC_CLOSE_FILE: {
      port->recv_and_send([&](rpc::Buffer *buffer, uint32_t id) {
        FILE *file = reinterpret_cast<FILE *>(buffer->data[0]);
        buffer->data[0] = fclose(file);
      });
      break;
    }
    case RPC_EXIT: {
      // Send a response to the client to signal that we are ready to exit.
      port->recv_and_send([](rpc::Buffer *) {});
      port->recv([](rpc::Buffer *buffer) {
        int status = 0;
        std::memcpy(&status, buffer->data, sizeof(int));
        exit(status);
      });
      break;
    }
    case RPC_ABORT: {
      // Send a response to the client to signal that we are ready to abort.
      port->recv_and_send([](rpc::Buffer *) {});
      port->recv([](rpc::Buffer *) {});
      abort();
      break;
    }
    case RPC_HOST_CALL: {
      uint64_t sizes[lane_size] = {0};
      void *args[lane_size] = {nullptr};
      port->recv_n(args, sizes, [&](uint64_t size) { return new char[size]; });
      port->recv([&](rpc::Buffer *buffer, uint32_t id) {
        reinterpret_cast<void (*)(void *)>(buffer->data[0])(args[id]);
      });
      port->send([&](rpc::Buffer *, uint32_t id) {
        delete[] reinterpret_cast<uint8_t *>(args[id]);
      });
      break;
    }
    case RPC_FEOF: {
      port->recv_and_send([](rpc::Buffer *buffer) {
        buffer->data[0] = feof(file::to_stream(buffer->data[0]));
      });
      break;
    }
    case RPC_FERROR: {
      port->recv_and_send([](rpc::Buffer *buffer) {
        buffer->data[0] = ferror(file::to_stream(buffer->data[0]));
      });
      break;
    }
    case RPC_CLEARERR: {
      port->recv_and_send([](rpc::Buffer *buffer) {
        clearerr(file::to_stream(buffer->data[0]));
      });
      break;
    }
    case RPC_FSEEK: {
      port->recv_and_send([](rpc::Buffer *buffer) {
        buffer->data[0] = fseek(file::to_stream(buffer->data[0]),
                                static_cast<long>(buffer->data[1]),
                                static_cast<int>(buffer->data[2]));
      });
      break;
    }
    case RPC_FTELL: {
      port->recv_and_send([](rpc::Buffer *buffer) {
        buffer->data[0] = ftell(file::to_stream(buffer->data[0]));
      });
      break;
    }
    case RPC_FFLUSH: {
      port->recv_and_send([](rpc::Buffer *buffer) {
        buffer->data[0] = fflush(file::to_stream(buffer->data[0]));
      });
      break;
    }
    case RPC_UNGETC: {
      port->recv_and_send([](rpc::Buffer *buffer) {
        buffer->data[0] = ungetc(static_cast<int>(buffer->data[0]),
                                 file::to_stream(buffer->data[1]));
      });
      break;
    }
    case RPC_NOOP: {
      port->recv([](rpc::Buffer *) {});
      break;
    }
    default: {
      auto handler =
          callbacks.find(static_cast<rpc_opcode_t>(port->get_opcode()));

      // We error out on an unhandled opcode.
      if (handler == callbacks.end())
        return RPC_STATUS_UNHANDLED_OPCODE;

      // Invoke the registered callback with a reference to the port.
      void *data =
          callback_data.at(static_cast<rpc_opcode_t>(port->get_opcode()));
      rpc_port_t port_ref{reinterpret_cast<uint64_t>(&*port), lane_size};
      (handler->second)(port_ref, data);
    }
    }

    // Increment the index so we start the scan after this port.
    index = port->get_index() + 1;
    port->close();
    return RPC_STATUS_CONTINUE;
  }

  std::variant<std::unique_ptr<rpc::Server<1>>,
               std::unique_ptr<rpc::Server<32>>,
               std::unique_ptr<rpc::Server<64>>>
      server;
};

struct Device {
  template <typename T>
  Device(uint32_t num_ports, void *buffer, std::unique_ptr<T> &&server)
      : buffer(buffer), server(std::move(server)), client(num_ports, buffer) {}
  void *buffer;
  Server server;
  rpc::Client client;
  std::unordered_map<rpc_opcode_t, rpc_opcode_callback_ty> callbacks;
  std::unordered_map<rpc_opcode_t, void *> callback_data;
};

// A struct containing all the runtime state required to run the RPC server.
struct State {
  State(uint32_t num_devices)
      : num_devices(num_devices), devices(num_devices), reference_count(0u) {}
  uint32_t num_devices;
  std::vector<std::unique_ptr<Device>> devices;
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
  if (state && state->reference_count-- == 1)
    delete state;

  return RPC_STATUS_SUCCESS;
}

template <uint32_t lane_size>
rpc_status_t server_init_impl(uint32_t device_id, uint64_t num_ports,
                              rpc_alloc_ty alloc, void *data) {
  uint64_t size = rpc::Server<lane_size>::allocation_size(num_ports);
  void *buffer = alloc(size, data);

  if (!buffer)
    return RPC_STATUS_ERROR;

  state->devices[device_id] = std::make_unique<Device>(
      num_ports, buffer,
      std::make_unique<rpc::Server<lane_size>>(num_ports, buffer));
  if (!state->devices[device_id])
    return RPC_STATUS_ERROR;

  return RPC_STATUS_SUCCESS;
}

rpc_status_t rpc_server_init(uint32_t device_id, uint64_t num_ports,
                             uint32_t lane_size, rpc_alloc_ty alloc,
                             void *data) {
  if (!state)
    return RPC_STATUS_NOT_INITIALIZED;
  if (device_id >= state->num_devices)
    return RPC_STATUS_OUT_OF_RANGE;

  if (!state->devices[device_id]) {
    switch (lane_size) {
    case 1:
      if (rpc_status_t err =
              server_init_impl<1>(device_id, num_ports, alloc, data))
        return err;
      break;
    case 32: {
      if (rpc_status_t err =
              server_init_impl<32>(device_id, num_ports, alloc, data))
        return err;
      break;
    }
    case 64:
      if (rpc_status_t err =
              server_init_impl<64>(device_id, num_ports, alloc, data))
        return err;
      break;
    default:
      return RPC_STATUS_INVALID_LANE_SIZE;
    }
  }

  return RPC_STATUS_SUCCESS;
}

rpc_status_t rpc_server_shutdown(uint32_t device_id, rpc_free_ty dealloc,
                                 void *data) {
  if (!state)
    return RPC_STATUS_NOT_INITIALIZED;
  if (device_id >= state->num_devices)
    return RPC_STATUS_OUT_OF_RANGE;
  if (!state->devices[device_id])
    return RPC_STATUS_ERROR;

  dealloc(state->devices[device_id]->buffer, data);
  if (state->devices[device_id])
    state->devices[device_id].release();

  return RPC_STATUS_SUCCESS;
}

rpc_status_t rpc_handle_server(uint32_t device_id) {
  if (!state)
    return RPC_STATUS_NOT_INITIALIZED;
  if (device_id >= state->num_devices)
    return RPC_STATUS_OUT_OF_RANGE;
  if (!state->devices[device_id])
    return RPC_STATUS_ERROR;

  uint32_t index = 0;
  for (;;) {
    auto &device = *state->devices[device_id];
    rpc_status_t status = device.server.handle_server(
        device.callbacks, device.callback_data, index);
    if (status != RPC_STATUS_CONTINUE)
      return status;
  }
}

rpc_status_t rpc_register_callback(uint32_t device_id, rpc_opcode_t opcode,
                                   rpc_opcode_callback_ty callback,
                                   void *data) {
  if (!state)
    return RPC_STATUS_NOT_INITIALIZED;
  if (device_id >= state->num_devices)
    return RPC_STATUS_OUT_OF_RANGE;
  if (!state->devices[device_id])
    return RPC_STATUS_ERROR;

  state->devices[device_id]->callbacks[opcode] = callback;
  state->devices[device_id]->callback_data[opcode] = data;
  return RPC_STATUS_SUCCESS;
}

const void *rpc_get_client_buffer(uint32_t device_id) {
  if (!state || device_id >= state->num_devices || !state->devices[device_id])
    return nullptr;
  return &state->devices[device_id]->client;
}

uint64_t rpc_get_client_size() { return sizeof(rpc::Client); }

using ServerPort = std::variant<rpc::Server<1>::Port *, rpc::Server<32>::Port *,
                                rpc::Server<64>::Port *>;

ServerPort get_port(rpc_port_t ref) {
  if (ref.lane_size == 1)
    return reinterpret_cast<rpc::Server<1>::Port *>(ref.handle);
  else if (ref.lane_size == 32)
    return reinterpret_cast<rpc::Server<32>::Port *>(ref.handle);
  else if (ref.lane_size == 64)
    return reinterpret_cast<rpc::Server<64>::Port *>(ref.handle);
  else
    __builtin_unreachable();
}

void rpc_send(rpc_port_t ref, rpc_port_callback_ty callback, void *data) {
  auto port = get_port(ref);
  std::visit(
      [=](auto &port) {
        port->send([=](rpc::Buffer *buffer) {
          callback(reinterpret_cast<rpc_buffer_t *>(buffer), data);
        });
      },
      port);
}

void rpc_send_n(rpc_port_t ref, const void *const *src, uint64_t *size) {
  auto port = get_port(ref);
  std::visit([=](auto &port) { port->send_n(src, size); }, port);
}

void rpc_recv(rpc_port_t ref, rpc_port_callback_ty callback, void *data) {
  auto port = get_port(ref);
  std::visit(
      [=](auto &port) {
        port->recv([=](rpc::Buffer *buffer) {
          callback(reinterpret_cast<rpc_buffer_t *>(buffer), data);
        });
      },
      port);
}

void rpc_recv_n(rpc_port_t ref, void **dst, uint64_t *size, rpc_alloc_ty alloc,
                void *data) {
  auto port = get_port(ref);
  auto alloc_fn = [=](uint64_t size) { return alloc(size, data); };
  std::visit([=](auto &port) { port->recv_n(dst, size, alloc_fn); }, port);
}

void rpc_recv_and_send(rpc_port_t ref, rpc_port_callback_ty callback,
                       void *data) {
  auto port = get_port(ref);
  std::visit(
      [=](auto &port) {
        port->recv_and_send([=](rpc::Buffer *buffer) {
          callback(reinterpret_cast<rpc_buffer_t *>(buffer), data);
        });
      },
      port);
}
