//===-- Loader Implementation for AMDHSA devices --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file impelements a simple loader to run images supporting the AMDHSA
// architecture. The file launches the '_start' kernel which should be provided
// by the device application start code and call ultimately call the 'main'
// function.
//
//===----------------------------------------------------------------------===//

#include "Loader.h"
#include "Server.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

/// Print the error code and exit if \p code indicates an error.
static void handle_error(hsa_status_t code) {
  if (code == HSA_STATUS_SUCCESS || code == HSA_STATUS_INFO_BREAK)
    return;

  const char *desc;
  if (hsa_status_string(code, &desc) != HSA_STATUS_SUCCESS)
    desc = "Unknown error";
  fprintf(stderr, "%s\n", desc);
  exit(EXIT_FAILURE);
}

static void handle_error(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  exit(EXIT_FAILURE);
}

/// Generic interface for iterating using the HSA callbacks.
template <typename elem_ty, typename func_ty, typename callback_ty>
hsa_status_t iterate(func_ty func, callback_ty cb) {
  auto l = [](elem_ty elem, void *data) -> hsa_status_t {
    callback_ty *unwrapped = static_cast<callback_ty *>(data);
    return (*unwrapped)(elem);
  };
  return func(l, static_cast<void *>(&cb));
}

/// Generic interface for iterating using the HSA callbacks.
template <typename elem_ty, typename func_ty, typename func_arg_ty,
          typename callback_ty>
hsa_status_t iterate(func_ty func, func_arg_ty func_arg, callback_ty cb) {
  auto l = [](elem_ty elem, void *data) -> hsa_status_t {
    callback_ty *unwrapped = static_cast<callback_ty *>(data);
    return (*unwrapped)(elem);
  };
  return func(func_arg, l, static_cast<void *>(&cb));
}

/// Iterate through all availible agents.
template <typename callback_ty>
hsa_status_t iterate_agents(callback_ty callback) {
  return iterate<hsa_agent_t>(hsa_iterate_agents, callback);
}

/// Iterate through all availible memory pools.
template <typename callback_ty>
hsa_status_t iterate_agent_memory_pools(hsa_agent_t agent, callback_ty cb) {
  return iterate<hsa_amd_memory_pool_t>(hsa_amd_agent_iterate_memory_pools,
                                        agent, cb);
}

template <hsa_device_type_t flag>
hsa_status_t get_agent(hsa_agent_t *output_agent) {
  // Find the first agent with a matching device type.
  auto cb = [&](hsa_agent_t hsa_agent) -> hsa_status_t {
    hsa_device_type_t type;
    hsa_status_t status =
        hsa_agent_get_info(hsa_agent, HSA_AGENT_INFO_DEVICE, &type);
    if (status != HSA_STATUS_SUCCESS)
      return status;

    if (type == flag) {
      // Ensure that a GPU agent supports kernel dispatch packets.
      if (type == HSA_DEVICE_TYPE_GPU) {
        hsa_agent_feature_t features;
        status =
            hsa_agent_get_info(hsa_agent, HSA_AGENT_INFO_FEATURE, &features);
        if (status != HSA_STATUS_SUCCESS)
          return status;
        if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)
          *output_agent = hsa_agent;
      } else {
        *output_agent = hsa_agent;
      }
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  };

  return iterate_agents(cb);
}

/// Retrieve a global memory pool with a \p flag from the agent.
template <hsa_amd_memory_pool_global_flag_t flag>
hsa_status_t get_agent_memory_pool(hsa_agent_t agent,
                                   hsa_amd_memory_pool_t *output_pool) {
  auto cb = [&](hsa_amd_memory_pool_t memory_pool) {
    uint32_t flags;
    hsa_amd_segment_t segment;
    if (auto err = hsa_amd_memory_pool_get_info(
            memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment))
      return err;
    if (auto err = hsa_amd_memory_pool_get_info(
            memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags))
      return err;

    if (segment != HSA_AMD_SEGMENT_GLOBAL)
      return HSA_STATUS_SUCCESS;

    if (flags & flag)
      *output_pool = memory_pool;

    return HSA_STATUS_SUCCESS;
  };
  return iterate_agent_memory_pools(agent, cb);
}

template <typename args_t>
hsa_status_t launch_kernel(hsa_agent_t dev_agent, hsa_executable_t executable,
                           hsa_amd_memory_pool_t kernargs_pool,
                           hsa_queue_t *queue, const LaunchParameters &params,
                           const char *kernel_name, args_t kernel_args) {
  // Look up the '_start' kernel in the loaded executable.
  hsa_executable_symbol_t symbol;
  if (hsa_status_t err = hsa_executable_get_symbol_by_name(
          executable, kernel_name, &dev_agent, &symbol))
    return err;

  // Retrieve different properties of the kernel symbol used for launch.
  uint64_t kernel;
  uint32_t args_size;
  uint32_t group_size;
  uint32_t private_size;

  std::pair<hsa_executable_symbol_info_t, void *> symbol_infos[] = {
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &args_size},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_size},
      {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_size}};

  for (auto &[info, value] : symbol_infos)
    if (hsa_status_t err = hsa_executable_symbol_get_info(symbol, info, value))
      return err;

  // Allocate space for the kernel arguments on the host and allow the GPU agent
  // to access it.
  void *args;
  if (hsa_status_t err = hsa_amd_memory_pool_allocate(kernargs_pool, args_size,
                                                      /*flags=*/0, &args))
    handle_error(err);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, args);

  // Initialie all the arguments (explicit and implicit) to zero, then set the
  // explicit arguments to the values created above.
  std::memset(args, 0, args_size);
  std::memcpy(args, &kernel_args, sizeof(args_t));

  // Obtain a packet from the queue.
  uint64_t packet_id = hsa_queue_add_write_index_relaxed(queue, 1);
  while (packet_id - hsa_queue_load_read_index_scacquire(queue) >= queue->size)
    ;

  const uint32_t mask = queue->size - 1;
  hsa_kernel_dispatch_packet_t *packet =
      static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
      (packet_id & mask);

  // Set up the packet for exeuction on the device. We currently only launch
  // with one thread on the device, forcing the rest of the wavefront to be
  // masked off.
  std::memset(packet, 0, sizeof(hsa_kernel_dispatch_packet_t));
  packet->setup = (1 + (params.num_blocks_y * params.num_threads_y != 1) +
                   (params.num_blocks_z * params.num_threads_z != 1))
                  << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet->workgroup_size_x = params.num_threads_x;
  packet->workgroup_size_y = params.num_threads_y;
  packet->workgroup_size_z = params.num_threads_z;
  packet->grid_size_x = params.num_blocks_x * params.num_threads_x;
  packet->grid_size_y = params.num_blocks_y * params.num_threads_y;
  packet->grid_size_z = params.num_blocks_z * params.num_threads_z;
  packet->private_segment_size = private_size;
  packet->group_segment_size = group_size;
  packet->kernel_object = kernel;
  packet->kernarg_address = args;

  // Create a signal to indicate when this packet has been completed.
  if (hsa_status_t err =
          hsa_signal_create(1, 0, nullptr, &packet->completion_signal))
    handle_error(err);

  // Initialize the packet header and set the doorbell signal to begin execution
  // by the HSA runtime.
  uint16_t setup = packet->setup;
  uint16_t header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  __atomic_store_n(&packet->header, header | (setup << 16), __ATOMIC_RELEASE);
  hsa_signal_store_relaxed(queue->doorbell_signal, packet_id);

  // Wait until the kernel has completed execution on the device. Periodically
  // check the RPC client for work to be performed on the server.
  while (hsa_signal_wait_scacquire(
             packet->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0,
             /*timeout_hint=*/1024, HSA_WAIT_STATE_ACTIVE) != 0)
    handle_server();

  // Destroy the resources acquired to launch the kernel and return.
  if (hsa_status_t err = hsa_amd_memory_pool_free(args))
    handle_error(err);
  if (hsa_status_t err = hsa_signal_destroy(packet->completion_signal))
    handle_error(err);

  return HSA_STATUS_SUCCESS;
}

int load(int argc, char **argv, char **envp, void *image, size_t size,
         const LaunchParameters &params) {
  // Initialize the HSA runtime used to communicate with the device.
  if (hsa_status_t err = hsa_init())
    handle_error(err);

  // Register a callback when the device encounters a memory fault.
  if (hsa_status_t err = hsa_amd_register_system_event_handler(
          [](const hsa_amd_event_t *event, void *) -> hsa_status_t {
            if (event->event_type == HSA_AMD_GPU_MEMORY_FAULT_EVENT)
              return HSA_STATUS_ERROR;
            return HSA_STATUS_SUCCESS;
          },
          nullptr))
    handle_error(err);

  // Obtain an agent for the device and host to use the HSA memory model.
  hsa_agent_t dev_agent;
  hsa_agent_t host_agent;
  if (hsa_status_t err = get_agent<HSA_DEVICE_TYPE_GPU>(&dev_agent))
    handle_error(err);
  if (hsa_status_t err = get_agent<HSA_DEVICE_TYPE_CPU>(&host_agent))
    handle_error(err);

  // Load the code object's ISA information and executable data segments.
  hsa_code_object_t object;
  if (hsa_status_t err = hsa_code_object_deserialize(image, size, "", &object))
    handle_error(err);

  hsa_executable_t executable;
  if (hsa_status_t err = hsa_executable_create_alt(
          HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO, "",
          &executable))
    handle_error(err);

  if (hsa_status_t err =
          hsa_executable_load_code_object(executable, dev_agent, object, ""))
    handle_error(err);

  // No modifications to the executable are allowed  after this point.
  if (hsa_status_t err = hsa_executable_freeze(executable, ""))
    handle_error(err);

  // Check the validity of the loaded executable. If the agents ISA features do
  // not match the executable's code object it will fail here.
  uint32_t result;
  if (hsa_status_t err = hsa_executable_validate(executable, &result))
    handle_error(err);
  if (result)
    handle_error(HSA_STATUS_ERROR);

  // Obtain memory pools to exchange data between the host and the device. The
  // fine-grained pool acts as pinned memory on the host for DMA transfers to
  // the device, the coarse-grained pool is for allocations directly on the
  // device, and the kernerl-argument pool is for executing the kernel.
  hsa_amd_memory_pool_t kernargs_pool;
  hsa_amd_memory_pool_t finegrained_pool;
  hsa_amd_memory_pool_t coarsegrained_pool;
  if (hsa_status_t err =
          get_agent_memory_pool<HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT>(
              host_agent, &kernargs_pool))
    handle_error(err);
  if (hsa_status_t err =
          get_agent_memory_pool<HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED>(
              host_agent, &finegrained_pool))
    handle_error(err);
  if (hsa_status_t err =
          get_agent_memory_pool<HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED>(
              dev_agent, &coarsegrained_pool))
    handle_error(err);

  // Allocate fine-grained memory on the host to hold the pointer array for the
  // copied argv and allow the GPU agent to access it.
  auto allocator = [&](uint64_t size) -> void * {
    void *dev_ptr = nullptr;
    if (hsa_status_t err = hsa_amd_memory_pool_allocate(finegrained_pool, size,
                                                        /*flags=*/0, &dev_ptr))
      handle_error(err);
    hsa_amd_agents_allow_access(1, &dev_agent, nullptr, dev_ptr);
    return dev_ptr;
  };
  void *dev_argv = copy_argument_vector(argc, argv, allocator);
  if (!dev_argv)
    handle_error("Failed to allocate device argv");

  // Allocate fine-grained memory on the host to hold the pointer array for the
  // copied environment array and allow the GPU agent to access it.
  void *dev_envp = copy_environment(envp, allocator);
  if (!dev_envp)
    handle_error("Failed to allocate device environment");

  // Allocate space for the return pointer and initialize it to zero.
  void *dev_ret;
  if (hsa_status_t err =
          hsa_amd_memory_pool_allocate(coarsegrained_pool, sizeof(int),
                                       /*flags=*/0, &dev_ret))
    handle_error(err);
  hsa_amd_memory_fill(dev_ret, 0, sizeof(int));

  // Allocate finegrained memory for the RPC server and client to share.
  uint64_t port_size = __llvm_libc::rpc::default_port_count;
  uint32_t wavefront_size = 0;
  if (hsa_status_t err = hsa_agent_get_info(
          dev_agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size))
    handle_error(err);
  void *server_inbox;
  void *server_outbox;
  void *buffer;
  if (hsa_status_t err = hsa_amd_memory_pool_allocate(
          finegrained_pool, port_size * sizeof(__llvm_libc::cpp::Atomic<int>),
          /*flags=*/0, &server_inbox))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_allocate(
          finegrained_pool, port_size * sizeof(__llvm_libc::cpp::Atomic<int>),
          /*flags=*/0, &server_outbox))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_allocate(
          finegrained_pool,
          port_size *
              align_up(sizeof(__llvm_libc::rpc::Header) +
                           (wavefront_size * sizeof(__llvm_libc::rpc::Buffer)),
                       alignof(__llvm_libc::rpc::Packet)),
          /*flags=*/0, &buffer))
    handle_error(err);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, server_inbox);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, server_outbox);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, buffer);

  // Initialize the RPC server's buffer for host-device communication.
  server.reset(port_size, wavefront_size, &lock, server_inbox, server_outbox,
               buffer);

  // Obtain a queue with the minimum (power of two) size, used to send commands
  // to the HSA runtime and launch execution on the device.
  uint64_t queue_size;
  if (hsa_status_t err = hsa_agent_get_info(
          dev_agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &queue_size))
    handle_error(err);
  hsa_queue_t *queue = nullptr;
  if (hsa_status_t err =
          hsa_queue_create(dev_agent, queue_size, HSA_QUEUE_TYPE_MULTI, nullptr,
                           nullptr, UINT32_MAX, UINT32_MAX, &queue))
    handle_error(err);

  LaunchParameters single_threaded_params = {1, 1, 1, 1, 1, 1};
  begin_args_t init_args = {argc,          dev_argv,     dev_envp,
                            server_outbox, server_inbox, buffer};
  if (hsa_status_t err =
          launch_kernel(dev_agent, executable, kernargs_pool, queue,
                        single_threaded_params, "_begin.kd", init_args))
    handle_error(err);

  start_args_t args = {argc, dev_argv, dev_envp, dev_ret};
  if (hsa_status_t err = launch_kernel(dev_agent, executable, kernargs_pool,
                                       queue, params, "_start.kd", args))
    handle_error(err);

  // Create a memory signal and copy the return value back from the device into
  // a new buffer.
  hsa_signal_t memory_signal;
  if (hsa_status_t err = hsa_signal_create(1, 0, nullptr, &memory_signal))
    handle_error(err);

  void *host_ret;
  if (hsa_status_t err =
          hsa_amd_memory_pool_allocate(finegrained_pool, sizeof(int),
                                       /*flags=*/0, &host_ret))
    handle_error(err);
  hsa_amd_agents_allow_access(1, &dev_agent, nullptr, host_ret);

  if (hsa_status_t err =
          hsa_amd_memory_async_copy(host_ret, host_agent, dev_ret, dev_agent,
                                    sizeof(int), 0, nullptr, memory_signal))
    handle_error(err);

  while (hsa_signal_wait_scacquire(memory_signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                   UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0)
    ;

  // Save the return value and perform basic clean-up.
  int ret = *static_cast<int *>(host_ret);

  end_args_t fini_args = {ret};
  if (hsa_status_t err =
          launch_kernel(dev_agent, executable, kernargs_pool, queue,
                        single_threaded_params, "_end.kd", fini_args))
    handle_error(err);

  // Free the memory allocated for the device.
  if (hsa_status_t err = hsa_amd_memory_pool_free(dev_argv))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_free(dev_ret))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_free(server_inbox))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_free(server_outbox))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_free(buffer))
    handle_error(err);
  if (hsa_status_t err = hsa_amd_memory_pool_free(host_ret))
    handle_error(err);

  if (hsa_status_t err = hsa_signal_destroy(memory_signal))
    handle_error(err);
  if (hsa_status_t err = hsa_queue_destroy(queue))
    handle_error(err);

  if (hsa_status_t err = hsa_executable_destroy(executable))
    handle_error(err);

  if (hsa_status_t err = hsa_code_object_destroy(object))
    handle_error(err);

  if (hsa_status_t err = hsa_shut_down())
    handle_error(err);

  return ret;
}
