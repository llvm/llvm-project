//===-- Loader Implementation for NVPTX devices --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file impelements a simple loader to run images supporting the NVPTX
// architecture. The file launches the '_start' kernel which should be provided
// by the device application start code and call ultimately call the 'main'
// function.
//
//===----------------------------------------------------------------------===//

#include "Loader.h"

#include "src/__support/RPC/rpc.h"

#include "cuda.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/// The arguments to the '_start' kernel.
struct kernel_args_t {
  int argc;
  void *argv;
  void *envp;
  void *ret;
  void *inbox;
  void *outbox;
  void *buffer;
};

static __llvm_libc::rpc::Server server;

/// Queries the RPC client at least once and performs server-side work if there
/// are any active requests.
void handle_server() {
  while (server.handle(
      [&](__llvm_libc::rpc::Buffer *buffer) {
        switch (static_cast<__llvm_libc::rpc::Opcode>(buffer->data[0])) {
        case __llvm_libc::rpc::Opcode::PRINT_TO_STDERR: {
          fputs(reinterpret_cast<const char *>(&buffer->data[1]), stderr);
          break;
        }
        case __llvm_libc::rpc::Opcode::EXIT: {
          exit(buffer->data[1]);
          break;
        }
        default:
          return;
        };
      },
      [](__llvm_libc::rpc::Buffer *buffer) {}))
    ;
}

static void handle_error(CUresult err) {
  if (err == CUDA_SUCCESS)
    return;

  const char *err_str = nullptr;
  CUresult result = cuGetErrorString(err, &err_str);
  if (result != CUDA_SUCCESS)
    fprintf(stderr, "Unknown Error\n");
  else
    fprintf(stderr, "%s\n", err_str);
  exit(1);
}

static void handle_error(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  exit(EXIT_FAILURE);
}

int load(int argc, char **argv, char **envp, void *image, size_t size,
         const LaunchParameters &params) {
  if (CUresult err = cuInit(0))
    handle_error(err);

  // Obtain the first device found on the system.
  CUdevice device;
  if (CUresult err = cuDeviceGet(&device, 0))
    handle_error(err);

  // Initialize the CUDA context and claim it for this execution.
  CUcontext context;
  if (CUresult err = cuDevicePrimaryCtxRetain(&context, device))
    handle_error(err);
  if (CUresult err = cuCtxSetCurrent(context))
    handle_error(err);

  // Initialize a non-blocking CUDA stream to execute the kernel.
  CUstream stream;
  if (CUresult err = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING))
    handle_error(err);

  // Load the image into a CUDA module.
  CUmodule binary;
  if (CUresult err = cuModuleLoadDataEx(&binary, image, 0, nullptr, nullptr))
    handle_error(err);

  // look up the '_start' kernel in the loaded module.
  CUfunction function;
  if (CUresult err = cuModuleGetFunction(&function, binary, "_start"))
    handle_error(err);

  // Allocate pinned memory on the host to hold the pointer array for the
  // copied argv and allow the GPU device to access it.
  auto allocator = [&](uint64_t size) -> void * {
    void *dev_ptr;
    if (CUresult err = cuMemAllocHost(&dev_ptr, size))
      handle_error(err);
    return dev_ptr;
  };
  void *dev_argv = copy_argument_vector(argc, argv, allocator);
  if (!dev_argv)
    handle_error("Failed to allocate device argv");

  // Allocate pinned memory on the host to hold the pointer array for the
  // copied environment array and allow the GPU device to access it.
  void *dev_envp = copy_environment(envp, allocator);
  if (!dev_envp)
    handle_error("Failed to allocate device environment");

  // Allocate space for the return pointer and initialize it to zero.
  CUdeviceptr dev_ret;
  if (CUresult err = cuMemAlloc(&dev_ret, sizeof(int)))
    handle_error(err);
  if (CUresult err = cuMemsetD32(dev_ret, 0, 1))
    handle_error(err);

  void *server_inbox = allocator(sizeof(__llvm_libc::cpp::Atomic<int>));
  void *server_outbox = allocator(sizeof(__llvm_libc::cpp::Atomic<int>));
  void *buffer = allocator(sizeof(__llvm_libc::rpc::Buffer));
  if (!server_inbox || !server_outbox || !buffer)
    handle_error("Failed to allocate memory the RPC client / server.");

  // Set up the arguments to the '_start' kernel on the GPU.
  uint64_t args_size = sizeof(kernel_args_t);
  kernel_args_t args;
  std::memset(&args, 0, args_size);
  args.argc = argc;
  args.argv = dev_argv;
  args.envp = dev_envp;
  args.ret = reinterpret_cast<void *>(dev_ret);
  args.inbox = server_outbox;
  args.outbox = server_inbox;
  args.buffer = buffer;
  void *args_config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &args,
                         CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                         CU_LAUNCH_PARAM_END};

  // Initialize the RPC server's buffer for host-device communication.
  server.reset(server_inbox, server_outbox, buffer);

  // Call the kernel with the given arguments.
  if (CUresult err = cuLaunchKernel(
          function, params.num_blocks_x, params.num_blocks_y,
          params.num_blocks_z, params.num_threads_x, params.num_threads_y,
          params.num_threads_z, 0, stream, nullptr, args_config))
    handle_error(err);

  // Wait until the kernel has completed execution on the device. Periodically
  // check the RPC client for work to be performed on the server.
  while (cuStreamQuery(stream) == CUDA_ERROR_NOT_READY)
    handle_server();

  // Copy the return value back from the kernel and wait.
  int host_ret = 0;
  if (CUresult err = cuMemcpyDtoH(&host_ret, dev_ret, sizeof(int)))
    handle_error(err);

  if (CUresult err = cuStreamSynchronize(stream))
    handle_error(err);

  // Free the memory allocated for the device.
  if (CUresult err = cuMemFree(dev_ret))
    handle_error(err);
  if (CUresult err = cuMemFreeHost(dev_argv))
    handle_error(err);
  if (CUresult err = cuMemFreeHost(server_inbox))
    handle_error(err);
  if (CUresult err = cuMemFreeHost(server_outbox))
    handle_error(err);
  if (CUresult err = cuMemFreeHost(buffer))
    handle_error(err);

  // Destroy the context and the loaded binary.
  if (CUresult err = cuModuleUnload(binary))
    handle_error(err);
  if (CUresult err = cuDevicePrimaryCtxRelease(device))
    handle_error(err);
  return host_ret;
}
