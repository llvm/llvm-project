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

#include "cuda.h"

#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace llvm;
using namespace object;

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

// Gets the names of all the globals that contain functions to initialize or
// deinitialize. We need to do this manually because the NVPTX toolchain does
// not contain the necessary binary manipulation tools.
template <typename Alloc>
Expected<void *> get_ctor_dtor_array(const void *image, const size_t size,
                                     Alloc allocator, CUmodule binary) {
  auto mem_buffer = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(image), size), "image",
      /*RequiresNullTerminator=*/false);
  Expected<ELF64LEObjectFile> elf_or_err =
      ELF64LEObjectFile::create(*mem_buffer);
  if (!elf_or_err)
    handle_error(toString(elf_or_err.takeError()).c_str());

  std::vector<std::pair<const char *, uint16_t>> ctors;
  std::vector<std::pair<const char *, uint16_t>> dtors;
  // CUDA has no way to iterate over all the symbols so we need to inspect the
  // ELF directly using the LLVM libraries.
  for (const auto &symbol : elf_or_err->symbols()) {
    auto name_or_err = symbol.getName();
    if (!name_or_err)
      handle_error(toString(name_or_err.takeError()).c_str());

    // Search for all symbols that contain a constructor or destructor.
    if (!name_or_err->starts_with("__init_array_object_") &&
        !name_or_err->starts_with("__fini_array_object_"))
      continue;

    uint16_t priority;
    if (name_or_err->rsplit('_').second.getAsInteger(10, priority))
      handle_error("Invalid priority for constructor or destructor");

    if (name_or_err->starts_with("__init"))
      ctors.emplace_back(std::make_pair(name_or_err->data(), priority));
    else
      dtors.emplace_back(std::make_pair(name_or_err->data(), priority));
  }
  // Lower priority constructors are run before higher ones. The reverse is true
  // for destructors.
  llvm::sort(ctors, [](auto x, auto y) { return x.second < y.second; });
  llvm::sort(dtors, [](auto x, auto y) { return x.second < y.second; });
  llvm::reverse(dtors);

  // Allocate host pinned memory to make these arrays visible to the GPU.
  CUdeviceptr *dev_memory = reinterpret_cast<CUdeviceptr *>(allocator(
      ctors.size() * sizeof(CUdeviceptr) + dtors.size() * sizeof(CUdeviceptr)));
  uint64_t global_size = 0;

  // Get the address of the global and then store the address of the constructor
  // function to call in the constructor array.
  CUdeviceptr *dev_ctors_start = dev_memory;
  CUdeviceptr *dev_ctors_end = dev_ctors_start + ctors.size();
  for (uint64_t i = 0; i < ctors.size(); ++i) {
    CUdeviceptr dev_ptr;
    if (CUresult err =
            cuModuleGetGlobal(&dev_ptr, &global_size, binary, ctors[i].first))
      handle_error(err);
    if (CUresult err =
            cuMemcpyDtoH(&dev_ctors_start[i], dev_ptr, sizeof(uintptr_t)))
      handle_error(err);
  }

  // Get the address of the global and then store the address of the destructor
  // function to call in the destructor array.
  CUdeviceptr *dev_dtors_start = dev_ctors_end;
  CUdeviceptr *dev_dtors_end = dev_dtors_start + dtors.size();
  for (uint64_t i = 0; i < dtors.size(); ++i) {
    CUdeviceptr dev_ptr;
    if (CUresult err =
            cuModuleGetGlobal(&dev_ptr, &global_size, binary, dtors[i].first))
      handle_error(err);
    if (CUresult err =
            cuMemcpyDtoH(&dev_dtors_start[i], dev_ptr, sizeof(uintptr_t)))
      handle_error(err);
  }

  // Obtain the address of the pointers the startup implementation uses to
  // iterate the constructors and destructors.
  CUdeviceptr init_start;
  if (CUresult err = cuModuleGetGlobal(&init_start, &global_size, binary,
                                       "__init_array_start"))
    handle_error(err);
  CUdeviceptr init_end;
  if (CUresult err = cuModuleGetGlobal(&init_end, &global_size, binary,
                                       "__init_array_end"))
    handle_error(err);
  CUdeviceptr fini_start;
  if (CUresult err = cuModuleGetGlobal(&fini_start, &global_size, binary,
                                       "__fini_array_start"))
    handle_error(err);
  CUdeviceptr fini_end;
  if (CUresult err = cuModuleGetGlobal(&fini_end, &global_size, binary,
                                       "__fini_array_end"))
    handle_error(err);

  // Copy the pointers to the newly written array to the symbols so the startup
  // implementation can iterate them.
  if (CUresult err =
          cuMemcpyHtoD(init_start, &dev_ctors_start, sizeof(uintptr_t)))
    handle_error(err);
  if (CUresult err = cuMemcpyHtoD(init_end, &dev_ctors_end, sizeof(uintptr_t)))
    handle_error(err);
  if (CUresult err =
          cuMemcpyHtoD(fini_start, &dev_dtors_start, sizeof(uintptr_t)))
    handle_error(err);
  if (CUresult err = cuMemcpyHtoD(fini_end, &dev_dtors_end, sizeof(uintptr_t)))
    handle_error(err);

  return dev_memory;
}

template <typename args_t>
CUresult launch_kernel(CUmodule binary, CUstream stream,
                       const LaunchParameters &params, const char *kernel_name,
                       args_t kernel_args) {
  // look up the '_start' kernel in the loaded module.
  CUfunction function;
  if (CUresult err = cuModuleGetFunction(&function, binary, kernel_name))
    handle_error(err);

  // Set up the arguments to the '_start' kernel on the GPU.
  uint64_t args_size = sizeof(args_t);
  void *args_config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &kernel_args,
                         CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                         CU_LAUNCH_PARAM_END};

  // Initialize a non-blocking CUDA stream to allocate memory if needed. This
  // needs to be done on a separate stream or else it will deadlock with the
  // executing kernel.
  CUstream memory_stream;
  if (CUresult err = cuStreamCreate(&memory_stream, CU_STREAM_NON_BLOCKING))
    handle_error(err);

  // Register RPC callbacks for the malloc and free functions on HSA.
  uint32_t device_id = 0;
  register_rpc_callbacks(device_id);

  rpc_register_callback(
      device_id, RPC_MALLOC,
      [](rpc_port_t port, void *data) {
        auto malloc_handler = [](rpc_buffer_t *buffer, void *data) -> void {
          CUstream memory_stream = *static_cast<CUstream *>(data);
          uint64_t size = buffer->data[0];
          CUdeviceptr dev_ptr;
          if (CUresult err = cuMemAllocAsync(&dev_ptr, size, memory_stream))
            handle_error(err);

          // Wait until the memory allocation is complete.
          while (cuStreamQuery(memory_stream) == CUDA_ERROR_NOT_READY)
            ;
          buffer->data[0] = static_cast<uintptr_t>(dev_ptr);
        };
        rpc_recv_and_send(port, malloc_handler, data);
      },
      &memory_stream);
  rpc_register_callback(
      device_id, RPC_FREE,
      [](rpc_port_t port, void *data) {
        auto free_handler = [](rpc_buffer_t *buffer, void *data) {
          CUstream memory_stream = *static_cast<CUstream *>(data);
          if (CUresult err = cuMemFreeAsync(
                  static_cast<CUdeviceptr>(buffer->data[0]), memory_stream))
            handle_error(err);
        };
        rpc_recv_and_send(port, free_handler, data);
      },
      &memory_stream);

  // Call the kernel with the given arguments.
  if (CUresult err = cuLaunchKernel(
          function, params.num_blocks_x, params.num_blocks_y,
          params.num_blocks_z, params.num_threads_x, params.num_threads_y,
          params.num_threads_z, 0, stream, nullptr, args_config))
    handle_error(err);

  // Wait until the kernel has completed execution on the device. Periodically
  // check the RPC client for work to be performed on the server.
  while (cuStreamQuery(stream) == CUDA_ERROR_NOT_READY)
    if (rpc_status_t err = rpc_handle_server(device_id))
      handle_error(err);

  // Handle the server one more time in case the kernel exited with a pending
  // send still in flight.
  if (rpc_status_t err = rpc_handle_server(device_id))
    handle_error(err);

  return CUDA_SUCCESS;
}

int load(int argc, char **argv, char **envp, void *image, size_t size,
         const LaunchParameters &params) {
  if (CUresult err = cuInit(0))
    handle_error(err);
  // Obtain the first device found on the system.
  uint32_t num_devices = 1;
  uint32_t device_id = 0;
  CUdevice device;
  if (CUresult err = cuDeviceGet(&device, device_id))
    handle_error(err);

  // Initialize the CUDA context and claim it for this execution.
  CUcontext context;
  if (CUresult err = cuDevicePrimaryCtxRetain(&context, device))
    handle_error(err);
  if (CUresult err = cuCtxSetCurrent(context))
    handle_error(err);

  // Increase the stack size per thread.
  // TODO: We should allow this to be passed in so only the tests that require a
  // larger stack can specify it to save on memory usage.
  if (CUresult err = cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 3 * 1024))
    handle_error(err);

  // Initialize a non-blocking CUDA stream to execute the kernel.
  CUstream stream;
  if (CUresult err = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING))
    handle_error(err);

  // Load the image into a CUDA module.
  CUmodule binary;
  if (CUresult err = cuModuleLoadDataEx(&binary, image, 0, nullptr, nullptr))
    handle_error(err);

  // Allocate pinned memory on the host to hold the pointer array for the
  // copied argv and allow the GPU device to access it.
  auto allocator = [&](uint64_t size) -> void * {
    void *dev_ptr;
    if (CUresult err = cuMemAllocHost(&dev_ptr, size))
      handle_error(err);
    return dev_ptr;
  };

  auto memory_or_err = get_ctor_dtor_array(image, size, allocator, binary);
  if (!memory_or_err)
    handle_error(toString(memory_or_err.takeError()).c_str());

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

  if (rpc_status_t err = rpc_init(num_devices))
    handle_error(err);

  uint32_t warp_size = 32;
  auto rpc_alloc = [](uint64_t size, void *) -> void * {
    void *dev_ptr;
    if (CUresult err = cuMemAllocHost(&dev_ptr, size))
      handle_error(err);
    return dev_ptr;
  };
  if (rpc_status_t err = rpc_server_init(device_id, RPC_MAXIMUM_PORT_COUNT,
                                         warp_size, rpc_alloc, nullptr))
    handle_error(err);

  LaunchParameters single_threaded_params = {1, 1, 1, 1, 1, 1};
  // Call the kernel to
  begin_args_t init_args = {argc, dev_argv, dev_envp,
                            rpc_get_buffer(device_id)};
  if (CUresult err = launch_kernel(binary, stream, single_threaded_params,
                                   "_begin", init_args))
    handle_error(err);

  start_args_t args = {argc, dev_argv, dev_envp,
                       reinterpret_cast<void *>(dev_ret)};
  if (CUresult err = launch_kernel(binary, stream, params, "_start", args))
    handle_error(err);

  // Copy the return value back from the kernel and wait.
  int host_ret = 0;
  if (CUresult err = cuMemcpyDtoH(&host_ret, dev_ret, sizeof(int)))
    handle_error(err);

  if (CUresult err = cuStreamSynchronize(stream))
    handle_error(err);

  end_args_t fini_args = {host_ret};
  if (CUresult err = launch_kernel(binary, stream, single_threaded_params,
                                   "_end", fini_args))
    handle_error(err);

  // Free the memory allocated for the device.
  if (CUresult err = cuMemFreeHost(*memory_or_err))
    handle_error(err);
  if (CUresult err = cuMemFree(dev_ret))
    handle_error(err);
  if (CUresult err = cuMemFreeHost(dev_argv))
    handle_error(err);
  if (rpc_status_t err = rpc_server_shutdown(
          device_id, [](void *ptr, void *) { cuMemFreeHost(ptr); }, nullptr))
    handle_error(err);

  // Destroy the context and the loaded binary.
  if (CUresult err = cuModuleUnload(binary))
    handle_error(err);
  if (CUresult err = cuDevicePrimaryCtxRelease(device))
    handle_error(err);
  if (rpc_status_t err = rpc_shutdown())
    handle_error(err);
  return host_ret;
}
