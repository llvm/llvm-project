//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_image_wrapper.hpp>

#include <detail/context_impl.hpp>
#include <detail/offload/offload_utils.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

ProgramWrapper::ProgramWrapper(ContextImpl &Context, ol_device_handle_t Device,
                               const DeviceImageManager &DevImage) {
  assert(Context.getOLHandleRef());
  assert(Device);

  llvm::StringRef Image = DevImage.getOffloadBinary().getImage();
  callAndThrow(Context, olCreateProgram, Context.getOLHandleRef(), Device,
               Image.data(), Image.size(), &MProgram);
}

ProgramWrapper::~ProgramWrapper() {
  assert(MProgram);
  std::ignore = olDestroyProgram(MProgram);
  // TODO: define a way to report errors from dtors.
}

ol_symbol_handle_t
ProgramWrapper::getOrCreateKernel(std::string_view KernelName,
                                  ContextImpl &Context) {
  auto It = MKernels.find(KernelName);
  if (It != MKernels.end())
    return It->second;

  ol_symbol_handle_t Kernel{};
  callAndThrow(Context, olGetSymbol, MProgram, KernelName.data(),
               OL_SYMBOL_KIND_KERNEL, &Kernel);
  MKernels.emplace(KernelName, Kernel);
  return Kernel;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
