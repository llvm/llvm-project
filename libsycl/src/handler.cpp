//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/handler_impl.hpp>
#include <detail/offload/offload_utils.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/__impl/handler.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

static void checkCommandGroupFunction(
    const std::function<std::shared_ptr<detail::EventImpl>()> &CGF) {
  if (CGF) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Attempt to set multiple actions for the command group");
  }
}

void handler::submitKernelImpl(detail::DeviceKernelInfo &KernelInfo,
                               void *ArgData, size_t ArgSize) {
  checkCommandGroupFunction(MImpl.MCGF);
  MImpl.MArgData.resize(ArgSize);
  std::memcpy(MImpl.MArgData.data(), ArgData, ArgSize);
  MImpl.MCGF = [this, &KernelInfo]() {
    auto EventsImpl = detail::getSyclObjImpls(MDepEvents);
    MImpl.MQueue.setKernelLaunchParams(std::move(EventsImpl), MImpl.MRange);
    MImpl.MQueue.submitKernelImpl(KernelInfo, MImpl.MArgData.data(),
                                  MImpl.MArgData.size());
    return MImpl.MQueue.getLastEvent();
  };
}

void handler::setKernelRange(const detail::UnifiedRangeView &Range) {
  MImpl.MRange = convertToOlRange(Range);
}

void handler::memcpy(void *dest, const void *src, std::size_t numBytes) {
  checkCommandGroupFunction(MImpl.MCGF);
  MImpl.MCGF = [this, dest, src, numBytes]() {
    return MImpl.MQueue.memcpy(dest, src, numBytes,
                               detail::getSyclObjImpls(MDepEvents));
  };
}

std::shared_ptr<detail::EventImpl> handler::finalize() {
  if (MImpl.MCGF)
    return MImpl.MCGF();

  auto EventsImpl = detail::getSyclObjImpls(MDepEvents);
  return MImpl.MQueue.submitWait(EventsImpl);
}

_LIBSYCL_END_NAMESPACE_SYCL
