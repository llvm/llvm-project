//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/handler.hpp>

#include <detail/context_impl.hpp>
#include <detail/handler_impl.hpp>
#include <detail/offload/offload_utils.hpp>
#include <detail/queue_impl.hpp>

#include <cstring>
#include <functional>
#include <memory>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

static void checkCommandGroupFunction(
    const std::function<std::shared_ptr<detail::EventImpl>()> &CGF,
    detail::ContextImpl &Context) {
  if (CGF) {
    throw sycl::exception(
        detail::createSyclObjFromImpl<context>(Context),
        sycl::make_error_code(sycl::errc::invalid),
        "Attempt to set multiple actions for the command group");
  }
}

void handler::submitKernelImpl(detail::DeviceKernelInfo &KernelInfo,
                               void *ArgData, std::size_t ArgSize) {
  checkCommandGroupFunction(MImpl.MCGF, MImpl.MQueue.getContext());
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
  MImpl.MRange = detail::convertToOlRange(Range);
}

void handler::memcpy(void *dest, const void *src, std::size_t numBytes) {
  checkCommandGroupFunction(MImpl.MCGF, MImpl.MQueue.getContext());
  MImpl.MCGF = [this, dest, src, numBytes]() {
    return MImpl.MQueue.memcpy(dest, src, numBytes,
                               detail::getSyclObjImpls(MDepEvents));
  };
}

void handler::prefetch(const void *ptr, std::size_t numBytes) {
  checkCommandGroupFunction(MImpl.MCGF, MImpl.MQueue.getContext());
  MImpl.MCGF = [this, ptr, numBytes]() {
    return MImpl.MQueue.prefetch(ptr, numBytes,
                                 detail::getSyclObjImpls(MDepEvents));
  };
}

std::shared_ptr<detail::EventImpl> handler::finalize() {
  if (MImpl.MCGF)
    return MImpl.MCGF();

  auto EventsImpl = detail::getSyclObjImpls(MDepEvents);
  return MImpl.MQueue.submitWait(EventsImpl);
}

void handler::fillImpl(void *Ptr, const void *Pattern, std::size_t PatternSize,
                       std::size_t Count) {
  checkCommandGroupFunction(MImpl.MCGF, MImpl.MQueue.getContext());
  MImpl.MFillPattern.resize(PatternSize);
  std::memcpy(MImpl.MFillPattern.data(), Pattern, PatternSize);
  MImpl.MCGF = [this, Ptr, PatternSize, Count]() {
    return MImpl.MQueue.fill(Ptr, MImpl.MFillPattern.data(), PatternSize, Count,
                             detail::getSyclObjImpls(MDepEvents));
  };
}

_LIBSYCL_END_NAMESPACE_SYCL
