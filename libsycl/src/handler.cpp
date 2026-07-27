//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>
#include <sycl/__impl/handler.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

static void checkSingleCommand(
    const std::function<std::shared_ptr<detail::EventImpl>()> &CGF) {
  if (CGF) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Attempt to set multiple actions for the command group");
  }
}

void handler::submitKernelImpl(detail::DeviceKernelInfo &KernelInfo,
                               void *ArgData, size_t ArgSize) {
  checkSingleCommand(MCGF);
  MArgData.resize(ArgSize);
  std::memcpy(MArgData.data(), ArgData, ArgSize);
  MCGF = [this, &KernelInfo]() {
    auto EventsImpl = detail::getSyclObjImpls(MDepEvents);
    MQueue.setKernelDependencies(std::move(EventsImpl));
    MQueue.submitKernelImpl(KernelInfo, MArgData.data(), MArgData.size());
    return MQueue.getLastEvent();
  };
}

void handler::setKernelRange(const detail::UnifiedRangeView &Range) {
  MQueue.setKernelRange(Range);
}

void handler::memcpy(void *dest, const void *src, std::size_t numBytes) {
  checkSingleCommand(MCGF);
  MCGF = [this, dest, src, numBytes]() {
    return MQueue.memcpy(dest, src, numBytes,
                         detail::getSyclObjImpls(MDepEvents));
  };
}

std::shared_ptr<detail::EventImpl> handler::finalize() {
  if (!MCGF) {
    auto EventsImpl = detail::getSyclObjImpls(MDepEvents);
    return MQueue.memcpy(nullptr, nullptr, 0, EventsImpl);
  }

  auto Event = MCGF();
  MArgData.clear();
  return Event;
}

_LIBSYCL_END_NAMESPACE_SYCL
