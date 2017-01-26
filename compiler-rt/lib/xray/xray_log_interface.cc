//===-- xray_log_interface.cc ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a function call tracing system.
//
//===----------------------------------------------------------------------===//
#include "xray/xray_log_interface.h"

#include "xray/xray_interface.h"
#include "xray_defs.h"

#include <memory>
#include <mutex>

std::mutex XRayImplMutex;
std::unique_ptr<XRayLogImpl> GlobalXRayImpl;

void __xray_set_log_impl(XRayLogImpl Impl) XRAY_NEVER_INSTRUMENT {
  if (Impl.log_init == nullptr || Impl.log_finalize == nullptr ||
      Impl.handle_arg0 == nullptr || Impl.flush_log == nullptr) {
    std::lock_guard<std::mutex> Guard(XRayImplMutex);
    GlobalXRayImpl.reset();
    return;
  }

  std::lock_guard<std::mutex> Guard(XRayImplMutex);
  GlobalXRayImpl.reset(new XRayLogImpl);
  *GlobalXRayImpl = Impl;
}

XRayLogInitStatus __xray_init(size_t BufferSize, size_t MaxBuffers, void *Args,
                              size_t ArgsSize) XRAY_NEVER_INSTRUMENT {
  std::lock_guard<std::mutex> Guard(XRayImplMutex);
  if (!GlobalXRayImpl)
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  return GlobalXRayImpl->log_init(BufferSize, MaxBuffers, Args, ArgsSize);
}

XRayLogInitStatus __xray_log_finalize() XRAY_NEVER_INSTRUMENT {
  std::lock_guard<std::mutex> Guard(XRayImplMutex);
  if (!GlobalXRayImpl)
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  return GlobalXRayImpl->log_finalize();
}

XRayLogFlushStatus __xray_log_flushLog() XRAY_NEVER_INSTRUMENT {
  std::lock_guard<std::mutex> Guard(XRayImplMutex);
  if (!GlobalXRayImpl)
    return XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  return GlobalXRayImpl->flush_log();
}
