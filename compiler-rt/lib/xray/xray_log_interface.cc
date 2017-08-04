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

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "xray/xray_interface.h"
#include "xray_defs.h"

#include <memory>

__sanitizer::SpinMutex XRayImplMutex;
std::unique_ptr<XRayLogImpl> GlobalXRayImpl;

void __xray_set_log_impl(XRayLogImpl Impl) XRAY_NEVER_INSTRUMENT {
  if (Impl.log_init == nullptr || Impl.log_finalize == nullptr ||
      Impl.handle_arg0 == nullptr || Impl.flush_log == nullptr) {
    __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
    GlobalXRayImpl.reset();
    __xray_remove_handler();
    __xray_remove_handler_arg1();
    return;
  }

  __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
  GlobalXRayImpl.reset(new XRayLogImpl);
  *GlobalXRayImpl = Impl;
  __xray_set_handler(Impl.handle_arg0);
}

void __xray_remove_log_impl() XRAY_NEVER_INSTRUMENT {
  __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
  GlobalXRayImpl.reset();
  __xray_remove_handler();
  __xray_remove_handler_arg1();
}

XRayLogInitStatus __xray_log_init(size_t BufferSize, size_t MaxBuffers,
                                  void *Args,
                                  size_t ArgsSize) XRAY_NEVER_INSTRUMENT {
  __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
  if (!GlobalXRayImpl)
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  return GlobalXRayImpl->log_init(BufferSize, MaxBuffers, Args, ArgsSize);
}

XRayLogInitStatus __xray_log_finalize() XRAY_NEVER_INSTRUMENT {
  __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
  if (!GlobalXRayImpl)
    return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
  return GlobalXRayImpl->log_finalize();
}

XRayLogFlushStatus __xray_log_flushLog() XRAY_NEVER_INSTRUMENT {
  __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
  if (!GlobalXRayImpl)
    return XRayLogFlushStatus::XRAY_LOG_NOT_FLUSHING;
  return GlobalXRayImpl->flush_log();
}
