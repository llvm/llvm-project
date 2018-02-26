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

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "xray/xray_interface.h"
#include "xray_defs.h"

__sanitizer::SpinMutex XRayImplMutex;
XRayLogImpl CurrentXRayImpl{nullptr, nullptr, nullptr, nullptr};
XRayLogImpl *GlobalXRayImpl = nullptr;

// We use a linked list of Mode to XRayLogImpl mappings. This is a linked list
// when it should be a map because we're avoiding having to depend on C++
// standard library data structures at this level of the implementation.
struct ModeImpl {
  ModeImpl *Next;
  const char *Mode;
  XRayLogImpl Impl;
};

ModeImpl SentinelModeImpl{
    nullptr, nullptr, {nullptr, nullptr, nullptr, nullptr}};
ModeImpl *ModeImpls = &SentinelModeImpl;

XRayLogRegisterStatus
__xray_log_register_mode(const char *Mode,
                         XRayLogImpl Impl) XRAY_NEVER_INSTRUMENT {
  if (Impl.flush_log == nullptr || Impl.handle_arg0 == nullptr ||
      Impl.log_finalize == nullptr || Impl.log_init == nullptr)
    return XRayLogRegisterStatus::XRAY_INCOMPLETE_IMPL;

  __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
  // First, look for whether the mode already has a registered implementation.
  for (ModeImpl *it = ModeImpls; it != &SentinelModeImpl; it = it->Next) {
    if (!__sanitizer::internal_strcmp(Mode, it->Mode))
      return XRayLogRegisterStatus::XRAY_DUPLICATE_MODE;
  }
  auto *NewModeImpl =
      static_cast<ModeImpl *>(__sanitizer::InternalAlloc(sizeof(ModeImpl)));
  NewModeImpl->Next = ModeImpls;
  NewModeImpl->Mode = __sanitizer::internal_strdup(Mode);
  NewModeImpl->Impl = Impl;
  ModeImpls = NewModeImpl;
  return XRayLogRegisterStatus::XRAY_REGISTRATION_OK;
}

XRayLogRegisterStatus
__xray_log_select_mode(const char *Mode) XRAY_NEVER_INSTRUMENT {
  __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
  for (ModeImpl *it = ModeImpls; it != &SentinelModeImpl; it = it->Next) {
    if (!__sanitizer::internal_strcmp(Mode, it->Mode)) {
      CurrentXRayImpl = it->Impl;
      GlobalXRayImpl = &CurrentXRayImpl;
      __xray_set_handler(it->Impl.handle_arg0);
      return XRayLogRegisterStatus::XRAY_REGISTRATION_OK;
    }
  }
  return XRayLogRegisterStatus::XRAY_MODE_NOT_FOUND;
}

void __xray_set_log_impl(XRayLogImpl Impl) XRAY_NEVER_INSTRUMENT {
  if (Impl.log_init == nullptr || Impl.log_finalize == nullptr ||
      Impl.handle_arg0 == nullptr || Impl.flush_log == nullptr) {
    __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
    GlobalXRayImpl = nullptr;
    __xray_remove_handler();
    __xray_remove_handler_arg1();
    return;
  }

  __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
  CurrentXRayImpl = Impl;
  GlobalXRayImpl = &CurrentXRayImpl;
  __xray_set_handler(Impl.handle_arg0);
}

void __xray_remove_log_impl() XRAY_NEVER_INSTRUMENT {
  __sanitizer::SpinMutexLock Guard(&XRayImplMutex);
  GlobalXRayImpl = nullptr;
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
