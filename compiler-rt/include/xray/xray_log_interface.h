//===-- xray_log_interface.h ----------------------------------------------===//
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
// APIs for installing a new logging implementation.
//
//===----------------------------------------------------------------------===//
///
/// XRay allows users to implement their own logging handlers and install them
/// to replace the default runtime-controllable implementation that comes with
/// compiler-rt/xray. The "flight data recorder" (FDR) mode implementation uses
/// this API to install itself in an XRay-enabled binary. See
/// compiler-rt/lib/xray_fdr_logging.{h,cc} for details of that implementation.
///
/// The high-level usage pattern for these APIs look like the following:
///
///   // Before we try initializing the log implementation, we must set it as
///   // the log implementation. We provide the function pointers that define
///   // the various initialization, finalization, and other pluggable hooks
///   // that we need.
///   __xray_set_log_impl({...});
///
///   // Once that's done, we can now initialize the implementation. Each
///   // implementation has a chance to let users customize the implementation
///   // with a struct that their implementation supports. Roughly this might
///   // look like:
///   MyImplementationOptions opts;
///   opts.enable_feature = true;
///   ...
///   auto init_status = __xray_log_init(
///       BufferSize, MaxBuffers, &opts, sizeof opts);
///   if (init_status != XRayLogInitStatus::XRAY_LOG_INITIALIZED) {
///     // deal with the error here, if there is one.
///   }
///
///   // When the log implementation has had the chance to initialize, we can
///   // now patch the sleds.
///   auto patch_status = __xray_patch();
///   if (patch_status != XRayPatchingStatus::SUCCESS) {
///     // deal with the error here, if it is an error.
///   }
///
///   // If we want to stop the implementation, we can then finalize it (before
///   // optionally flushing the log).
///   auto fin_status = __xray_log_finalize();
///   if (fin_status != XRayLogInitStatus::XRAY_LOG_FINALIZED) {
///     // deal with the error here, if it is an error.
///   }
///
///   // We can optionally wait before flushing the log to give other threads a
///   // chance to see that the implementation is already finalized. Also, at
///   // this point we can optionally unpatch the sleds to reduce overheads at
///   // runtime.
///   auto unpatch_status = __xray_unpatch();
///   if (unpatch_status != XRayPatchingStatus::SUCCESS) {
//      // deal with the error here, if it is an error.
//    }
///
///   // If there are logs or data to be flushed somewhere, we can do so only
///   // after we've finalized the log. Some implementations may not actually
///   // have anything to log (it might keep the data in memory, or periodically
///   // be logging the data anyway).
///   auto flush_status = __xray_log_flushLog();
///   if (flush_status != XRayLogFlushStatus::XRAY_LOG_FLUSHED) {
///     // deal with the error here, if it is an error.
///   }
///
///
/// NOTE: Before calling __xray_patch() again, consider re-initializing the
/// implementation first. Some implementations might stay in an "off" state when
/// they are finalized, while some might be in an invalid/unknown state.
///
#ifndef XRAY_XRAY_LOG_INTERFACE_H
#define XRAY_XRAY_LOG_INTERFACE_H

#include "xray/xray_interface.h"
#include <stddef.h>

extern "C" {

/// This enum defines the valid states in which the logging implementation can
/// be at.
enum XRayLogInitStatus {
  /// The default state is uninitialized, and in case there were errors in the
  /// initialization, the implementation MUST return XRAY_LOG_UNINITIALIZED.
  XRAY_LOG_UNINITIALIZED = 0,

  /// Some implementations support multi-stage init (or asynchronous init), and
  /// may return XRAY_LOG_INITIALIZING to signal callers of the API that
  /// there's an ongoing initialization routine running. This allows
  /// implementations to support concurrent threads attempting to initialize,
  /// while only signalling success in one.
  XRAY_LOG_INITIALIZING = 1,

  /// When an implementation is done initializing, it MUST return
  /// XRAY_LOG_INITIALIZED. When users call `__xray_patch()`, they are
  /// guaranteed that the implementation installed with
  /// `__xray_set_log_impl(...)` has been initialized.
  XRAY_LOG_INITIALIZED = 2,

  /// Some implementations might support multi-stage finalization (or
  /// asynchronous finalization), and may return XRAY_LOG_FINALIZING to signal
  /// callers of the API that there's an ongoing finalization routine running.
  /// This allows implementations to support concurrent threads attempting to
  /// finalize, while only signalling success/completion in one.
  XRAY_LOG_FINALIZING = 3,

  /// When an implementation is done finalizing, it MUST return
  /// XRAY_LOG_FINALIZED. It is up to the implementation to determine what the
  /// semantics of a finalized implementation is. Some implementations might
  /// allow re-initialization once the log is finalized, while some might always
  /// be on (and that finalization is a no-op).
  XRAY_LOG_FINALIZED = 4,
};

/// This enum allows an implementation to signal log flushing operations via
/// `__xray_log_flushLog()`, and the state of flushing the log.
enum XRayLogFlushStatus {
  XRAY_LOG_NOT_FLUSHING = 0,
  XRAY_LOG_FLUSHING = 1,
  XRAY_LOG_FLUSHED = 2,
};

/// This enum indicates the installation state of a logging implementation, when
/// associating a mode to a particular logging implementation through
/// `__xray_log_register_impl(...)` or through `__xray_log_select_mode(...`.
enum XRayLogRegisterStatus {
  XRAY_REGISTRATION_OK = 0,
  XRAY_DUPLICATE_MODE = 1,
  XRAY_MODE_NOT_FOUND = 2,
  XRAY_INCOMPLETE_IMPL = 3,
};

/// A valid XRay logging implementation MUST provide all of the function
/// pointers in XRayLogImpl when being installed through `__xray_set_log_impl`.
/// To be precise, ALL the functions pointers MUST NOT be nullptr.
struct XRayLogImpl {
  /// The log initialization routine provided by the implementation, always
  /// provided with the following parameters:
  ///
  ///   - buffer size
  ///   - maximum number of buffers
  ///   - a pointer to an argument struct that the implementation MUST handle
  ///   - the size of the argument struct
  ///
  /// See XRayLogInitStatus for details on what the implementation MUST return
  /// when called.
  ///
  /// If the implementation needs to install handlers aside from the 0-argument
  /// function call handler, it MUST do so in this initialization handler.
  ///
  /// See xray_interface.h for available handler installation routines.
  XRayLogInitStatus (*log_init)(size_t, size_t, void *, size_t);

  /// The log finalization routine provided by the implementation.
  ///
  /// See XRayLogInitStatus for details on what the implementation MUST return
  /// when called.
  XRayLogInitStatus (*log_finalize)();

  /// The 0-argument function call handler. XRay logging implementations MUST
  /// always have a handler for function entry and exit events. In case the
  /// implementation wants to support arg1 (or other future extensions to XRay
  /// logging) those MUST be installed by the installed 'log_init' handler.
  ///
  /// Because we didn't want to change the ABI of this struct, the arg1 handler
  /// may be silently overwritten during initialization as well.
  void (*handle_arg0)(int32_t, XRayEntryType);

  /// The log implementation provided routine for when __xray_log_flushLog() is
  /// called.
  ///
  /// See XRayLogFlushStatus for details on what the implementation MUST return
  /// when called.
  XRayLogFlushStatus (*flush_log)();
};

/// This function installs a new logging implementation that XRay will use. In
/// case there are any nullptr members in Impl, XRay will *uninstall any
/// existing implementations*. It does NOT patch the instrumentation sleds.
///
/// NOTE: This function does NOT attempt to finalize the currently installed
/// implementation. Use with caution.
///
/// It is guaranteed safe to call this function in the following states:
///
///   - When the implementation is UNINITIALIZED.
///   - When the implementation is FINALIZED.
///   - When there is no current implementation installed.
///
/// It is logging implementation defined what happens when this function is
/// called while in any other states.
void __xray_set_log_impl(XRayLogImpl Impl);

/// This function registers a logging implementation against a "mode"
/// identifier. This allows multiple modes to be registered, and chosen at
/// runtime using the same mode identifier through
/// `__xray_log_select_mode(...)`.
///
/// We treat the Mode identifier as a null-terminated byte string, as the
/// identifier used when retrieving the log impl.
///
/// Returns:
///   - XRAY_REGISTRATION_OK on success.
///   - XRAY_DUPLICATE_MODE when an implementation is already associated with
///     the provided Mode; does not update the already-registered
///     implementation.
XRayLogRegisterStatus __xray_log_register_mode(const char *Mode,
                                               XRayLogImpl Impl);

/// This function selects the implementation associated with Mode that has been
/// registered through __xray_log_register_mode(...) and installs that
/// implementation (as if through calling __xray_set_log_impl(...)). The same
/// caveats apply to __xray_log_select_mode(...) as with
/// __xray_log_set_log_impl(...).
///
/// Returns:
///   - XRAY_REGISTRATION_OK on success.
///   - XRAY_MODE_NOT_FOUND if there is no implementation associated with Mode;
///     does not update the currently installed implementation.
XRayLogRegisterStatus __xray_log_select_mode(const char *Mode);

/// This function removes the currently installed implementation. It will also
/// uninstall any handlers that have been previously installed. It does NOT
/// unpatch the instrumentation sleds.
///
/// NOTE: This function does NOT attempt to finalize the currently installed
/// implementation. Use with caution.
///
/// It is guaranteed safe to call this function in the following states:
///
///   - When the implementation is UNINITIALIZED.
///   - When the implementation is FINALIZED.
///   - When there is no current implementation installed.
///
/// It is logging implementation defined what happens when this function is
/// called while in any other states.
void __xray_remove_log_impl();

/// Invokes the installed implementation initialization routine. See
/// XRayLogInitStatus for what the return values mean.
XRayLogInitStatus __xray_log_init(size_t BufferSize, size_t MaxBuffers,
                                  void *Args, size_t ArgsSize);

/// Invokes the installed implementation finalization routine. See
/// XRayLogInitStatus for what the return values mean.
XRayLogInitStatus __xray_log_finalize();

/// Invokes the install implementation log flushing routine. See
/// XRayLogFlushStatus for what the return values mean.
XRayLogFlushStatus __xray_log_flushLog();

} // extern "C"

namespace __xray {

/// Options used by the LLVM XRay FDR logging implementation.
struct FDRLoggingOptions {
  bool ReportErrors = false;
  int Fd = -1;
};

/// Options used by the LLVM XRay Basic (Naive) logging implementation.
struct BasicLoggingOptions {
  int DurationFilterMicros = 0;
  size_t MaxStackDepth = 0;
  size_t ThreadBufferSize = 0;
};

} // namespace __xray

#endif // XRAY_XRAY_LOG_INTERFACE_H
