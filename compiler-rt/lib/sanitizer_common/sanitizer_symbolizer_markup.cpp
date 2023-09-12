//===-- sanitizer_symbolizer_markup.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries.
//
// Implementation of offline markup symbolizer.
//===----------------------------------------------------------------------===//

#include "sanitizer_symbolizer_markup.h"

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_platform.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"
#include "sanitizer_symbolizer_fuchsia.h"

namespace __sanitizer {

void RenderDataMarkup(InternalScopedString *buffer, const DataInfo *DI) {
  buffer->append(kFormatData, DI->start);
}

bool RenderNeedsSymbolizationMarkup() { return false; }

void RenderFrameMarkup(InternalScopedString *buffer, int frame_no,
                       uptr address) {
  CHECK(!RenderNeedsSymbolizationMarkup());
  buffer->append(kFormatFrame, frame_no, address);
}

// Simplier view of a LoadedModule. It only holds information necessary to
// identify unique modules.
struct RenderedModule {
  char *full_name;
  u8 uuid[kModuleUUIDSize];  // BuildId
  uptr base_address;
};

bool ModulesEq(const LoadedModule *module,
               const RenderedModule *renderedModule) {
  return module->base_address() == renderedModule->base_address &&
         internal_memcmp(module->uuid(), renderedModule->uuid,
                         module->uuid_size()) == 0 &&
         internal_strcmp(module->full_name(), renderedModule->full_name) == 0;
}

bool ModuleHasBeenRendered(
    const LoadedModule *module,
    const InternalMmapVectorNoCtor<RenderedModule> *renderedModules) {
  for (auto *it = renderedModules->begin(); it != renderedModules->end();
       ++it) {
    const auto &renderedModule = *it;
    if (ModulesEq(module, &renderedModule)) {
      return true;
    }
  }
  return false;
}

void RenderModulesMarkup(InternalScopedString *buffer,
                         const ListOfModules *modules) {
  // Keeps track of the modules that have been rendered.
  static bool initialized = false;
  static InternalMmapVectorNoCtor<RenderedModule> renderedModules;
  if (!initialized) {
    renderedModules.Initialize(modules->size());
    initialized = true;
  }

  if (!renderedModules.size()) {
    buffer->append("{{{reset}}}\n");
  }

  for (auto *moduleIt = modules->begin(); moduleIt != modules->end();
       ++moduleIt) {
    const LoadedModule &module = *moduleIt;

    if (ModuleHasBeenRendered(&module, &renderedModules)) {
      continue;
    }

    buffer->append("{{{module:%d:%s:elf:", renderedModules.size(),
                   module.full_name());
    for (uptr i = 0; i < module.uuid_size(); i++) {
      buffer->append("%02x", module.uuid()[i]);
    }
    buffer->append("}}}\n");

    for (const auto &range : module.ranges()) {
      buffer->append("{{{mmap:%p:%p:load:%d:r", range.beg,
                     range.end - range.beg, renderedModules.size());
      if (range.writable)
        buffer->append("w");
      if (range.executable)
        buffer->append("x");

      // module.base_address = dlpi_addr
      // range.beg = dlpi_addr + p_vaddr
      // relative address = p_vaddr = range.beg - module.base_address
      buffer->append(":%p}}}\n", range.beg - module.base_address());
    }

    renderedModules.push_back({});
    RenderedModule &curModule = renderedModules.back();
    curModule.full_name = internal_strdup(module.full_name());

    // kModuleUUIDSize is the size of curModule.uuid
    CHECK_GE(kModuleUUIDSize, module.uuid_size());
    internal_memcpy(curModule.uuid, module.uuid(), module.uuid_size());

    curModule.base_address = module.base_address();
  }
}

bool MarkupSymbolizer::SymbolizePC(uptr addr, SymbolizedStack *stack) {
  char buffer[kFormatFunctionMax];
  internal_snprintf(buffer, sizeof(buffer), kFormatFunction, addr);
  stack->info.function = internal_strdup(buffer);
  return true;
}

bool MarkupSymbolizer::SymbolizeData(uptr addr, DataInfo *info) {
  info->Clear();
  info->start = addr;
  return true;
}

// This is used by UBSan for type names, and by ASan for global variable names.
// It's expected to return a static buffer that will be reused on each call.
const char *MarkupSymbolizer::Demangle(const char *name) {
  static char buffer[kFormatDemangleMax];
  internal_snprintf(buffer, sizeof(buffer), kFormatDemangle, name);
  return buffer;
}

#if SANITIZER_SYMBOLIZER_MARKUP
#  include <limits.h>
#  include <unwind.h>

// This generic support for offline symbolizing is based on the
// Fuchsia port.  We don't do any actual symbolization per se.
// Instead, we emit text containing raw addresses and raw linkage
// symbol names, embedded in Fuchsia's symbolization markup format.
// Fuchsia's logging infrastructure emits enough information about
// process memory layout that a post-processing filter can do the
// symbolization and pretty-print the markup.  See the spec at:
// https://fuchsia.googlesource.com/zircon/+/master/docs/symbolizer_markup.md

// This is used by UBSan for type names, and by ASan for global variable names.
// It's expected to return a static buffer that will be reused on each call.
const char *Symbolizer::Demangle(const char *name) {
  SymbolizerTool *markupSymbolizer = tools_.front();
  CHECK(markupSymbolizer);
  return markupSymbolizer->Demangle(name);
}

// This is used mostly for suppression matching.  Making it work
// would enable "interceptor_via_lib" suppressions.  It's also used
// once in UBSan to say "in module ..." in a message that also
// includes an address in the module, so post-processing can already
// pretty-print that so as to indicate the module.
bool Symbolizer::GetModuleNameAndOffsetForPC(uptr pc, const char **module_name,
                                             uptr *module_address) {
  return false;
}

// This is mainly used by hwasan for online symbolization. This isn't needed
// since hwasan can always just dump stack frames for offline symbolization.
bool Symbolizer::SymbolizeFrame(uptr addr, FrameInfo *info) { return false; }

// This is used in some places for suppression checking, which we
// don't really support for Fuchsia.  It's also used in UBSan to
// identify a PC location to a function name, so we always fill in
// the function member with a string containing markup around the PC
// value.
// TODO(mcgrathr): Under SANITIZER_GO, it's currently used by TSan
// to render stack frames, but that should be changed to use
// RenderStackFrame.
SymbolizedStack *Symbolizer::SymbolizePC(uptr addr) {
  SymbolizerTool *markupSymbolizer = tools_.front();
  CHECK(markupSymbolizer);

  SymbolizedStack *s = SymbolizedStack::New(addr);
  markupSymbolizer->SymbolizePC(addr, s);
  return s;
}

// Always claim we succeeded, so that RenderDataInfo will be called.
bool Symbolizer::SymbolizeData(uptr addr, DataInfo *info) {
  SymbolizerTool *markupSymbolizer = tools_.front();
  CHECK(markupSymbolizer);
  return markupSymbolizer->SymbolizeData(addr, info);
}

// We ignore the format argument to __sanitizer_symbolize_global.
void RenderData(InternalScopedString *buffer, const char *format,
                const DataInfo *DI, bool symbolizer_markup,
                const char *strip_path_prefix) {
  RenderDataMarkup(buffer, DI);
}

bool RenderNeedsSymbolization(const char *format, bool symbolizer_markup) {
  return RenderNeedsSymbolizationMarkup();
}

// We don't support the stack_trace_format flag at all.
void RenderFrame(InternalScopedString *buffer, const char *format, int frame_no,
                 uptr address, const AddressInfo *info, bool vs_style,
                 const char *strip_path_prefix) {
  RenderFrameMarkup(buffer, frame_no, address);
}

Symbolizer *Symbolizer::PlatformInit() {
  IntrusiveList<SymbolizerTool> tools;
  SymbolizerTool *tool = new(symbolizer_allocator_) MarkupSymbolizer();
  tools.push_back(tool);
  return new (symbolizer_allocator_) Symbolizer(tools);
}

void Symbolizer::LateInitialize() { Symbolizer::GetOrInit(); }

void StartReportDeadlySignal() {}
void ReportDeadlySignal(const SignalContext &sig, u32 tid,
                        UnwindSignalStackCallbackType unwind,
                        const void *unwind_context) {}

#  if SANITIZER_CAN_SLOW_UNWIND
struct UnwindTraceArg {
  BufferedStackTrace *stack;
  u32 max_depth;
};

_Unwind_Reason_Code Unwind_Trace(struct _Unwind_Context *ctx, void *param) {
  UnwindTraceArg *arg = static_cast<UnwindTraceArg *>(param);
  CHECK_LT(arg->stack->size, arg->max_depth);
  uptr pc = _Unwind_GetIP(ctx);
  if (pc < PAGE_SIZE)
    return _URC_NORMAL_STOP;
  arg->stack->trace_buffer[arg->stack->size++] = pc;
  return (arg->stack->size == arg->max_depth ? _URC_NORMAL_STOP
                                             : _URC_NO_REASON);
}

void BufferedStackTrace::UnwindSlow(uptr pc, u32 max_depth) {
  CHECK_GE(max_depth, 2);
  size = 0;
  UnwindTraceArg arg = {this, Min(max_depth + 1, kStackTraceMax)};
  _Unwind_Backtrace(Unwind_Trace, &arg);
  CHECK_GT(size, 0);
  // We need to pop a few frames so that pc is on top.
  uptr to_pop = LocatePcInTrace(pc);
  // trace_buffer[0] belongs to the current function so we always pop it,
  // unless there is only 1 frame in the stack trace (1 frame is always better
  // than 0!).
  PopStackFrames(Min(to_pop, static_cast<uptr>(1)));
  trace_buffer[0] = pc;
}

void BufferedStackTrace::UnwindSlow(uptr pc, void *context, u32 max_depth) {
  CHECK(context);
  CHECK_GE(max_depth, 2);
  UNREACHABLE("signal context doesn't exist");
}
#  endif  // SANITIZER_CAN_SLOW_UNWIND

#endif  // SANITIZER_SYMBOLIZER_MARKUP

}  // namespace __sanitizer
