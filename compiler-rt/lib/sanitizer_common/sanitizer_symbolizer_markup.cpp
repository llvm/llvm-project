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
// This generic support for offline symbolizing is based on the
// Fuchsia port.  We don't do any actual symbolization per se.
// Instead, we emit text containing raw addresses and raw linkage
// symbol names, embedded in Fuchsia's symbolization markup format.
// See the spec at: https://llvm.org/docs/SymbolizerMarkupFormat.html
//===----------------------------------------------------------------------===//

#include "sanitizer_symbolizer_markup.h"

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_platform.h"
#include "sanitizer_symbolizer.h"
#include "sanitizer_symbolizer_markup_constants.h"

namespace __sanitizer {

void MarkupStackTracePrinter::RenderData(InternalScopedString *buffer,
                                         const char *format, const DataInfo *DI,
                                         const char *strip_path_prefix) {
  RenderContext(buffer);
  buffer->AppendF(kFormatData, DI->start);
}

bool MarkupStackTracePrinter::RenderNeedsSymbolization(const char *format) {
  return false;
}

void MarkupStackTracePrinter::RenderFrame(InternalScopedString *buffer,
                                          const char *format, int frame_no,
                                          uptr address, const AddressInfo *info,
                                          bool vs_style,
                                          const char *strip_path_prefix) {
  CHECK(!RenderNeedsSymbolization(format));
  RenderContext(buffer);
  buffer->AppendF(kFormatFrame, frame_no, address);
}

bool MarkupSymbolizerTool::SymbolizePC(uptr addr, SymbolizedStack *stack) {
  char buffer[kFormatFunctionMax];
  internal_snprintf(buffer, sizeof(buffer), kFormatFunction, addr);
  stack->info.function = internal_strdup(buffer);
  return true;
}

bool MarkupSymbolizerTool::SymbolizeData(uptr addr, DataInfo *info) {
  info->Clear();
  info->start = addr;
  return true;
}

// This is used by UBSan for type names, and by ASan for global variable names.
// It's expected to return a static buffer that will be reused on each call.
const char *MarkupSymbolizerTool::Demangle(const char *name) {
  static char buffer[kFormatDemangleMax];
  internal_snprintf(buffer, sizeof(buffer), kFormatDemangle, name);
  return buffer;
}

// Fuchsia's implementation of symbolizer markup doesn't need to emit contextual
// elements at this point.
// Fuchsia's logging infrastructure emits enough information about
// process memory layout that a post-processing filter can do the
// symbolization and pretty-print the markup.
#if !SANITIZER_SYMBOLIZER_MARKUP_FUCHSIA

// Simplier view of a LoadedModule. It only holds information necessary to
// identify unique modules.
struct RenderedModule {
  char *full_name;
  uptr base_address;
  u8 uuid[kModuleUUIDSize];  // BuildId
};

static bool ModulesEq(const LoadedModule &module,
                      const RenderedModule &renderedModule) {
  return module.base_address() == renderedModule.base_address &&
         internal_memcmp(module.uuid(), renderedModule.uuid,
                         module.uuid_size()) == 0 &&
         internal_strcmp(module.full_name(), renderedModule.full_name) == 0;
}

static bool ModuleHasBeenRendered(
    const LoadedModule &module,
    const InternalMmapVectorNoCtor<RenderedModule> &renderedModules) {
  for (const auto &renderedModule : renderedModules) {
    if (ModulesEq(module, renderedModule)) {
      return true;
    }
  }
  return false;
}

static void RenderModule(InternalScopedString *buffer,
                         const LoadedModule &module, uptr moduleId) {
  buffer->AppendF("{{{module:%d:%s:elf:", moduleId, module.full_name());
  for (uptr i = 0; i < module.uuid_size(); i++) {
    buffer->AppendF("%02x", module.uuid()[i]);
  }
  buffer->Append("}}}\n");
}

static void RenderMmaps(InternalScopedString *buffer,
                        const LoadedModule &module, uptr moduleId) {
  for (const auto &range : module.ranges()) {
    //{{{mmap:starting_addr:size_in_hex:load:module_Id:r(w|x):relative_addr}}}
    buffer->AppendF("{{{mmap:%p:%p:load:%d:", range.beg, range.end - range.beg,
                    moduleId);

    // All module mmaps are readable at least
    buffer->Append("r");
    if (range.writable)
      buffer->Append("w");
    if (range.executable)
      buffer->Append("x");

    // module.base_address == dlpi_addr
    // range.beg == dlpi_addr + p_vaddr
    // relative address == p_vaddr == range.beg - module.base_address
    buffer->AppendF(":%p}}}\n", range.beg - module.base_address());
  }
}

void MarkupStackTracePrinter::RenderContext(InternalScopedString *buffer) {
  // Keeps track of the modules that have been rendered.
  static bool initialized = false;
  static InternalMmapVectorNoCtor<RenderedModule> renderedModules;
  if (!initialized) {
    // arbitrary initial size, counting the main module plus some important libs
    // like libc.
    renderedModules.Initialize(3);
    initialized = true;
  }

  if (renderedModules.size() == 0) {
    buffer->Append("{{{reset}}}\n");
  }

  const auto &modules = Symbolizer::GetOrInit()->GetRefreshedListOfModules();

  for (const auto &module : modules) {
    if (ModuleHasBeenRendered(module, renderedModules)) {
      continue;
    }

    // symbolizer markup id, used to refer to this modules from other contextual
    // elements
    uptr moduleId = renderedModules.size();

    RenderModule(buffer, module, moduleId);
    RenderMmaps(buffer, module, moduleId);

    RenderedModule renderedModule{
        internal_strdup(module.full_name()), module.base_address(), {}};

    // kModuleUUIDSize is the size of curModule.uuid
    CHECK_GE(kModuleUUIDSize, module.uuid_size());
    internal_memcpy(renderedModule.uuid, module.uuid(), module.uuid_size());
    renderedModules.push_back(renderedModule);
  }
}
#endif  // !SANITIZER_SYMBOLIZER_MARKUP_FUCHSIA
}  // namespace __sanitizer
