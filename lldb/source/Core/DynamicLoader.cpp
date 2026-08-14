//===-- DynamicLoader.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/DynamicLoader.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Progress.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolLocator.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-private-interfaces.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

#include <cassert>

using namespace lldb;
using namespace lldb_private;

DynamicLoader *DynamicLoader::FindPlugin(Process *process,
                                         llvm::StringRef plugin_name) {
  DynamicLoaderCreateInstance create_callback = nullptr;
  if (!plugin_name.empty()) {
    create_callback =
        PluginManager::GetDynamicLoaderCreateCallbackForPluginName(plugin_name);
    if (create_callback) {
      std::unique_ptr<DynamicLoader> instance_up(
          create_callback(process, true));
      if (instance_up)
        return instance_up.release();
    }
  } else {
    for (auto create_callback :
         PluginManager::GetDynamicLoaderCreateCallbacks()) {
      std::unique_ptr<DynamicLoader> instance_up(
          create_callback(process, false));
      if (instance_up)
        return instance_up.release();
    }
  }
  return nullptr;
}

DynamicLoader::DynamicLoader(Process *process) : m_process(process) {}

// Accessors to the global setting as to whether to stop at image (shared
// library) loading/unloading.

bool DynamicLoader::GetStopWhenImagesChange() const {
  return m_process->GetStopOnSharedLibraryEvents();
}

void DynamicLoader::SetStopWhenImagesChange(bool stop) {
  m_process->SetStopOnSharedLibraryEvents(stop);
}

ModuleSP DynamicLoader::GetTargetExecutable() {
  Target &target = m_process->GetTarget();
  ModuleSP executable = target.GetExecutableModule();

  if (executable) {
    if (FileSystem::Instance().Exists(executable->GetFileSpec())) {
      ModuleSpec module_spec(executable->GetFileSpec(),
                             executable->GetArchitecture());
      auto module_sp = std::make_shared<Module>(module_spec);
      // If we're a coredump and we already have a main executable, we don't
      // need to reload the module list that target already has
      if (!m_process->IsLiveDebugSession()) {
        return executable;
      }
      // Check if the executable has changed and set it to the target
      // executable if they differ.
      if (module_sp && module_sp->GetUUID().IsValid() &&
          executable->GetUUID().IsValid()) {
        if (module_sp->GetUUID() != executable->GetUUID())
          executable.reset();
      } else if (executable->FileHasChanged()) {
        executable.reset();
      }

      if (!executable) {
        executable = target.GetOrCreateModule(module_spec, true /* notify */);
        if (executable.get() != target.GetExecutableModulePointer()) {
          // Don't load dependent images since we are in dyld where we will
          // know and find out about all images that are loaded
          target.SetExecutableModule(executable, eLoadDependentsNo);
        }
      }
    }
  }
  return executable;
}

void DynamicLoader::UpdateLoadedSections(ModuleSP module, addr_t link_map_addr,
                                         addr_t base_addr,
                                         bool base_addr_is_offset) {
  UpdateLoadedSectionsCommon(module, base_addr, base_addr_is_offset);
}

void DynamicLoader::UpdateLoadedSectionsCommon(ModuleSP module,
                                               addr_t base_addr,
                                               bool base_addr_is_offset) {
  bool changed;
  module->SetLoadAddress(m_process->GetTarget(), base_addr, base_addr_is_offset,
                         changed);
}

void DynamicLoader::UnloadSections(const ModuleSP module) {
  UnloadSectionsCommon(module);
}

void DynamicLoader::UnloadSectionsCommon(const ModuleSP module) {
  Target &target = m_process->GetTarget();
  const SectionList *sections = GetSectionListFromModule(module);

  assert(sections && "SectionList missing from unloaded module.");

  const size_t num_sections = sections->GetSize();
  for (size_t i = 0; i < num_sections; ++i) {
    SectionSP section_sp(sections->GetSectionAtIndex(i));
    target.SetSectionUnloaded(section_sp);
  }
}

const SectionList *
DynamicLoader::GetSectionListFromModule(const ModuleSP module) const {
  SectionList *sections = nullptr;
  if (module) {
    ObjectFile *obj_file = module->GetObjectFile();
    if (obj_file != nullptr) {
      sections = obj_file->GetSectionList();
    }
  }
  return sections;
}

ModuleSP DynamicLoader::FindModuleViaTarget(const ModuleSpec &spec) {
  ModuleSpec module_spec(spec);
  Target &target = m_process->GetTarget();
  // The process may be able to augment the module_spec with a UUID.
  if (!module_spec.GetUUID().IsValid())
    m_process->FindModuleUUID(module_spec);
  if (ModuleSP module_sp = target.GetImages().FindFirstModule(module_spec))
    return module_sp;

  if (ModuleSP module_sp =
          target.GetOrCreateModule(module_spec, /*notify=*/false))
    return module_sp;

  return nullptr;
}

ModuleSP DynamicLoader::LoadModuleAtAddress(const FileSpec &file,
                                            addr_t link_map_addr,
                                            addr_t base_addr,
                                            bool base_addr_is_offset) {
  Target &target = m_process->GetTarget();
  ModuleSpec module_spec(file, target.GetArchitecture());
  module_spec.SetLoadAddress(base_addr);
  ModuleSP module_sp = FindModuleViaTarget(module_spec);
  // We have a core file, try to load the image from memory if we didn't find
  // the module.
  if (!module_sp && !m_process->IsLiveDebugSession()) {
    llvm::Expected<ModuleSP> memory_module_sp_or_err =
        m_process->ReadModuleFromMemory(file, base_addr);
    if (auto err = memory_module_sp_or_err.takeError())
      LLDB_LOG_ERROR(GetLog(LLDBLog::DynamicLoader), std::move(err),
                     "Failed to read module from memory: {0}");
    else {
      module_sp = *memory_module_sp_or_err;
      m_process->GetTarget().GetImages().AppendIfNeeded(module_sp, false);
    }
  }
  if (module_sp)
    UpdateLoadedSections(module_sp, link_map_addr, base_addr,
                         base_addr_is_offset);
  return module_sp;
}

static ModuleSP ReadUnnamedMemoryModule(Process *process, addr_t addr,
                                        llvm::StringRef name) {
  char namebuf[80];
  if (name.empty()) {
    snprintf(namebuf, sizeof(namebuf), "memory-image-0x%" PRIx64, addr);
    name = namebuf;
  }
  llvm::Expected<ModuleSP> module_sp_or_err =
      process->ReadModuleFromMemory(FileSpec(name), addr);
  if (auto err = module_sp_or_err.takeError()) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::DynamicLoader), std::move(err),
                   "Failed to read module from memory: {0}");
    return {};
  }
  return *module_sp_or_err;
}

static std::string
GetBinaryDescription(const DynamicLoader::BinarySpec &bin_spec) {
  StreamString desc;
  if (!bin_spec.name.empty())
    desc << bin_spec.name << " ";
  if (bin_spec.uuid.IsValid())
    desc << bin_spec.uuid.GetAsString();
  if (!bin_spec.value_is_offset && bin_spec.value != LLDB_INVALID_ADDRESS) {
    desc << " at 0x";
    desc.PutHex64(bin_spec.value);
  }
  return desc.GetString().str();
}

static std::string
GetBinaryNotFoundMessage(const DynamicLoader::BinarySpec &bin_spec) {
  StreamString msg;
  msg << "Unable to find file";
  if (!bin_spec.name.empty())
    msg << " " << bin_spec.name;
  if (bin_spec.uuid.IsValid())
    msg << " with UUID " << bin_spec.uuid.GetAsString();
  if (bin_spec.value != LLDB_INVALID_ADDRESS) {
    if (bin_spec.value_is_offset)
      msg.Printf(" with slide 0x%" PRIx64, bin_spec.value);
    else
      msg.Printf(" at address 0x%" PRIx64, bin_spec.value);
  }
  return msg.GetString().str();
}

/// Search for a binary with a known UUID, and create a module for it.
///
/// Does not mutate the Target, but does read from it, and reaches the global
/// shared module list, the symbol locator plugins, and a locate module callback
/// the user may have installed.
static void SearchForBinary(Target &target, DynamicLoader::BinarySpec &bin_spec,
                            const FileSpecList &search_paths) {
  ModuleSpec module_spec;
  module_spec.SetTarget(target.shared_from_this());
  module_spec.GetUUID() = bin_spec.uuid;
  FileSpec name_filespec(bin_spec.name);
  if (FileSystem::Instance().Exists(name_filespec))
    module_spec.GetFileSpec() = name_filespec;

  // Has lldb already seen a module with this UUID? A module whose symbols are
  // already in hand is the answer, and searching would only find them again.
  // Without them the search still has something to add.
  ModuleList::GetSharedModule(module_spec, bin_spec.module_sp, nullptr, nullptr,
                              /*invoke_locate_callback=*/true,
                              /*invoke_symbol_locators=*/false);
  if (bin_spec.module_sp && bin_spec.module_sp->GetSymbolFileFileSpec())
    return;

  // Search for the binary and its symbols.
  SymbolLocator::Request request;
  request.module_spec = module_spec;
  request.platform = target.GetPlatform();
  request.external_lookup = bin_spec.force_symbol_search;

  llvm::Expected<SymbolLocator::Result> located =
      SymbolLocator::Locate(request, search_paths);
  if (!located) {
    // This function's caller names the binary it could not find, so a plain
    // miss needs nothing added to it. An explanation from a symbol server does.
    llvm::Error error = located.takeError();
    if (error.isA<SymbolLocator::NotFound>())
      llvm::consumeError(std::move(error));
    else
      bin_spec.error = Status::FromError(std::move(error));
    return;
  }

  // A binary was found. Its symbols are another matter, and the caller reports
  // that in its own order.
  if (located->symbol_error)
    bin_spec.error = Status::FromError(std::move(*located->symbol_error));

  // Create a module for what was found, sharing it with any other Target that
  // asks for the same binary. The module is not registered with this Target
  // until LoadBinaryInTarget. The locators have run, so don't run them again.
  ModuleSP located_module_sp;
  ModuleList::GetSharedModule(located->module_spec, located_module_sp, nullptr,
                              nullptr, /*invoke_locate_callback=*/false,
                              /*invoke_symbol_locators=*/false);

  // A located binary always yields a module, whatever ObjectFile makes of the
  // file, because the caller has nowhere else to record what the search found.
  if (!located_module_sp)
    located_module_sp = std::make_shared<Module>(located->module_spec);

  // Published only now, so that a search that came up empty leaves whatever the
  // shared module list had in hand.
  bin_spec.module_sp = std::move(located_module_sp);
  bin_spec.module_sp->GetSymbolLocatorStatistics().merge(located->statistics);
}

static void FindBinaryUUIDInMemory(Process *process,
                                   DynamicLoader::BinarySpec &bin_spec) {
  bin_spec.memory_module_sp =
      ReadUnnamedMemoryModule(process, bin_spec.value, bin_spec.name);
  if (bin_spec.memory_module_sp)
    bin_spec.uuid = bin_spec.memory_module_sp->GetUUID();
}

void DynamicLoader::LocateBinaries(
    Process *process, llvm::MutableArrayRef<BinarySpec> bin_specs) {
  Target &target = process->GetTarget();
  const FileSpecList search_paths = Target::GetDefaultDebugFileSearchPaths();

  for (BinarySpec &bin_spec : bin_specs) {
    if (!bin_spec.uuid.IsValid() && !bin_spec.value_is_offset)
      FindBinaryUUIDInMemory(process, bin_spec);
    if (!bin_spec.uuid.IsValid())
      continue;
    Progress progress("Locating binary", GetBinaryDescription(bin_spec));
    SearchForBinary(target, bin_spec, search_paths);
  }
}

llvm::Expected<ModuleSP>
DynamicLoader::LoadBinaryInTarget(Process *process, BinarySpec &bin_spec) {
  Target &target = process->GetTarget();

  // The error belongs to this function now: every path below either reports it
  // or folds it into the failure.
  llvm::Error search_error = bin_spec.error.takeError();

  // If we couldn't find the binary anywhere else, as a last resort,
  // read it out of memory.
  if (bin_spec.allow_memory_image_last_resort && !bin_spec.module_sp &&
      bin_spec.value != LLDB_INVALID_ADDRESS && !bin_spec.value_is_offset) {
    if (!bin_spec.memory_module_sp)
      bin_spec.memory_module_sp =
          ReadUnnamedMemoryModule(process, bin_spec.value, bin_spec.name);
    if (bin_spec.memory_module_sp)
      bin_spec.module_sp = bin_spec.memory_module_sp;
  }

  Log *log = GetLog(LLDBLog::DynamicLoader);
  if (!bin_spec.module_sp) {
    std::string message = GetBinaryNotFoundMessage(bin_spec);
    LLDB_LOG(log, "{0}", message);
    llvm::Error error = llvm::createStringError(message);
    if (search_error)
      return llvm::joinErrors(std::move(search_error), std::move(error));
    return std::move(error);
  }

  // A binary was found, but a symbol server may still have had something to say
  // about its symbols.  Name the binary: a symbol locator's error is not
  // required to identify what it was asked to look for.
  if (search_error)
    *target.GetDebugger().GetAsyncErrorStream()
        << GetBinaryDescription(bin_spec) << ": "
        << llvm::toString(std::move(search_error)) << "\n";

  // Ensure the Target has an architecture set in case
  // we need it while processing this binary/eh_frame/debug info.
  if (!target.GetArchitecture().IsValid())
    target.SetArchitecture(bin_spec.module_sp->GetArchitecture());
  target.GetImages().AppendIfNeeded(bin_spec.module_sp, false);

  bool changed = false;
  if (bin_spec.set_address_in_target) {
    if (bin_spec.module_sp->GetObjectFile()) {
      if (bin_spec.value != LLDB_INVALID_ADDRESS) {
        LLDB_LOGF(log,
                  "DynamicLoader::LoadBinaryInTarget Loading "
                  "binary %s UUID %s at %s 0x%" PRIx64,
                  bin_spec.name.c_str(), bin_spec.uuid.GetAsString().c_str(),
                  bin_spec.value_is_offset ? "offset" : "address",
                  bin_spec.value);
        bin_spec.module_sp->SetLoadAddress(target, bin_spec.value,
                                           bin_spec.value_is_offset, changed);
      } else {
        // No address/offset/slide, load the binary at file address,
        // offset 0.
        LLDB_LOGF(log,
                  "DynamicLoader::LoadBinaryInTarget Loading "
                  "binary %s UUID %s at file address",
                  bin_spec.name.c_str(), bin_spec.uuid.GetAsString().c_str());
        bin_spec.module_sp->SetLoadAddress(target, 0, true /* value_is_slide */,
                                           changed);
      }
    } else {
      // In-memory image, load at its true address, offset 0.
      LLDB_LOGF(log,
                "DynamicLoader::LoadBinaryInTarget Loading binary "
                "%s UUID %s from memory at address 0x%" PRIx64,
                bin_spec.name.c_str(), bin_spec.uuid.GetAsString().c_str(),
                bin_spec.value);
      bin_spec.module_sp->SetLoadAddress(target, 0, true /* value_is_slide */,
                                         changed);
    }
  }

  if (bin_spec.notify) {
    ModuleList added_module;
    added_module.Append(bin_spec.module_sp, false);
    target.ModulesDidLoad(added_module);
  }

  return bin_spec.module_sp;
}

llvm::Expected<ModuleSP>
DynamicLoader::LocateAndLoadBinary(Process *process, BinarySpec &bin_spec) {
  LocateBinaries(process, bin_spec);
  return LoadBinaryInTarget(process, bin_spec);
}

int64_t DynamicLoader::ReadUnsignedIntWithSizeInBytes(addr_t addr,
                                                      int size_in_bytes) {
  Status error;
  uint64_t value =
      m_process->ReadUnsignedIntegerFromMemory(addr, size_in_bytes, 0, error);
  if (error.Fail())
    return -1;
  else
    return (int64_t)value;
}

addr_t DynamicLoader::ReadPointer(addr_t addr) {
  Status error;
  addr_t value = m_process->ReadPointerFromMemory(addr, error);
  if (error.Fail())
    return LLDB_INVALID_ADDRESS;
  else
    return value;
}

void DynamicLoader::LoadOperatingSystemPlugin(bool flush)
{
    if (m_process)
        m_process->LoadOperatingSystemPlugin(flush);
}
