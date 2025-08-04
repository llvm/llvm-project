//===-- SBSaveCoreOptions.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSaveCoreOptions.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/SaveCoreOptions.h"
#include "lldb/Target/ThreadCollection.h"
#include "lldb/Utility/Instrumentation.h"

#include "Utils.h"

using namespace lldb;

SBSaveCoreOptions::SBSaveCoreOptions() {
  LLDB_INSTRUMENT_VA(this)

  m_opaque_up = std::make_unique<lldb_private::SaveCoreOptions>();
}

SBSaveCoreOptions::SBSaveCoreOptions(const SBSaveCoreOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBSaveCoreOptions::~SBSaveCoreOptions() = default;

const SBSaveCoreOptions &
SBSaveCoreOptions::operator=(const SBSaveCoreOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

SBError SBSaveCoreOptions::SetPluginName(const char *name) {
  LLDB_INSTRUMENT_VA(this, name);
  return SBError(m_opaque_up->SetPluginName(name));
}

void SBSaveCoreOptions::SetStyle(lldb::SaveCoreStyle style) {
  LLDB_INSTRUMENT_VA(this, style);
  m_opaque_up->SetStyle(style);
}

void SBSaveCoreOptions::SetOutputFile(lldb::SBFileSpec file_spec) {
  LLDB_INSTRUMENT_VA(this, file_spec);
  m_opaque_up->SetOutputFile(file_spec.ref());
}

const char *SBSaveCoreOptions::GetPluginName() const {
  LLDB_INSTRUMENT_VA(this);
  const auto name = m_opaque_up->GetPluginName();
  if (!name)
    return nullptr;
  return lldb_private::ConstString(name.value()).GetCString();
}

SBFileSpec SBSaveCoreOptions::GetOutputFile() const {
  LLDB_INSTRUMENT_VA(this);
  const auto file_spec = m_opaque_up->GetOutputFile();
  if (file_spec)
    return SBFileSpec(file_spec.value());
  return SBFileSpec();
}

lldb::SaveCoreStyle SBSaveCoreOptions::GetStyle() const {
  LLDB_INSTRUMENT_VA(this);
  return m_opaque_up->GetStyle();
}

SBError SBSaveCoreOptions::SetProcess(lldb::SBProcess process) {
  LLDB_INSTRUMENT_VA(this, process);
  return m_opaque_up->SetProcess(process.GetSP());
}

SBProcess SBSaveCoreOptions::GetProcess() {
  LLDB_INSTRUMENT_VA(this);
  return SBProcess(m_opaque_up->GetProcess());
}

SBError SBSaveCoreOptions::AddThread(lldb::SBThread thread) {
  LLDB_INSTRUMENT_VA(this, thread);
  return m_opaque_up->AddThread(thread.GetSP());
}

bool SBSaveCoreOptions::RemoveThread(lldb::SBThread thread) {
  LLDB_INSTRUMENT_VA(this, thread);
  return m_opaque_up->RemoveThread(thread.GetSP());
}

lldb::SBError
SBSaveCoreOptions::AddMemoryRegionToSave(const SBMemoryRegionInfo &region) {
  LLDB_INSTRUMENT_VA(this, region);
  // Currently add memory region can't fail, so we always return a success
  // SBerror, but because these API's live forever, this is the most future
  // proof thing to do.
  m_opaque_up->AddMemoryRegionToSave(region.ref());
  return SBError();
}

lldb::SBThreadCollection SBSaveCoreOptions::GetThreadsToSave() const {
  LLDB_INSTRUMENT_VA(this);
  lldb::ThreadCollectionSP threadcollection_sp =
      std::make_shared<lldb_private::ThreadCollection>(
          m_opaque_up->GetThreadsToSave());
  return SBThreadCollection(threadcollection_sp);
}

void SBSaveCoreOptions::Clear() {
  LLDB_INSTRUMENT_VA(this);
  m_opaque_up->Clear();
}

uint64_t SBSaveCoreOptions::GetCurrentSizeInBytes(SBError &error) {
  LLDB_INSTRUMENT_VA(this, error);
  llvm::Expected<uint64_t> expected_bytes =
      m_opaque_up->GetCurrentSizeInBytes();
  if (!expected_bytes) {
    error =
        SBError(lldb_private::Status::FromError(expected_bytes.takeError()));
    return 0;
  }
  // Clear the error, so if the clearer uses it we set it to success.
  error.Clear();
  return *expected_bytes;
}

lldb::SBMemoryRegionInfoList SBSaveCoreOptions::GetMemoryRegionsToSave() {
  LLDB_INSTRUMENT_VA(this);
  llvm::Expected<lldb_private::CoreFileMemoryRanges> memory_ranges =
      m_opaque_up->GetMemoryRegionsToSave();
  if (!memory_ranges) {
    llvm::consumeError(memory_ranges.takeError());
    return SBMemoryRegionInfoList();
  }

  SBMemoryRegionInfoList memory_region_infos;
  for (const auto &range : *memory_ranges) {
    SBMemoryRegionInfo region_info(
        nullptr, range.GetRangeBase(), range.GetRangeEnd(),
        range.data.lldb_permissions, /*mapped=*/true);
    memory_region_infos.Append(region_info);
  }

  return memory_region_infos;
}

lldb_private::SaveCoreOptions &SBSaveCoreOptions::ref() const {
  return *m_opaque_up;
}
