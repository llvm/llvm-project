//===-- SBReproducer.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBReproducer.h"
#include "lldb/Utility/Instrumentation.h"

using namespace lldb;
using namespace lldb_private;

SBReplayOptions::SBReplayOptions() = default;

SBReplayOptions::SBReplayOptions(const SBReplayOptions &rhs) = default;

SBReplayOptions::~SBReplayOptions() = default;

SBReplayOptions &SBReplayOptions::operator=(const SBReplayOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs)
  return *this;
}

void SBReplayOptions::SetVerify(bool verify) {
  LLDB_INSTRUMENT_VA(this, verify);
}

bool SBReplayOptions::GetVerify() const {
  LLDB_INSTRUMENT_VA(this);
  return false;
}

void SBReplayOptions::SetCheckVersion(bool check) {
  LLDB_INSTRUMENT_VA(this, check);
}

bool SBReplayOptions::GetCheckVersion() const {
  LLDB_INSTRUMENT_VA(this);
  return false;
}

const char *SBReproducer::Capture() {
  LLDB_INSTRUMENT()
  return "Reproducer capture has been removed";
}

const char *SBReproducer::Capture(const char *path) {
  LLDB_INSTRUMENT_VA(path)
  return "Reproducer capture has been removed";
}

const char *SBReproducer::PassiveReplay(const char *path) {
  LLDB_INSTRUMENT_VA(path)
  return "Reproducer replay has been removed";
}

const char *SBReproducer::Replay(const char *path) {
  LLDB_INSTRUMENT_VA(path)
  return "Reproducer replay has been removed";
}

const char *SBReproducer::Replay(const char *path, bool skip_version_check) {
  LLDB_INSTRUMENT_VA(path, skip_version_check)
  return "Reproducer replay has been removed";
}

const char *SBReproducer::Replay(const char *path,
                                 const SBReplayOptions &options) {
  LLDB_INSTRUMENT_VA(path, options)
  return "Reproducer replay has been removed";
}

const char *SBReproducer::Finalize(const char *path) {
  LLDB_INSTRUMENT_VA(path)
  return "Reproducer finalize has been removed";
}

bool SBReproducer::Generate() {
  LLDB_INSTRUMENT()
  return false;
}

bool SBReproducer::SetAutoGenerate(bool b) {
  LLDB_INSTRUMENT_VA(b)
  return false;
}

const char *SBReproducer::GetPath() {
  LLDB_INSTRUMENT()
  return "Reproducer GetPath has been removed";
}

void SBReproducer::SetWorkingDirectory(const char *path) {
  LLDB_INSTRUMENT_VA(path)
}
