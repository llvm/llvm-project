//===-- SBFileSpec.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFileSpec.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBStream.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Stream.h"

#include "llvm/ADT/SmallString.h"

#include <inttypes.h>
#include <limits.h>

using namespace lldb;
using namespace lldb_private;

SBFileSpec::SBFileSpec() : m_opaque_up(new lldb_private::FileSpec()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBFileSpec);
}

SBFileSpec::SBFileSpec(const SBFileSpec &rhs) : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBFileSpec, (const lldb::SBFileSpec &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBFileSpec::SBFileSpec(const lldb_private::FileSpec &fspec)
    : m_opaque_up(new lldb_private::FileSpec(fspec)) {}

// Deprecated!!!
SBFileSpec::SBFileSpec(const char *path) : m_opaque_up(new FileSpec(path)) {
  LLDB_RECORD_CONSTRUCTOR(SBFileSpec, (const char *), path);

  FileSystem::Instance().Resolve(*m_opaque_up);
}

SBFileSpec::SBFileSpec(const char *path, bool resolve)
    : m_opaque_up(new FileSpec(path)) {
  LLDB_RECORD_CONSTRUCTOR(SBFileSpec, (const char *, bool), path, resolve);

  if (resolve)
    FileSystem::Instance().Resolve(*m_opaque_up);
}

SBFileSpec::~SBFileSpec() = default;

const SBFileSpec &SBFileSpec::operator=(const SBFileSpec &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBFileSpec &,
                     SBFileSpec, operator=,(const lldb::SBFileSpec &), rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return LLDB_RECORD_RESULT(*this);
}

bool SBFileSpec::operator==(const SBFileSpec &rhs) const {
  LLDB_RECORD_METHOD_CONST(bool, SBFileSpec, operator==,(const SBFileSpec &rhs),
                           rhs);

  return ref() == rhs.ref();
}

bool SBFileSpec::operator!=(const SBFileSpec &rhs) const {
  LLDB_RECORD_METHOD_CONST(bool, SBFileSpec, operator!=,(const SBFileSpec &rhs),
                           rhs);

  return !(*this == rhs);
}

bool SBFileSpec::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBFileSpec, IsValid);
  return this->operator bool();
}
SBFileSpec::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBFileSpec, operator bool);

  return m_opaque_up->operator bool();
}

bool SBFileSpec::Exists() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBFileSpec, Exists);

  return FileSystem::Instance().Exists(*m_opaque_up);
}

bool SBFileSpec::ResolveExecutableLocation() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBFileSpec, ResolveExecutableLocation);

  return FileSystem::Instance().ResolveExecutableLocation(*m_opaque_up);
}

int SBFileSpec::ResolvePath(const char *src_path, char *dst_path,
                            size_t dst_len) {
  LLDB_RECORD_STATIC_METHOD(int, SBFileSpec, ResolvePath,
                            (const char *, char *, size_t), src_path, dst_path,
                            dst_len);

  llvm::SmallString<64> result(src_path);
  FileSystem::Instance().Resolve(result);
  ::snprintf(dst_path, dst_len, "%s", result.c_str());
  return std::min(dst_len - 1, result.size());
}

const char *SBFileSpec::GetFilename() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBFileSpec, GetFilename);

  return m_opaque_up->GetFilename().AsCString();
}

const char *SBFileSpec::GetDirectory() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBFileSpec, GetDirectory);

  FileSpec directory{*m_opaque_up};
  directory.GetFilename().Clear();
  return directory.GetCString();
}

void SBFileSpec::SetFilename(const char *filename) {
  LLDB_RECORD_METHOD(void, SBFileSpec, SetFilename, (const char *), filename);

  if (filename && filename[0])
    m_opaque_up->GetFilename().SetCString(filename);
  else
    m_opaque_up->GetFilename().Clear();
}

void SBFileSpec::SetDirectory(const char *directory) {
  LLDB_RECORD_METHOD(void, SBFileSpec, SetDirectory, (const char *), directory);

  if (directory && directory[0])
    m_opaque_up->GetDirectory().SetCString(directory);
  else
    m_opaque_up->GetDirectory().Clear();
}

uint32_t SBFileSpec::GetPath(char *dst_path, size_t dst_len) const {
  LLDB_RECORD_CHAR_PTR_METHOD_CONST(uint32_t, SBFileSpec, GetPath,
                                    (char *, size_t), dst_path, "", dst_len);

  uint32_t result = m_opaque_up->GetPath(dst_path, dst_len);

  if (result == 0 && dst_path && dst_len > 0)
    *dst_path = '\0';
  return result;
}

const lldb_private::FileSpec *SBFileSpec::operator->() const {
  return m_opaque_up.get();
}

const lldb_private::FileSpec *SBFileSpec::get() const {
  return m_opaque_up.get();
}

const lldb_private::FileSpec &SBFileSpec::operator*() const {
  return *m_opaque_up;
}

const lldb_private::FileSpec &SBFileSpec::ref() const { return *m_opaque_up; }

void SBFileSpec::SetFileSpec(const lldb_private::FileSpec &fs) {
  *m_opaque_up = fs;
}

bool SBFileSpec::GetDescription(SBStream &description) const {
  LLDB_RECORD_METHOD_CONST(bool, SBFileSpec, GetDescription, (lldb::SBStream &),
                           description);

  Stream &strm = description.ref();
  char path[PATH_MAX];
  if (m_opaque_up->GetPath(path, sizeof(path)))
    strm.PutCString(path);
  return true;
}

void SBFileSpec::AppendPathComponent(const char *fn) {
  LLDB_RECORD_METHOD(void, SBFileSpec, AppendPathComponent, (const char *), fn);

  m_opaque_up->AppendPathComponent(fn);
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBFileSpec>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBFileSpec, ());
  LLDB_REGISTER_CONSTRUCTOR(SBFileSpec, (const lldb::SBFileSpec &));
  LLDB_REGISTER_CONSTRUCTOR(SBFileSpec, (const char *));
  LLDB_REGISTER_CONSTRUCTOR(SBFileSpec, (const char *, bool));
  LLDB_REGISTER_METHOD(const lldb::SBFileSpec &,
                       SBFileSpec, operator=,(const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD_CONST(bool,
                             SBFileSpec, operator==,(const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD_CONST(bool,
                             SBFileSpec, operator!=,(const lldb::SBFileSpec &));
  LLDB_REGISTER_METHOD_CONST(bool, SBFileSpec, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBFileSpec, operator bool, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBFileSpec, Exists, ());
  LLDB_REGISTER_METHOD(bool, SBFileSpec, ResolveExecutableLocation, ());
  LLDB_REGISTER_STATIC_METHOD(int, SBFileSpec, ResolvePath,
                              (const char *, char *, size_t));
  LLDB_REGISTER_METHOD_CONST(const char *, SBFileSpec, GetFilename, ());
  LLDB_REGISTER_METHOD_CONST(const char *, SBFileSpec, GetDirectory, ());
  LLDB_REGISTER_METHOD(void, SBFileSpec, SetFilename, (const char *));
  LLDB_REGISTER_METHOD(void, SBFileSpec, SetDirectory, (const char *));
  LLDB_REGISTER_METHOD_CONST(bool, SBFileSpec, GetDescription,
                             (lldb::SBStream &));
  LLDB_REGISTER_METHOD(void, SBFileSpec, AppendPathComponent, (const char *));
  LLDB_REGISTER_CHAR_PTR_METHOD_CONST(uint32_t, SBFileSpec, GetPath);
}

}
}
