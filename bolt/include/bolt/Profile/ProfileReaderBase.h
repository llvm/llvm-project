//===- bolt/Profile/ProfileReaderBase.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Interface to be implemented by all profile readers.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_PROFILE_READER_BASE_H
#define BOLT_PROFILE_PROFILE_READER_BASE_H

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace bolt {

class BinaryContext;
class BinaryFunction;
class BoltAddressTranslation;

/// LTO-generated function names take a form:
///
///   <function_name>.lto_priv.<decimal_number>/...
///     or
///   <function_name>.constprop.<decimal_number>/...
///
/// they can also be:
///
///   <function_name>.lto_priv.<decimal_number1>.lto_priv.<decimal_number2>/...
///
/// The <decimal_number> is a global counter used for the whole program. As a
/// result, a tiny change in a program may affect the naming of many LTO
/// functions. For us this means that if we do a precise name matching, then
/// a large set of functions could be left without a profile.
///
/// To solve this issue, we try to match a function to any profile:
///
///   <function_name>.(lto_priv|consprop).*
///
/// The name before an asterisk above represents a common LTO name for a family
/// of functions. Later, out of all matching profiles we pick the one with the
/// best match.
///
/// Return a common part of LTO name for a given \p Name.
Optional<StringRef> getLTOCommonName(const StringRef Name);

class ProfileReaderBase {
protected:
  /// Name of the file with profile.
  std::string Filename;

public:
  ProfileReaderBase() = delete;
  ProfileReaderBase(const ProfileReaderBase &) = delete;
  ProfileReaderBase &operator=(const ProfileReaderBase &) = delete;
  ProfileReaderBase(ProfileReaderBase &&) = delete;
  ProfileReaderBase &operator=(ProfileReaderBase &&) = delete;

  /// Construct a reader for a given file.
  explicit ProfileReaderBase(StringRef Filename) : Filename(Filename) {}

  virtual ~ProfileReaderBase() = default;

  /// Return the name of the file containing the profile.
  StringRef getFilename() const { return Filename; }

  /// Instruct the profiler to use address-translation tables.
  virtual void setBAT(BoltAddressTranslation *BAT) {}

  /// Pre-process the profile when functions in \p BC are discovered,
  /// but not yet disassembled. Once the profile is pre-processed, calls to
  /// mayHaveProfileData() should be able to identify if the function possibly
  /// has a profile available.
  virtual Error preprocessProfile(BinaryContext &BC) = 0;

  /// Assign profile to all objects in the \p BC while functions are
  /// in pre-CFG state with instruction addresses available.
  virtual Error readProfilePreCFG(BinaryContext &BC) = 0;

  /// Assign profile to all objects in the \p BC.
  virtual Error readProfile(BinaryContext &BC) = 0;

  /// Return the string identifying the reader.
  virtual StringRef getReaderName() const = 0;

  /// Return true if the function \p BF may have a profile available.
  /// The result is based on the name(s) of the function alone and the profile
  /// match is not guaranteed.
  virtual bool mayHaveProfileData(const BinaryFunction &BF);

  /// Return true if the profile contains an entry for a local object
  /// that has an associated file name.
  virtual bool hasLocalsWithFileName() const { return true; }

  /// Return all event names used to collect this profile.
  virtual StringSet<> getEventNames() const { return StringSet<>(); }

  /// Return true if the source of the profile should be trusted. E.g., even
  /// good source of profile data may contain discrepancies. Nevertheless, the
  /// rest of the profile is correct.
  virtual bool isTrustedSource() const = 0;
};

} // namespace bolt
} // namespace llvm

#endif
