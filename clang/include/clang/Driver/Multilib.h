//===--- Multilib.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_MULTILIB_H
#define LLVM_CLANG_DRIVER_MULTILIB_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Option.h"
#include <functional>
#include <string>
#include <vector>

namespace clang {
namespace driver {

/// This corresponds to a single GCC Multilib, or a segment of one controlled
/// by a command line flag
class Multilib {
public:
  typedef std::vector<std::string> flags_list;

private:
  std::string GCCSuffix;
  std::string OSSuffix;
  std::string IncludeSuffix;
  flags_list Flags;

public:
  Multilib(StringRef GCCSuffix = "", StringRef OSSuffix = "",
           StringRef IncludeSuffix = "");

  /// \brief Get the detected GCC installation path suffix for the multi-arch
  /// target variant. Always starts with a '/', unless empty
  const std::string &gccSuffix() const {
    assert(GCCSuffix.empty() ||
           (StringRef(GCCSuffix).front() == '/' && GCCSuffix.size() > 1));
    return GCCSuffix;
  }
  /// Set the GCC installation path suffix.
  Multilib &gccSuffix(StringRef S);

  /// \brief Get the detected os path suffix for the multi-arch
  /// target variant. Always starts with a '/', unless empty
  const std::string &osSuffix() const {
    assert(OSSuffix.empty() ||
           (StringRef(OSSuffix).front() == '/' && OSSuffix.size() > 1));
    return OSSuffix;
  }
  /// Set the os path suffix.
  Multilib &osSuffix(StringRef S);

  /// \brief Get the include directory suffix. Always starts with a '/', unless
  /// empty
  const std::string &includeSuffix() const {
    assert(IncludeSuffix.empty() ||
           (StringRef(IncludeSuffix).front() == '/' && IncludeSuffix.size() > 1));
    return IncludeSuffix;
  }
  /// Set the include directory suffix
  Multilib &includeSuffix(StringRef S);

  /// \brief Get the flags that indicate or contraindicate this multilib's use
  /// All elements begin with either '+' or '-'
  const flags_list &flags() const { return Flags; }
  flags_list &flags() { return Flags; }
  /// Add a flag to the flags list
  Multilib &flag(StringRef F) {
    assert(F.front() == '+' || F.front() == '-');
    Flags.push_back(F);
    return *this;
  }

  /// \brief print summary of the Multilib
  void print(raw_ostream &OS) const;

  /// Check whether any of the 'against' flags contradict the 'for' flags.
  bool isValid() const;

  /// Check whether the default is selected
  bool isDefault() const
  { return GCCSuffix.empty() && OSSuffix.empty() && IncludeSuffix.empty(); }

  bool operator==(const Multilib &Other) const;
};

raw_ostream &operator<<(raw_ostream &OS, const Multilib &M);

class MultilibSet {
public:
  typedef std::vector<Multilib> multilib_list;
  typedef multilib_list::iterator iterator;
  typedef multilib_list::const_iterator const_iterator;

  typedef std::function<std::vector<std::string>(const Multilib &M)>
      IncludeDirsFunc;

  typedef llvm::function_ref<bool(const Multilib &)> FilterCallback;

private:
  multilib_list Multilibs;
  IncludeDirsFunc IncludeCallback;
  IncludeDirsFunc FilePathsCallback;

public:
  MultilibSet() {}

  /// Add an optional Multilib segment
  MultilibSet &Maybe(const Multilib &M);

  /// Add a set of mutually incompatible Multilib segments
  MultilibSet &Either(const Multilib &M1, const Multilib &M2);
  MultilibSet &Either(const Multilib &M1, const Multilib &M2,
                      const Multilib &M3);
  MultilibSet &Either(const Multilib &M1, const Multilib &M2,
                      const Multilib &M3, const Multilib &M4);
  MultilibSet &Either(const Multilib &M1, const Multilib &M2,
                      const Multilib &M3, const Multilib &M4,
                      const Multilib &M5);
  MultilibSet &Either(ArrayRef<Multilib> Ms);

  /// Filter out some subset of the Multilibs using a user defined callback
  MultilibSet &FilterOut(FilterCallback F);
  /// Filter out those Multilibs whose gccSuffix matches the given expression
  MultilibSet &FilterOut(const char *Regex);

  /// Add a completed Multilib to the set
  void push_back(const Multilib &M);

  /// Union this set of multilibs with another
  void combineWith(const MultilibSet &MS);

  /// Remove all of thie multilibs from the set
  void clear() { Multilibs.clear(); }

  iterator begin() { return Multilibs.begin(); }
  const_iterator begin() const { return Multilibs.begin(); }

  iterator end() { return Multilibs.end(); }
  const_iterator end() const { return Multilibs.end(); }

  /// Pick the best multilib in the set, \returns false if none are compatible
  bool select(const Multilib::flags_list &Flags, Multilib &M) const;

  unsigned size() const { return Multilibs.size(); }

  void print(raw_ostream &OS) const;

  MultilibSet &setIncludeDirsCallback(IncludeDirsFunc F) {
    IncludeCallback = std::move(F);
    return *this;
  }
  const IncludeDirsFunc &includeDirsCallback() const { return IncludeCallback; }

  MultilibSet &setFilePathsCallback(IncludeDirsFunc F) {
    FilePathsCallback = std::move(F);
    return *this;
  }
  const IncludeDirsFunc &filePathsCallback() const { return FilePathsCallback; }

private:
  /// Apply the filter to Multilibs and return the subset that remains
  static multilib_list filterCopy(FilterCallback F, const multilib_list &Ms);

  /// Apply the filter to the multilib_list, removing those that don't match
  static void filterInPlace(FilterCallback F, multilib_list &Ms);
};

raw_ostream &operator<<(raw_ostream &OS, const MultilibSet &MS);
}
}

#endif

