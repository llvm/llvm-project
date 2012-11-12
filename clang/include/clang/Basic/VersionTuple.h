//===- VersionTuple.h - Version Number Handling -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::VersionTuple class, which represents a version in
/// the form major[.minor[.subminor]].
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_VERSIONTUPLE_H
#define LLVM_CLANG_BASIC_VERSIONTUPLE_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/Optional.h"
#include <string>

namespace clang {

/// \brief Represents a version number in the form major[.minor[.subminor]].
class VersionTuple {
  unsigned Major;
  unsigned Minor : 31;
  unsigned Subminor : 31;
  unsigned HasMinor : 1;
  unsigned HasSubminor : 1;

public:
  VersionTuple() 
    : Major(0), Minor(0), Subminor(0), HasMinor(false), HasSubminor(false) { }

  explicit VersionTuple(unsigned Major)
    : Major(Major), Minor(0), Subminor(0), HasMinor(false), HasSubminor(false)
  { }

  explicit VersionTuple(unsigned Major, unsigned Minor)
    : Major(Major), Minor(Minor), Subminor(0), HasMinor(true), 
      HasSubminor(false)
  { }

  explicit VersionTuple(unsigned Major, unsigned Minor, unsigned Subminor)
    : Major(Major), Minor(Minor), Subminor(Subminor), HasMinor(true), 
      HasSubminor(true)
  { }

  /// \brief Determine whether this version information is empty
  /// (e.g., all version components are zero).
  bool empty() const { return Major == 0 && Minor == 0 && Subminor == 0; }

  /// \brief Retrieve the major version number.
  unsigned getMajor() const { return Major; }

  /// \brief Retrieve the minor version number, if provided.
  llvm::Optional<unsigned> getMinor() const { 
    if (!HasMinor)
      return llvm::Optional<unsigned>();
    return Minor;
  }

  /// \brief Retrieve the subminor version number, if provided.
  llvm::Optional<unsigned> getSubminor() const { 
    if (!HasSubminor)
      return llvm::Optional<unsigned>();
    return Subminor;
  }

  /// \brief Determine if two version numbers are equivalent. If not
  /// provided, minor and subminor version numbers are considered to be zero.
  friend bool operator==(const VersionTuple& X, const VersionTuple &Y) {
    return X.Major == Y.Major && X.Minor == Y.Minor && X.Subminor == Y.Subminor;
  }

  /// \brief Determine if two version numbers are not equivalent.
  ///
  /// If not provided, minor and subminor version numbers are considered to be 
  /// zero.
  friend bool operator!=(const VersionTuple &X, const VersionTuple &Y) {
    return !(X == Y);
  }

  /// \brief Determine whether one version number precedes another.
  ///
  /// If not provided, minor and subminor version numbers are considered to be
  /// zero.
  friend bool operator<(const VersionTuple &X, const VersionTuple &Y) {
    if (X.Major != Y.Major)
      return X.Major < Y.Major;

    if (X.Minor != Y.Minor)
      return X.Minor < Y.Minor;

    return X.Subminor < Y.Subminor;
  }

  /// \brief Determine whether one version number follows another.
  ///
  /// If not provided, minor and subminor version numbers are considered to be
  /// zero.
  friend bool operator>(const VersionTuple &X, const VersionTuple &Y) {
    return Y < X;
  }

  /// \brief Determine whether one version number precedes or is
  /// equivalent to another. 
  ///
  /// If not provided, minor and subminor version numbers are considered to be
  /// zero.
  friend bool operator<=(const VersionTuple &X, const VersionTuple &Y) {
    return !(Y < X);
  }

  /// \brief Determine whether one version number follows or is
  /// equivalent to another.
  ///
  /// If not provided, minor and subminor version numbers are considered to be
  /// zero.
  friend bool operator>=(const VersionTuple &X, const VersionTuple &Y) {
    return !(X < Y);
  }

  /// \brief Retrieve a string representation of the version number.
  std::string getAsString() const;

  /// \brief Try to parse the given string as a version number.
  /// \returns \c true if the string does not match the regular expression
  ///   [0-9]+(\.[0-9]+(\.[0-9]+))
  bool tryParse(StringRef string);
};

/// \brief Print a version number.
raw_ostream& operator<<(raw_ostream &Out, const VersionTuple &V);

} // end namespace clang
#endif // LLVM_CLANG_BASIC_VERSIONTUPLE_H
