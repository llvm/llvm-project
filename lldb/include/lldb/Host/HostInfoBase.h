//===-- HostInfoBase.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_HostInfoBase_h_
#define lldb_Host_HostInfoBase_h_

#include "lldb/Core/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-enumerations.h"

#include "llvm/ADT/StringRef.h"

#include <stdint.h>

#include <string>

namespace lldb_private {

class FileSpec;

class HostInfoBase {
private:
  // Static class, unconstructable.
  HostInfoBase() {}
  ~HostInfoBase() {}

public:
  static void Initialize();
  static void Terminate();

  //------------------------------------------------------------------
  /// Gets the host target triple as a const string.
  ///
  /// @return
  ///     A const string object containing the host target triple.
  //------------------------------------------------------------------
  static llvm::StringRef GetTargetTriple();

  //------------------------------------------------------------------
  /// Gets the host architecture.
  ///
  /// @return
  ///     A const architecture object that represents the host
  ///     architecture.
  //------------------------------------------------------------------
  enum ArchitectureKind {
    eArchKindDefault, // The overall default architecture that applications will
                      // run on this host
    eArchKind32, // If this host supports 32 bit programs, return the default 32
                 // bit arch
    eArchKind64  // If this host supports 64 bit programs, return the default 64
                 // bit arch
  };

  static const ArchSpec &
  GetArchitecture(ArchitectureKind arch_kind = eArchKindDefault);

  //------------------------------------------------------------------
  /// Find a resource files that are related to LLDB.
  ///
  /// Operating systems have different ways of storing shared
  /// libraries and related resources. This function abstracts the
  /// access to these paths.
  ///
  /// @param[in] path_type
  ///     The type of LLDB resource path you are looking for. If the
  ///     enumeration ends with "Dir", then only the \a file_spec's
  ///     directory member gets filled in.
  ///
  /// @param[in] file_spec
  ///     A file spec that gets filled in with the appropriate path.
  ///
  /// @return
  ///     \b true if \a resource_path was resolved, \a false otherwise.
  //------------------------------------------------------------------
  static bool GetLLDBPath(lldb::PathType type, FileSpec &file_spec);

  //---------------------------------------------------------------------------
  /// If the triple does not specify the vendor, os, and environment parts, we
  /// "augment" these using information from the host and return the resulting
  /// ArchSpec object.
  //---------------------------------------------------------------------------
  static ArchSpec GetAugmentedArchSpec(llvm::StringRef triple);

protected:
  static bool ComputeSharedLibraryDirectory(FileSpec &file_spec);
  static bool ComputeSupportExeDirectory(FileSpec &file_spec);
  static bool ComputeProcessTempFileDirectory(FileSpec &file_spec);
  static bool ComputeGlobalTempFileDirectory(FileSpec &file_spec);
  static bool ComputeTempFileBaseDirectory(FileSpec &file_spec);
  static bool ComputeHeaderDirectory(FileSpec &file_spec);
  static bool ComputeSystemPluginsDirectory(FileSpec &file_spec);
  static bool ComputeClangDirectory(FileSpec &file_spec);
  static bool ComputeUserPluginsDirectory(FileSpec &file_spec);

  static void ComputeHostArchitectureSupport(ArchSpec &arch_32,
                                             ArchSpec &arch_64);
};
}

#endif
