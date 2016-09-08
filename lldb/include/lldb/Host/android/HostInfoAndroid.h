//===-- HostInfoAndroid.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_android_HostInfoAndroid_h_
#define lldb_Host_android_HostInfoAndroid_h_

#include "lldb/Host/linux/HostInfoLinux.h"

namespace lldb_private {

class HostInfoAndroid : public HostInfoLinux {
  friend class HostInfoBase;

public:
  static FileSpec GetDefaultShell();
  static FileSpec ResolveLibraryPath(const std::string &path,
                                     const ArchSpec &arch);

protected:
  static void ComputeHostArchitectureSupport(ArchSpec &arch_32,
                                             ArchSpec &arch_64);
  static bool ComputeTempFileBaseDirectory(FileSpec &file_spec);
};

} // end of namespace lldb_private

#endif // #ifndef lldb_Host_android_HostInfoAndroid_h_
