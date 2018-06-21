//===-- HostInfoWindows.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_windows_HostInfoWindows_h_
#define lldb_Host_windows_HostInfoWindows_h_

#include "lldb/Host/HostInfoBase.h"
#include "lldb/Utility/FileSpec.h"

namespace lldb_private {

class HostInfoWindows : public HostInfoBase {
  friend class HostInfoBase;

private:
  // Static class, unconstructable.
  HostInfoWindows();
  ~HostInfoWindows();

public:
  static void Initialize();
  static void Terminate();

  static size_t GetPageSize();

  static bool GetOSVersion(uint32_t &major, uint32_t &minor, uint32_t &update);
  static bool GetOSBuildString(std::string &s);
  static bool GetOSKernelDescription(std::string &s);
  static bool GetHostname(std::string &s);
  static FileSpec GetProgramFileSpec();
  static FileSpec GetDefaultShell();

  static bool GetEnvironmentVar(const std::string &var_name, std::string &var);

protected:
  static bool ComputePythonDirectory(FileSpec &file_spec);

private:
  static FileSpec m_program_filespec;
};
}

#endif
