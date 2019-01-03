//===-- SWIG Interface for SBInitializerOptions -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBInitializerOptions
{
public:
  SBInitializerOptions ();
  SBInitializerOptions (const lldb::SBInitializerOptions &rhs);
  ~SBInitializerOptions();

  void SetCaptureReproducer(bool b);
  void SetReplayReproducer(bool b);
  void SetReproducerPath(const char* path);
};

} // namespace lldb
