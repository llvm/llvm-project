//===-- SWIG Interface for SBInitializerOptions -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
