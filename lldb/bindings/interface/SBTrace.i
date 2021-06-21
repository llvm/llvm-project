//===-- SWIG Interface for SBTrace.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a processor trace."
) SBTrace;
class LLDB_API SBTrace {
public:
  SBTrace();

  const char *GetStartConfigurationHelp();

  SBError Start(const SBStructuredData &configuration);

  SBError Start(const SBThread &thread, const SBStructuredData &configuration);

  SBError Stop();

  SBError Stop(const SBThread &thread);

  explicit operator bool() const;

  bool IsValid();
};
} // namespace lldb
