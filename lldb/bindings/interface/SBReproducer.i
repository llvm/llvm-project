//===-- SWIG Interface for SBReproducer--------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {
class SBReproducer
{
    public:
        static const char *Capture(const char *path);
        static const char *PassiveReplay(const char *path);
        static bool SetAutoGenerate(bool b);
        static void SetWorkingDirectory(const char *path);
};
}
