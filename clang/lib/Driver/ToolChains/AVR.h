//===--- AVR.h - AVR Tool and ToolChain Implementations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AVR_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AVR_H

#include "Gnu.h"
#include "InputInfo.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Tool.h"

namespace clang {
namespace driver {
namespace toolchains {

class LLVM_LIBRARY_VISIBILITY AVRToolChain : public Generic_ELF {
public:
  AVRToolChain(const Driver &D, const llvm::Triple &Triple,
               const llvm::opt::ArgList &Args);

protected:
  Tool *buildLinker() const override;

private:
  /// Whether libgcc, libct, and friends should be linked.
  ///
  /// This is not done if the user does not specify a
  /// microcontroller on the command line.
  bool LinkStdlib;

  llvm::Optional<std::string> findAVRLibcInstallation() const;
};

} // end namespace toolchains

namespace tools {
namespace AVR {
class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const llvm::Triple &Triple, const ToolChain &TC, bool LinkStdlib)
      : GnuTool("AVR::Linker", "avr-ld", TC), Triple(Triple),
        LinkStdlib(LinkStdlib) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

protected:
  const llvm::Triple &Triple;
  bool LinkStdlib;
};
} // end namespace AVR
} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AVR_H
