//===--- P2.h - P2 Tool and ToolChain Implementations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_P2_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_P2_H

#include "Gnu.h"
#include "InputInfo.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Tool.h"

namespace clang {
  namespace driver {
    namespace toolchains {

      class LLVM_LIBRARY_VISIBILITY P2ToolChain : public Generic_ELF {
      public:
        P2ToolChain(const Driver &D, const llvm::Triple &Triple,
                     const llvm::opt::ArgList &Args);
        bool IsIntegratedAssemblerDefault() const { return true; }

      protected:
        Tool *buildLinker() const override;

      private:
      };

    } // end namespace toolchains

    namespace tools {
      namespace P2 {

        // class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
        // public:
        //   Assembler(const ToolChain &TC)
        //       : Tool("solaris::Assembler", "assembler", TC) {}

        //   bool hasIntegratedCPP() const override { return false; }

        //   void ConstructJob(Compilation &C, const JobAction &JA,
        //                     const InputInfo &Output, const InputInfoList &Inputs,
        //                     const llvm::opt::ArgList &TCArgs,
        //                     const char *LinkingOutput) const override;
        // };

        class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
        public:
          Linker(const llvm::Triple &Triple, const ToolChain &TC)
              : GnuTool("P2::Linker", "P2-ld", TC), Triple(Triple) {}

          bool hasIntegratedCPP() const override { return false; }
          bool isLinkJob() const override { return true; }
          void ConstructJob(Compilation &C, const JobAction &JA,
                            const InputInfo &Output, const InputInfoList &Inputs,
                            const llvm::opt::ArgList &TCArgs,
                            const char *LinkingOutput) const override;

        protected:
          const llvm::Triple &Triple;
        };
      } // end namespace P2
    } // end namespace tools
  } // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_P2_H
