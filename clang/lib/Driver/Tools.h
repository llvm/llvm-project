//===--- Tools.h - Tool Implementations -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLS_H

#include "clang/Basic/VersionTuple.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Compiler.h"

namespace clang {
class ObjCRuntime;

namespace driver {
class Command;
class Driver;

namespace toolchains {
class MachO;
}

namespace tools {

namespace visualstudio {
class Compiler;
}

using llvm::opt::ArgStringList;

SmallString<128> getCompilerRT(const ToolChain &TC,
                               const llvm::opt::ArgList &Args,
                               StringRef Component, bool Shared = false);

/// \brief Clang compiler tool.
class LLVM_LIBRARY_VISIBILITY Clang : public Tool {
public:
  static const char *getBaseInputName(const llvm::opt::ArgList &Args,
                                      const InputInfo &Input);
  static const char *getBaseInputStem(const llvm::opt::ArgList &Args,
                                      const InputInfoList &Inputs);
  static const char *getDependencyFileName(const llvm::opt::ArgList &Args,
                                           const InputInfoList &Inputs);

private:
  void AddPreprocessingOptions(Compilation &C, const JobAction &JA,
                               const Driver &D, const llvm::opt::ArgList &Args,
                               llvm::opt::ArgStringList &CmdArgs,
                               const InputInfo &Output,
                               const InputInfoList &Inputs,
                               const ToolChain *AuxToolChain) const;

  void AddAArch64TargetArgs(const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs) const;
  void AddARMTargetArgs(const llvm::Triple &Triple,
                        const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs,
                        bool KernelOrKext) const;
  void AddARM64TargetArgs(const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs) const;
  void AddMIPSTargetArgs(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) const;
  void AddPPCTargetArgs(const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs) const;
  void AddR600TargetArgs(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) const;
  void AddSparcTargetArgs(const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs) const;
  void AddSystemZTargetArgs(const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs) const;
  void AddX86TargetArgs(const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs) const;
  void AddHexagonTargetArgs(const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs) const;

  enum RewriteKind { RK_None, RK_Fragile, RK_NonFragile };

  ObjCRuntime AddObjCRuntimeArgs(const llvm::opt::ArgList &args,
                                 llvm::opt::ArgStringList &cmdArgs,
                                 RewriteKind rewrite) const;

  void AddClangCLArgs(const llvm::opt::ArgList &Args,
                      llvm::opt::ArgStringList &CmdArgs,
                      enum CodeGenOptions::DebugInfoKind *DebugInfoKind,
                      bool *EmitCodeView) const;

  visualstudio::Compiler *getCLFallback() const;

  mutable std::unique_ptr<visualstudio::Compiler> CLFallback;

public:
  // CAUTION! The first constructor argument ("clang") is not arbitrary,
  // as it is for other tools. Some operations on a Tool actually test
  // whether that tool is Clang based on the Tool's Name as a string.
  Clang(const ToolChain &TC) : Tool("clang", "clang frontend", TC, RF_Full) {}

  bool hasGoodDiagnostics() const override { return true; }
  bool hasIntegratedAssembler() const override { return true; }
  bool hasIntegratedCPP() const override { return true; }
  bool canEmitIR() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

/// \brief Clang integrated assembler tool.
class LLVM_LIBRARY_VISIBILITY ClangAs : public Tool {
public:
  ClangAs(const ToolChain &TC)
      : Tool("clang::as", "clang integrated assembler", TC, RF_Full) {}
  void AddMIPSTargetArgs(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) const;
  bool hasGoodDiagnostics() const override { return true; }
  bool hasIntegratedAssembler() const override { return false; }
  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

/// \brief Base class for all GNU tools that provide the same behavior when
/// it comes to response files support
class LLVM_LIBRARY_VISIBILITY GnuTool : public Tool {
  virtual void anchor();

public:
  GnuTool(const char *Name, const char *ShortName, const ToolChain &TC)
      : Tool(Name, ShortName, TC, RF_Full, llvm::sys::WEM_CurrentCodePage) {}
};

/// gcc - Generic GCC tool implementations.
namespace gcc {
class LLVM_LIBRARY_VISIBILITY Common : public GnuTool {
public:
  Common(const char *Name, const char *ShortName, const ToolChain &TC)
      : GnuTool(Name, ShortName, TC) {}

  // A gcc tool has an "integrated" assembler that it will call to produce an
  // object. Let it use that assembler so that we don't have to deal with
  // assembly syntax incompatibilities.
  bool hasIntegratedAssembler() const override { return true; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

  /// RenderExtraToolArgs - Render any arguments necessary to force
  /// the particular tool mode.
  virtual void RenderExtraToolArgs(const JobAction &JA,
                                   llvm::opt::ArgStringList &CmdArgs) const = 0;
};

class LLVM_LIBRARY_VISIBILITY Preprocessor : public Common {
public:
  Preprocessor(const ToolChain &TC)
      : Common("gcc::Preprocessor", "gcc preprocessor", TC) {}

  bool hasGoodDiagnostics() const override { return true; }
  bool hasIntegratedCPP() const override { return false; }

  void RenderExtraToolArgs(const JobAction &JA,
                           llvm::opt::ArgStringList &CmdArgs) const override;
};

class LLVM_LIBRARY_VISIBILITY Compiler : public Common {
public:
  Compiler(const ToolChain &TC) : Common("gcc::Compiler", "gcc frontend", TC) {}

  bool hasGoodDiagnostics() const override { return true; }
  bool hasIntegratedCPP() const override { return true; }

  void RenderExtraToolArgs(const JobAction &JA,
                           llvm::opt::ArgStringList &CmdArgs) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public Common {
public:
  Linker(const ToolChain &TC) : Common("gcc::Linker", "linker (via gcc)", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void RenderExtraToolArgs(const JobAction &JA,
                           llvm::opt::ArgStringList &CmdArgs) const override;
};
} // end namespace gcc

namespace hexagon {
// For Hexagon, we do not need to instantiate tools for PreProcess, PreCompile
// and Compile.
// We simply use "clang -cc1" for those actions.
class LLVM_LIBRARY_VISIBILITY Assembler : public GnuTool {
public:
  Assembler(const ToolChain &TC)
      : GnuTool("hexagon::Assembler", "hexagon-as", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void RenderExtraToolArgs(const JobAction &JA,
                           llvm::opt::ArgStringList &CmdArgs) const;
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("hexagon::Linker", "hexagon-ld", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  virtual void RenderExtraToolArgs(const JobAction &JA,
                                   llvm::opt::ArgStringList &CmdArgs) const;
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace hexagon.

namespace amdgpu {

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("amdgpu::Linker", "lld", TC) {}
  bool isLinkJob() const override { return true; }
  bool hasIntegratedCPP() const override { return false; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

} // end namespace amdgpu

namespace wasm {

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  explicit Linker(const ToolChain &TC);
  bool isLinkJob() const override;
  bool hasIntegratedCPP() const override;
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

} // end namespace wasm

namespace arm {
std::string getARMTargetCPU(StringRef CPU, StringRef Arch,
                            const llvm::Triple &Triple);
const std::string getARMArch(StringRef Arch,
                             const llvm::Triple &Triple);
StringRef getARMCPUForMArch(StringRef Arch, const llvm::Triple &Triple);
StringRef getLLVMArchSuffixForARM(StringRef CPU, StringRef Arch,
                                  const llvm::Triple &Triple);

void appendEBLinkFlags(const llvm::opt::ArgList &Args, ArgStringList &CmdArgs,
                       const llvm::Triple &Triple);
} // end namespace arm

namespace mips {
typedef enum { NanLegacy = 1, Nan2008 = 2 } NanEncoding;

enum class FloatABI {
  Invalid,
  Soft,
  Hard,
};

NanEncoding getSupportedNanEncoding(StringRef &CPU);
void getMipsCPUAndABI(const llvm::opt::ArgList &Args,
                      const llvm::Triple &Triple, StringRef &CPUName,
                      StringRef &ABIName);
std::string getMipsABILibSuffix(const llvm::opt::ArgList &Args,
                                const llvm::Triple &Triple);
bool hasMipsAbiArg(const llvm::opt::ArgList &Args, const char *Value);
bool isUCLibc(const llvm::opt::ArgList &Args);
bool isNaN2008(const llvm::opt::ArgList &Args, const llvm::Triple &Triple);
bool isFPXXDefault(const llvm::Triple &Triple, StringRef CPUName,
                   StringRef ABIName, mips::FloatABI FloatABI);
bool shouldUseFPXX(const llvm::opt::ArgList &Args, const llvm::Triple &Triple,
                   StringRef CPUName, StringRef ABIName,
                   mips::FloatABI FloatABI);
} // end namespace mips

namespace ppc {
bool hasPPCAbiArg(const llvm::opt::ArgList &Args, const char *Value);
} // end namespace ppc

/// cloudabi -- Directly call GNU Binutils linker
namespace cloudabi {
class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("cloudabi::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace cloudabi

namespace darwin {
llvm::Triple::ArchType getArchTypeForMachOArchName(StringRef Str);
void setTripleTypeForMachOArchName(llvm::Triple &T, StringRef Str);

class LLVM_LIBRARY_VISIBILITY MachOTool : public Tool {
  virtual void anchor();

protected:
  void AddMachOArch(const llvm::opt::ArgList &Args,
                    llvm::opt::ArgStringList &CmdArgs) const;

  const toolchains::MachO &getMachOToolChain() const {
    return reinterpret_cast<const toolchains::MachO &>(getToolChain());
  }

public:
  MachOTool(
      const char *Name, const char *ShortName, const ToolChain &TC,
      ResponseFileSupport ResponseSupport = RF_None,
      llvm::sys::WindowsEncodingMethod ResponseEncoding = llvm::sys::WEM_UTF8,
      const char *ResponseFlag = "@")
      : Tool(Name, ShortName, TC, ResponseSupport, ResponseEncoding,
             ResponseFlag) {}
};

class LLVM_LIBRARY_VISIBILITY Assembler : public MachOTool {
public:
  Assembler(const ToolChain &TC)
      : MachOTool("darwin::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public MachOTool {
  bool NeedsTempPath(const InputInfoList &Inputs) const;
  void AddLinkArgs(Compilation &C, const llvm::opt::ArgList &Args,
                   llvm::opt::ArgStringList &CmdArgs,
                   const InputInfoList &Inputs) const;

public:
  Linker(const ToolChain &TC)
      : MachOTool("darwin::Linker", "linker", TC, RF_FileList,
                  llvm::sys::WEM_UTF8, "-filelist") {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Lipo : public MachOTool {
public:
  Lipo(const ToolChain &TC) : MachOTool("darwin::Lipo", "lipo", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Dsymutil : public MachOTool {
public:
  Dsymutil(const ToolChain &TC)
      : MachOTool("darwin::Dsymutil", "dsymutil", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isDsymutilJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY VerifyDebug : public MachOTool {
public:
  VerifyDebug(const ToolChain &TC)
      : MachOTool("darwin::VerifyDebug", "dwarfdump", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace darwin

/// openbsd -- Directly call GNU Binutils assembler and linker
namespace openbsd {
class LLVM_LIBRARY_VISIBILITY Assembler : public GnuTool {
public:
  Assembler(const ToolChain &TC)
      : GnuTool("openbsd::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("openbsd::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace openbsd

/// bitrig -- Directly call GNU Binutils assembler and linker
namespace bitrig {
class LLVM_LIBRARY_VISIBILITY Assembler : public GnuTool {
public:
  Assembler(const ToolChain &TC)
      : GnuTool("bitrig::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("bitrig::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace bitrig

/// freebsd -- Directly call GNU Binutils assembler and linker
namespace freebsd {
class LLVM_LIBRARY_VISIBILITY Assembler : public GnuTool {
public:
  Assembler(const ToolChain &TC)
      : GnuTool("freebsd::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("freebsd::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace freebsd

/// netbsd -- Directly call GNU Binutils assembler and linker
namespace netbsd {
class LLVM_LIBRARY_VISIBILITY Assembler : public GnuTool {
public:
  Assembler(const ToolChain &TC)
      : GnuTool("netbsd::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("netbsd::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace netbsd

/// Directly call GNU Binutils' assembler and linker.
namespace gnutools {
class LLVM_LIBRARY_VISIBILITY Assembler : public GnuTool {
public:
  Assembler(const ToolChain &TC) : GnuTool("GNU::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("GNU::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace gnutools

namespace nacltools {
class LLVM_LIBRARY_VISIBILITY AssemblerARM : public gnutools::Assembler {
public:
  AssemblerARM(const ToolChain &TC) : gnutools::Assembler(TC) {}

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("NaCl::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace nacltools

/// minix -- Directly call GNU Binutils assembler and linker
namespace minix {
class LLVM_LIBRARY_VISIBILITY Assembler : public GnuTool {
public:
  Assembler(const ToolChain &TC)
      : GnuTool("minix::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("minix::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace minix

/// solaris -- Directly call Solaris assembler and linker
namespace solaris {
class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
public:
  Assembler(const ToolChain &TC)
      : Tool("solaris::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("solaris::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace solaris

/// dragonfly -- Directly call GNU Binutils assembler and linker
namespace dragonfly {
class LLVM_LIBRARY_VISIBILITY Assembler : public GnuTool {
public:
  Assembler(const ToolChain &TC)
      : GnuTool("dragonfly::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("dragonfly::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace dragonfly

/// Visual studio tools.
namespace visualstudio {
VersionTuple getMSVCVersion(const Driver *D, const llvm::Triple &Triple,
                            const llvm::opt::ArgList &Args, bool IsWindowsMSVC);

class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC)
      : Tool("visualstudio::Linker", "linker", TC, RF_Full,
             llvm::sys::WEM_UTF16) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Compiler : public Tool {
public:
  Compiler(const ToolChain &TC)
      : Tool("visualstudio::Compiler", "compiler", TC, RF_Full,
             llvm::sys::WEM_UTF16) {}

  bool hasIntegratedAssembler() const override { return true; }
  bool hasIntegratedCPP() const override { return true; }
  bool isLinkJob() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

  std::unique_ptr<Command> GetCommand(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const llvm::opt::ArgList &TCArgs,
                                      const char *LinkingOutput) const;
};
} // end namespace visualstudio

/// MinGW -- Directly call GNU Binutils assembler and linker
namespace MinGW {
class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
public:
  Assembler(const ToolChain &TC) : Tool("MinGW::Assemble", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("MinGW::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  void AddLibGCC(const llvm::opt::ArgList &Args, ArgStringList &CmdArgs) const;
};
} // end namespace MinGW

namespace arm {
enum class FloatABI {
  Invalid,
  Soft,
  SoftFP,
  Hard,
};

FloatABI getARMFloatABI(const ToolChain &TC, const llvm::opt::ArgList &Args);
} // end namespace arm

namespace ppc {
enum class FloatABI {
  Invalid,
  Soft,
  Hard,
};

FloatABI getPPCFloatABI(const Driver &D, const llvm::opt::ArgList &Args);
} // end namespace ppc

namespace XCore {
// For XCore, we do not need to instantiate tools for PreProcess, PreCompile and
// Compile.
// We simply use "clang -cc1" for those actions.
class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
public:
  Assembler(const ToolChain &TC) : Tool("XCore::Assembler", "XCore-as", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("XCore::Linker", "XCore-ld", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace XCore.

namespace CrossWindows {
class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
public:
  Assembler(const ToolChain &TC) : Tool("CrossWindows::Assembler", "as", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC)
      : Tool("CrossWindows::Linker", "ld", TC, RF_Full) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace CrossWindows

/// SHAVE tools -- Directly call moviCompile and moviAsm
namespace SHAVE {
class LLVM_LIBRARY_VISIBILITY Compiler : public Tool {
public:
  Compiler(const ToolChain &TC) : Tool("moviCompile", "movicompile", TC) {}

  bool hasIntegratedCPP() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
public:
  Assembler(const ToolChain &TC) : Tool("moviAsm", "moviAsm", TC) {}

  bool hasIntegratedCPP() const override { return false; } // not sure.

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace SHAVE

/// The Myriad toolchain uses tools that are in two different namespaces.
/// The Compiler and Assembler as defined above are in the SHAVE namespace,
/// whereas the linker, which accepts code for a mixture of Sparc and SHAVE,
/// is in the Myriad namespace.
namespace Myriad {
class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("shave::Linker", "ld", TC) {}
  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace Myriad

namespace PS4cpu {
class LLVM_LIBRARY_VISIBILITY Assemble : public Tool {
public:
  Assemble(const ToolChain &TC)
      : Tool("PS4cpu::Assemble", "assembler", TC, RF_Full) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output,
                    const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Link : public Tool {
public:
  Link(const ToolChain &TC) : Tool("PS4cpu::Link", "linker", TC, RF_Full) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output,
                    const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace PS4cpu

} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLS_H
