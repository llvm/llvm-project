//===--- Tools.h - Tool Implementations -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_DRIVER_TOOLS_H_
#define CLANG_LIB_DRIVER_TOOLS_H_

#include "clang/Driver/Tool.h"
#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/Compiler.h"

namespace clang {
  class ObjCRuntime;

namespace driver {
  class Driver;

namespace toolchains {
  class Darwin;
}

namespace tools {

  /// \brief Clang compiler tool.
  class LLVM_LIBRARY_VISIBILITY Clang : public Tool {
    void AddPreprocessingOptions(Compilation &C,
                                 const Driver &D,
                                 const ArgList &Args,
                                 ArgStringList &CmdArgs,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs) const;

    void AddARMTargetArgs(const ArgList &Args, ArgStringList &CmdArgs,
                          bool KernelOrKext) const;
    void AddMIPSTargetArgs(const ArgList &Args, ArgStringList &CmdArgs) const;
    void AddPPCTargetArgs(const ArgList &Args, ArgStringList &CmdArgs) const;
    void AddSparcTargetArgs(const ArgList &Args, ArgStringList &CmdArgs) const;
    void AddX86TargetArgs(const ArgList &Args, ArgStringList &CmdArgs) const;
    void AddHexagonTargetArgs (const ArgList &Args, ArgStringList &CmdArgs) const;

    enum RewriteKind { RK_None, RK_Fragile, RK_NonFragile };

    ObjCRuntime AddObjCRuntimeArgs(const ArgList &args, ArgStringList &cmdArgs,
                                   RewriteKind rewrite) const;

  public:
    Clang(const ToolChain &TC) : Tool("clang", "clang frontend", TC) {}

    virtual bool hasGoodDiagnostics() const { return true; }
    virtual bool hasIntegratedAssembler() const { return true; }
    virtual bool hasIntegratedCPP() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };

  /// \brief Clang integrated assembler tool.
  class LLVM_LIBRARY_VISIBILITY ClangAs : public Tool {
    void AddARMTargetArgs(const ArgList &Args, ArgStringList &CmdArgs) const;
  public:
    ClangAs(const ToolChain &TC) : Tool("clang::as",
                                        "clang integrated assembler", TC) {}

    virtual bool hasGoodDiagnostics() const { return true; }
    virtual bool hasIntegratedAssembler() const { return false; }
    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };

  /// gcc - Generic GCC tool implementations.
namespace gcc {
  class LLVM_LIBRARY_VISIBILITY Common : public Tool {
  public:
    Common(const char *Name, const char *ShortName,
           const ToolChain &TC) : Tool(Name, ShortName, TC) {}

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;

    /// RenderExtraToolArgs - Render any arguments necessary to force
    /// the particular tool mode.
    virtual void RenderExtraToolArgs(const JobAction &JA,
                                     ArgStringList &CmdArgs) const = 0;
  };


  class LLVM_LIBRARY_VISIBILITY Preprocess : public Common {
  public:
    Preprocess(const ToolChain &TC) : Common("gcc::Preprocess",
                                             "gcc preprocessor", TC) {}

    virtual bool hasGoodDiagnostics() const { return true; }
    virtual bool hasIntegratedCPP() const { return false; }

    virtual void RenderExtraToolArgs(const JobAction &JA,
                                     ArgStringList &CmdArgs) const;
  };

  class LLVM_LIBRARY_VISIBILITY Precompile : public Common  {
  public:
    Precompile(const ToolChain &TC) : Common("gcc::Precompile",
                                             "gcc precompile", TC) {}

    virtual bool hasGoodDiagnostics() const { return true; }
    virtual bool hasIntegratedCPP() const { return true; }

    virtual void RenderExtraToolArgs(const JobAction &JA,
                                     ArgStringList &CmdArgs) const;
  };

  class LLVM_LIBRARY_VISIBILITY Compile : public Common  {
  public:
    Compile(const ToolChain &TC) : Common("gcc::Compile",
                                          "gcc frontend", TC) {}

    virtual bool hasGoodDiagnostics() const { return true; }
    virtual bool hasIntegratedCPP() const { return true; }

    virtual void RenderExtraToolArgs(const JobAction &JA,
                                     ArgStringList &CmdArgs) const;
  };

  class LLVM_LIBRARY_VISIBILITY Assemble : public Common  {
  public:
    Assemble(const ToolChain &TC) : Common("gcc::Assemble",
                                           "assembler (via gcc)", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void RenderExtraToolArgs(const JobAction &JA,
                                     ArgStringList &CmdArgs) const;
  };

  class LLVM_LIBRARY_VISIBILITY Link : public Common  {
  public:
    Link(const ToolChain &TC) : Common("gcc::Link",
                                       "linker (via gcc)", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void RenderExtraToolArgs(const JobAction &JA,
                                     ArgStringList &CmdArgs) const;
  };
} // end namespace gcc

namespace hexagon {
  // For Hexagon, we do not need to instantiate tools for PreProcess, PreCompile and Compile.
  // We simply use "clang -cc1" for those actions.
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool {
  public:
    Assemble(const ToolChain &TC) : Tool("hexagon::Assemble",
      "hexagon-as", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void RenderExtraToolArgs(const JobAction &JA,
                                     ArgStringList &CmdArgs) const;
    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };

  class LLVM_LIBRARY_VISIBILITY Link : public Tool {
  public:
    Link(const ToolChain &TC) : Tool("hexagon::Link",
      "hexagon-ld", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void RenderExtraToolArgs(const JobAction &JA,
                                     ArgStringList &CmdArgs) const;
    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace hexagon.


namespace darwin {
  llvm::Triple::ArchType getArchTypeForDarwinArchName(StringRef Str);

  class LLVM_LIBRARY_VISIBILITY DarwinTool : public Tool {
    virtual void anchor();
  protected:
    void AddDarwinArch(const ArgList &Args, ArgStringList &CmdArgs) const;

    const toolchains::Darwin &getDarwinToolChain() const {
      return reinterpret_cast<const toolchains::Darwin&>(getToolChain());
    }

  public:
    DarwinTool(const char *Name, const char *ShortName,
               const ToolChain &TC) : Tool(Name, ShortName, TC) {}
  };

  class LLVM_LIBRARY_VISIBILITY CC1 : public DarwinTool  {
    virtual void anchor();
  public:
    static const char *getBaseInputName(const ArgList &Args,
                                 const InputInfoList &Input);
    static const char *getBaseInputStem(const ArgList &Args,
                                 const InputInfoList &Input);
    static const char *getDependencyFileName(const ArgList &Args,
                                             const InputInfoList &Inputs);

  protected:
    const char *getCC1Name(types::ID Type) const;

    void AddCC1Args(const ArgList &Args, ArgStringList &CmdArgs) const;
    void RemoveCC1UnsupportedArgs(ArgStringList &CmdArgs) const;
    void AddCC1OptionsArgs(const ArgList &Args, ArgStringList &CmdArgs,
                           const InputInfoList &Inputs,
                           const ArgStringList &OutputArgs) const;
    void AddCPPOptionsArgs(const ArgList &Args, ArgStringList &CmdArgs,
                           const InputInfoList &Inputs,
                           const ArgStringList &OutputArgs) const;
    void AddCPPUniqueOptionsArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs,
                                 const InputInfoList &Inputs) const;
    void AddCPPArgs(const ArgList &Args, ArgStringList &CmdArgs) const;

  public:
    CC1(const char *Name, const char *ShortName,
        const ToolChain &TC) : DarwinTool(Name, ShortName, TC) {}

    virtual bool hasGoodDiagnostics() const { return true; }
    virtual bool hasIntegratedCPP() const { return true; }
  };

  class LLVM_LIBRARY_VISIBILITY Preprocess : public CC1  {
  public:
    Preprocess(const ToolChain &TC) : CC1("darwin::Preprocess",
                                          "gcc preprocessor", TC) {}

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };

  class LLVM_LIBRARY_VISIBILITY Compile : public CC1  {
  public:
    Compile(const ToolChain &TC) : CC1("darwin::Compile", "gcc frontend", TC) {}

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };

  class LLVM_LIBRARY_VISIBILITY Assemble : public DarwinTool  {
  public:
    Assemble(const ToolChain &TC) : DarwinTool("darwin::Assemble",
                                               "assembler", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };

  class LLVM_LIBRARY_VISIBILITY Link : public DarwinTool  {
    bool NeedsTempPath(const InputInfoList &Inputs) const;
    void AddLinkArgs(Compilation &C, const ArgList &Args,
                     ArgStringList &CmdArgs, const InputInfoList &Inputs) const;

  public:
    Link(const ToolChain &TC) : DarwinTool("darwin::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };

  class LLVM_LIBRARY_VISIBILITY Lipo : public DarwinTool  {
  public:
    Lipo(const ToolChain &TC) : DarwinTool("darwin::Lipo", "lipo", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };

  class LLVM_LIBRARY_VISIBILITY Dsymutil : public DarwinTool  {
  public:
    Dsymutil(const ToolChain &TC) : DarwinTool("darwin::Dsymutil",
                                               "dsymutil", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };

  class LLVM_LIBRARY_VISIBILITY VerifyDebug : public DarwinTool  {
  public:
    VerifyDebug(const ToolChain &TC) : DarwinTool("darwin::VerifyDebug",
						  "dwarfdump", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
			      const InputInfo &Output,
			      const InputInfoList &Inputs,
			      const ArgList &TCArgs,
			      const char *LinkingOutput) const;
  };

}

  /// openbsd -- Directly call GNU Binutils assembler and linker
namespace openbsd {
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool  {
  public:
    Assemble(const ToolChain &TC) : Tool("openbsd::Assemble", "assembler",
                                         TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("openbsd::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace openbsd

  /// bitrig -- Directly call GNU Binutils assembler and linker
namespace bitrig {
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool  {
  public:
    Assemble(const ToolChain &TC) : Tool("bitrig::Assemble", "assembler",
                                         TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("bitrig::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace bitrig

  /// freebsd -- Directly call GNU Binutils assembler and linker
namespace freebsd {
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool  {
  public:
    Assemble(const ToolChain &TC) : Tool("freebsd::Assemble", "assembler",
                                         TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("freebsd::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace freebsd

  /// netbsd -- Directly call GNU Binutils assembler and linker
namespace netbsd {
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool  {

  public:
    Assemble(const ToolChain &TC)
      : Tool("netbsd::Assemble", "assembler", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {

  public:
    Link(const ToolChain &TC)
      : Tool("netbsd::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace netbsd

  /// linux -- Directly call GNU Binutils assembler and linker
namespace linuxtools {
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool  {
  public:
    Assemble(const ToolChain &TC) : Tool("linux::Assemble", "assembler",
                                         TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("linux::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
}
  /// minix -- Directly call GNU Binutils assembler and linker
namespace minix {
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool  {
  public:
    Assemble(const ToolChain &TC) : Tool("minix::Assemble", "assembler",
                                         TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("minix::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace minix

  /// solaris -- Directly call Solaris assembler and linker
namespace solaris {
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool  {
  public:
    Assemble(const ToolChain &TC) : Tool("solaris::Assemble", "assembler",
                                         TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("solaris::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace solaris

  /// auroraux -- Directly call GNU Binutils assembler and linker
namespace auroraux {
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool  {
  public:
    Assemble(const ToolChain &TC) : Tool("auroraux::Assemble", "assembler",
                                         TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("auroraux::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace auroraux

  /// dragonfly -- Directly call GNU Binutils assembler and linker
namespace dragonfly {
  class LLVM_LIBRARY_VISIBILITY Assemble : public Tool  {
  public:
    Assemble(const ToolChain &TC) : Tool("dragonfly::Assemble", "assembler",
                                         TC) {}

    virtual bool hasIntegratedCPP() const { return false; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("dragonfly::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace dragonfly

  /// Visual studio tools.
namespace visualstudio {
  class LLVM_LIBRARY_VISIBILITY Link : public Tool  {
  public:
    Link(const ToolChain &TC) : Tool("visualstudio::Link", "linker", TC) {}

    virtual bool hasIntegratedCPP() const { return false; }
    virtual bool isLinkJob() const { return true; }

    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const ArgList &TCArgs,
                              const char *LinkingOutput) const;
  };
} // end namespace visualstudio

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // CLANG_LIB_DRIVER_TOOLS_H_
