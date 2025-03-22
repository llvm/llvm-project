#include "C2000.h"
#include "ToolChains/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Multilib.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

void c2000::getC2000TargetFeatures(const Driver &D, const ArgList &Args,
                                   std::vector<StringRef> &Features) {

  for (auto *A : Args) {
    if (A->getOption().matches(options::OPT_eabi)) {
      StringRef abi = A->getValue();
      if (abi.starts_with("eabi"))
        Features.push_back("+eabi");
      continue;
    }
    if (A->getOption().matches(options::OPT_relaxed_ansi)) {
      Features.push_back("+relaxed_ansi");
      continue;
    }
    if (A->getOption().matches(options::OPT_fp_mode)) {
      StringRef fp_mode = A->getValue();
      if (fp_mode.starts_with("relaxed"))
        Features.push_back("+relaxed");
      continue;
    }
    if (A->getOption().matches(options::OPT_cla_support)) {
      Features.push_back("+cla_support");
      StringRef cla_support = A->getValue();
      if (cla_support.starts_with("cla0"))
        Features.push_back("+cla0");
      else if (cla_support.starts_with("cla0"))
        Features.push_back("+cla1");
      else if (cla_support.starts_with("cla2"))
        Features.push_back("+cla2");
      continue;
    }
    if (A->getOption().matches(options::OPT_float_support)) {
      StringRef float_support = A->getValue();
      if (float_support.starts_with("fpu64"))
        Features.push_back("+fpu64");
      else if (float_support.starts_with("fpu32"))
        Features.push_back("+fpu32");
      continue;
    }
    if (A->getOption().matches(options::OPT_idiv_support)) {
      StringRef idiv_support = A->getValue();
      if (idiv_support.starts_with("idiv0"))
        Features.push_back("+idiv0");
      continue;
    }
    if (A->getOption().matches(options::OPT_tmu_support)) {
      Features.push_back("+tmu_support");
      StringRef tmu_support = A->getValue();
      if (tmu_support.starts_with("tmu1"))
        Features.push_back("+tmu1");
      continue;
    }
    if (A->getOption().matches(options::OPT_vcu_support)) {
      Features.push_back("+vcu_support");
      StringRef vcu_support = A->getValue();
      if (vcu_support.starts_with("vcu2"))
        Features.push_back("+vcu2");
      else if (vcu_support.starts_with("vcrc"))
        Features.push_back("+vcrc");
      continue;
    }
    if (A->getOption().matches(options::OPT_opt_level)) {
      StringRef opt_level = A->getValue();
      if (!opt_level.starts_with("off"))
        Features.push_back("+opt_level");
    } else if (A->getOption().matches(options::OPT_O)) {
      Features.push_back("+opt_level");
    }
  }
}
