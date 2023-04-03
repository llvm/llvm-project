/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef SRC_HIPBIN_NVIDIA_H_
#define SRC_HIPBIN_NVIDIA_H_

#include "hipBin_base.h"
#include "hipBin_util.h"
#include <iostream>
#include <vector>
#include <string>

class HipBinNvidia : public HipBinBase {
 private:
  HipBinUtil* hipBinUtilPtr_;
  string cudaPath_ = "";
  PlatformInfo platformInfoNV_;
  string hipCFlags_, hipCXXFlags_, hipLdFlags_;

 public:
  HipBinNvidia();
  virtual ~HipBinNvidia() = default;
  virtual bool detectPlatform();
  virtual void constructCompilerPath();
  virtual const string& getCompilerPath() const;
  virtual const PlatformInfo& getPlatformInfo() const;
  virtual string getCppConfig();
  virtual void printFull();
  virtual void printCompilerInfo() const;
  virtual string getCompilerVersion();
  virtual void checkHipconfig();
  virtual string getDeviceLibPath() const;
  virtual string getHipLibPath() const;
  virtual string getHipCC() const;
  virtual string getCompilerIncludePath();
  virtual string getHipInclude() const;
  virtual void initializeHipCXXFlags();
  virtual void initializeHipCFlags();
  virtual void initializeHipLdFlags();
  virtual const string& getHipCXXFlags() const;
  virtual const string& getHipCFlags() const;
  virtual const string& getHipLdFlags() const;
  virtual void executeHipCCCmd(vector<string> argv);
};

HipBinNvidia::HipBinNvidia() {
  PlatformInfo  platformInfo;
  platformInfo.os = getOSInfo();
  platformInfo.platform = nvidia;
  platformInfo.runtime = cuda;
  platformInfo.compiler = nvcc;
  platformInfoNV_ = platformInfo;
  constructCompilerPath();
}

// detects if cuda is installed
bool HipBinNvidia::detectPlatform() {
  string out;
  const string& nvccPath = getCompilerPath();
  fs::path cmdNv = nvccPath;
  cmdNv /= "bin/nvcc";
  const OsType& os = getOSInfo();
  const EnvVariables& var = getEnvVariables();
  bool detected = false;
  if (var.hipPlatformEnv_.empty()) {
    if (canRunCompiler(cmdNv.string(), out) || (canRunCompiler("nvcc", out))) {
      detected = true;
    }
  } else {
    if (var.hipPlatformEnv_ == "nvidia" || var.hipPlatformEnv_ == "nvcc") {
      detected = true;
      if (var.hipPlatformEnv_ == "nvcc")
        std::cerr << "Warning: HIP_PLATFORM=nvcc is deprecated."
             << "Please use HIP_PLATFORM=nvidia." << endl;
    }
  }
  return detected;
}



// returns device lib path
string HipBinNvidia::getDeviceLibPath() const {
  cout << "TODO Not required for now" << endl;
  return "";
}

// returns compiler path
string HipBinNvidia::getHipCC() const {
  string hipCC;
  const string& cudaPath = getCompilerPath();
  fs::path hipCCPath;
  hipCCPath = cudaPath;
  hipCCPath /= "bin/nvcc";
  hipCC = hipCCPath.string();
  return hipCC;
}

// returns compiler include path
string HipBinNvidia::getCompilerIncludePath() {
  cout << "TODO Not required for now" << endl;
  return "";
}


// checks Hipconfig
void HipBinNvidia::checkHipconfig() {
  cout << endl << "Check system installation: " << endl;
  cout << "check hipconfig in PATH..." << endl;
  if (system("which hipconfig > /dev/null 2>&1") != 0) {
    std::cerr << "FAIL " << endl;
  } else {
    cout << "good" << endl;
  }
}

// prints full
void HipBinNvidia::printFull() {
  const string& hipVersion = getHipVersion();
  const string&  hipPath = getHipPath();
  const string& roccmPath = getRoccmPath();
  const PlatformInfo& platformInfo = getPlatformInfo();
  const string& ccpConfig = getCppConfig();
  const string& cudaPath = getCompilerPath();
  cout << "HIP version: " << hipVersion << endl;
  cout << endl << "==hipconfig" << endl;
  cout << "HIP_PATH           :" << hipPath << endl;
  cout << "ROCM_PATH          :" << roccmPath << endl;
  cout << "HIP_COMPILER       :" << CompilerTypeStr(
                                    platformInfo.compiler) << endl;
  cout << "HIP_PLATFORM       :" << PlatformTypeStr(
                                    platformInfo.platform) << endl;
  cout << "HIP_RUNTIME        :" << RuntimeTypeStr(
                                    platformInfo.runtime) << endl;
  cout << "CPP_CONFIG         :" << ccpConfig << endl;
  cout << endl << "== nvcc" << endl;
  cout << "CUDA_PATH          :" << cudaPath <<endl;
  printCompilerInfo();
  cout << endl << "== Envirnoment Variables" << endl;
  printEnvironmentVariables();
  getSystemInfo();
  if (fs::exists("/usr/bin/lsb_release"))
    system("/usr/bin/lsb_release -a");
}

// returns hip include
string HipBinNvidia::getHipInclude() const {
  string hipPath, hipInclude;
  hipPath = getHipPath();
  fs::path hipIncludefs = hipPath;
  hipIncludefs /= "include";
  hipInclude = hipIncludefs.string();
  return hipInclude;
}

// initializes Hip ld Flags
void HipBinNvidia::initializeHipLdFlags() {
  string hipLdFlags;
  const string& cudaPath = getCompilerPath();
  hipLdFlags = " -Wno-deprecated-gpu-targets -lcuda -lcudart -L" +
               cudaPath + "/lib64";
  hipLdFlags_ = hipLdFlags;
}


// returns hipc Flags
const string& HipBinNvidia::getHipCFlags() const {
  return hipCFlags_;
}

// returns hip ld flags
const string& HipBinNvidia::getHipLdFlags() const {
  return hipLdFlags_;
}

// initialize Hipc flags
void HipBinNvidia::initializeHipCFlags() {
  string hipCFlags;
  const string& cudaPath = getCompilerPath();
  hipCFlags += " -isystem " + cudaPath + "/include";
  string hipIncludePath;
  hipIncludePath = getHipInclude();
  hipCFlags += " -isystem \"" + hipIncludePath + "\"";
  hipCFlags_ = hipCFlags;
}

// returns Hipccx flags
const string& HipBinNvidia::getHipCXXFlags() const {
  return hipCXXFlags_;
}

// initializes the HIPCCX flags
void HipBinNvidia::initializeHipCXXFlags() {
  string hipCXXFlags = " -Wno-deprecated-gpu-targets ";
  const string& cudaPath = getCompilerPath();
  hipCXXFlags += " -isystem " + cudaPath + "/include";
  string hipIncludePath;
  hipIncludePath = getHipInclude();
  hipCXXFlags += " -isystem \"" + hipIncludePath + "\"";
  hipCXXFlags_ = hipCXXFlags;
}

// returns Hip Lib Path
string HipBinNvidia::getHipLibPath() const {
  string hipLibPath;
  const EnvVariables& env = getEnvVariables();
  hipLibPath = env.hipLibPathEnv_;
  return hipLibPath;
}

// gets nvcc compiler Path
void HipBinNvidia::constructCompilerPath() {
  string complierPath;
  const EnvVariables& envVariables = getEnvVariables();
  if (envVariables.cudaPathEnv_.empty()) {
    fs::path cudaPathfs;
    cudaPathfs = "/usr/local/cuda";
    complierPath = cudaPathfs.string();
  } else {
    complierPath = envVariables.cudaPathEnv_;
  }
  cudaPath_ = complierPath;
}


// returns nvcc compiler Path
const string& HipBinNvidia::getCompilerPath() const {
  return cudaPath_;
}

// returns nvcc information
void HipBinNvidia::printCompilerInfo() const {
  string cmd;
  fs::path nvcc;
  nvcc = getCompilerPath();
  nvcc /= "bin/nvcc";
  cmd = nvcc.string() + " --version";
  system(cmd.c_str());
}

// returns nvcc version
string HipBinNvidia::getCompilerVersion() {
  string complierVersion, cmd;
  fs::path nvcc;
  nvcc = getCompilerPath();
  nvcc /= "bin/nvcc";
  cmd = nvcc.string() + " --version";
  system(cmd.c_str());
  return complierVersion;
}

// returns nvidia platform
const PlatformInfo& HipBinNvidia::getPlatformInfo() const {
  return platformInfoNV_;
}

// returns the cpp config
string HipBinNvidia::getCppConfig() {
  string cppConfig =
  " - D__HIP_PLATFORM_NVCC__ = -D__HIP_PLATFORM_NVIDIA__ = -I";
  string hipPath;
  hipPath = getHipPath();
  cppConfig += hipPath;
  cppConfig += "/include -I";
  cppConfig += cudaPath_;
  cppConfig += "/include";
  return cppConfig;
}

// performs hipcc command
void HipBinNvidia::executeHipCCCmd(vector<string> argv) {
  if (argv.size() < 2) {
    cout<< "No Arguments passed, exiting ...\n";
    exit(EXIT_SUCCESS);
  }
  const EnvVariables& var = getEnvVariables();
  int verbose = 0;
  if (!var.verboseEnv_.empty())
    verbose = stoi(var.verboseEnv_);
  // Verbose: 0x1=commands, 0x2=paths, 0x4=hipcc args
  // set if user explicitly requests -stdlib=libc++.
  // (else we default to libstdc++ for better interop with g++):
  bool setStdLib = 0;
  bool default_amdgpu_target = 1;
  bool compileOnly = 0;
  bool needCXXFLAGS = 0;  // need to add CXX flags to compile step
  bool needCFLAGS = 0;    // need to add C flags to compile step
  bool needLDFLAGS = 1;   // need to add LDFLAGS to compile step.
  bool fileTypeFlag = 0;  // to see if -x flag is mentioned
  bool hasOMPTargets = 0;  // If OMP targets is mentioned
  bool hasC = 0;          // options contain a c-style file
  // options contain a cpp-style file (NVCC must force recognition as GPU file)
  bool hasCXX = 0;
  // options contain a cu-style file (HCC must force recognition as GPU file)
  bool hasCU = 0;
  // options contain a hip-style file (HIP-Clang must pass offloading options)
  bool hasHIP = 0;
  bool printHipVersion = 0;    // print HIP version
  bool printCXXFlags = 0;      // print HIPCXXFLAGS
  bool printLDFlags = 0;       // print HIPLDFLAGS
  bool runCmd = 1;
  bool buildDeps = 0;
  bool linkType = 1;
  bool setLinkType = 0;
  string hsacoVersion;
  bool funcSupp = 0;      // enable function support
  bool rdc = 0;           // whether -fgpu-rdc is on
  string prevArg;
  // TODO(hipcc): convert toolArgs to an array rather than a string
  string toolArgs;
  string optArg;
  vector<string> options, inputs;
  // TODO(hipcc): hipcc uses --amdgpu-target for historical reasons.
  // It should be replaced by clang option --offload-arch.
  vector<string> targetOpts = {"--offload-arch=", "--amdgpu-target="};
  string targetsStr;
  bool skipOutputFile = false;
  const OsType& os = getOSInfo();
  string hip_compile_cxx_as_hip;
  if (var.hipCompileCxxAsHipEnv_.empty()) {
    hip_compile_cxx_as_hip = "1";
  } else {
    hip_compile_cxx_as_hip = var.hipCompileCxxAsHipEnv_;
  }
  string HIPLDARCHFLAGS;
  initializeHipCXXFlags();
  initializeHipCFlags();
  initializeHipLdFlags();
  string HIPCXXFLAGS, HIPCFLAGS, HIPLDFLAGS;
  HIPCFLAGS = getHipCFlags();
  HIPCXXFLAGS = getHipCXXFlags();
  HIPLDFLAGS = getHipLdFlags();
  string hipPath;
  hipPath = getHipPath();
  const PlatformInfo& platformInfo = getPlatformInfo();
  const string& nvccPath = getCompilerPath();
  const string& hipVersion = getHipVersion();
  if (verbose & 0x2) {
    cout << "HIP_PATH=" << hipPath << endl;
    cout << "HIP_PLATFORM=" <<  PlatformTypeStr(platformInfo.platform) <<endl;
    cout << "HIP_COMPILER=" << CompilerTypeStr(platformInfo.compiler) <<endl;
    cout << "HIP_RUNTIME=" << RuntimeTypeStr(platformInfo.runtime) <<endl;
    cout << "CUDA_PATH=" << nvccPath <<endl;
  }
  if (verbose & 0x4) {
    cout <<  "hipcc-args: ";
    for (unsigned int i = 1; i< argv.size(); i++) {
      cout <<  argv.at(i) << " ";
    }
    cout << endl;
  }
  // Handle code object generation
  string ISACMD;
  ISACMD += hipPath + "/bin/hipcc -ptx ";
  if (argv.at(1) == "--genco") {
    for (unsigned int i = 2; i < argv.size(); i++) {
      string isaarg = argv.at(i);
      ISACMD += " ";
      if (!hipBinUtilPtr_->substringPresent(isaarg,"--rocm-path=")) {
        ISACMD += isaarg;
      }
    }
    if (verbose & 0x1) {
      cout<< "hipcc-cmd: " << ISACMD << "\n";
    }
    system(ISACMD.c_str());
    exit(EXIT_SUCCESS);
  }
  for (unsigned int argcount = 1; argcount < argv.size(); argcount++) {
    // Save $arg, it can get changed in the loop.
    string arg = argv.at(argcount);
    regex toRemove("\\s+");
    // TODO(hipcc): figure out why this space removal is wanted.
    // TODO(hipcc): If someone has gone to the effort of quoting
    // the spaces to the shell
    // TODO(hipcc): why are we removing it here?
    string trimarg = hipBinUtilPtr_->replaceRegex(arg, toRemove, "");
    bool swallowArg = false;
    bool escapeArg = true;
    if (arg == "-c" || arg == "--genco" || arg == "-E") {
      compileOnly = true;
      needLDFLAGS  = false;
    }
    if (skipOutputFile) {
      // TODO(hipcc): handle filename with shell metacharacters
      toolArgs += " \"" + arg +"\"";
      prevArg = arg;
      skipOutputFile = 0;
      continue;
    }
    if (arg == "-o") {
      needLDFLAGS = 1;
      skipOutputFile = 1;
    }
    if ((trimarg == "-stdlib=libc++") && (setStdLib == 0)) {
      HIPCXXFLAGS += " -stdlib=libc++";
      setStdLib = 1;
    }
    // Check target selection option: --offload-arch= and --amdgpu-target=...
    for (unsigned int i = 0; i <targetOpts.size(); i++) {
      string targetOpt = targetOpts.at(i);
      string pattern = "^" + targetOpt + ".*";
      if (hipBinUtilPtr_->stringRegexMatch(arg, pattern)) {
        // If targets string is not empty, add a comma before
        // adding new target option value.
        targetsStr.size() >0 ? targetsStr += ",": targetsStr += "";
        targetsStr += arg.substr(targetOpt.size());
        default_amdgpu_target = 0;
      }
    }
    if (trimarg == "--version") {
      printHipVersion = 1;
    }
    if (trimarg == "--short-version") {
      printHipVersion = 1;
      runCmd = 0;
    }
    if (trimarg == "--cxxflags") {
      printCXXFlags = 1;
      runCmd = 0;
    }
    if (trimarg == "--ldflags") {
      printLDFlags = 1;
      runCmd = 0;
    }
    if (trimarg == "-M") {
      compileOnly = 1;
      buildDeps = 1;
    }
    if (trimarg == "-use_fast_math") {
      HIPCXXFLAGS += " -DHIP_FAST_MATH ";
      HIPCFLAGS += " -DHIP_FAST_MATH ";
    }
    if ((trimarg == "-use-staticlib") && (setLinkType == 0)) {
      linkType = 0;
      setLinkType = 1;
      swallowArg = 1;
    }
    if ((trimarg == "-use-sharedlib") && (setLinkType == 0)) {
      linkType = 1;
      setLinkType = 1;
    }
    if (hipBinUtilPtr_->stringRegexMatch(arg, "^-O.*")) {
      optArg = arg;
    }
    if (hipBinUtilPtr_->substringPresent(
                        arg, "--amdhsa-code-object-version=")) {
      arg = hipBinUtilPtr_->replaceStr(
                            arg, "--amdhsa-code-object-version=", "");
      hsacoVersion = arg;
      swallowArg = 1;
    }
    // nvcc does not handle standard compiler options properly
    // This can prevent hipcc being used as standard CXX/C Compiler
    // To fix this we need to pass -Xcompiler for options
    if (arg == "-fPIC" || hipBinUtilPtr_->substringPresent(arg, "-Wl,")) {
      HIPCXXFLAGS += " -Xcompiler "+ arg;
      swallowArg = 1;
    }
    if (arg == "-x") {
      fileTypeFlag = 1;
    } else if ((arg == "c" && prevArg == "-x") || (arg == "-xc")) {
      fileTypeFlag = 1;
      hasC = 1;
      hasCXX = 0;
      hasHIP = 0;
    } else if ((arg == "c++" && prevArg == "-x") || (arg == "-xc++")) {
      fileTypeFlag = 1;
      hasC = 0;
      hasCXX = 1;
      hasHIP = 0;
    } else if ((arg == "hip" && prevArg == "-x") || (arg == "-xhip")) {
      fileTypeFlag = 1;
      hasC = 0;
      hasCXX = 0;
      hasHIP = 1;
    } else if (hipBinUtilPtr_->substringPresent(arg, "-fopenmp-targets=")) {
      hasOMPTargets = 1;
    } else if (hipBinUtilPtr_->stringRegexMatch(arg, "^-.*")) {
      if  (arg == "-fgpu-rdc") {
        rdc = 1;
      } else if (arg == "-fno-gpu-rdc") {
        rdc = 0;
      }
      if (hipBinUtilPtr_->stringRegexMatch(arg, "^--hipcc.*")) {
        swallowArg = 1;
        if (arg == "--hipcc-func-supp") {
          funcSupp = 1;
        } else if (arg == "--hipcc-no-func-supp") {
          funcSupp = 0;
        }
      } else {
        options.push_back(arg);
      }
    } else if (prevArg != "-o") {
    if (fileTypeFlag == 0) {
      if (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.c$")) {
        hasC = 1;
        needCFLAGS = 1;
        toolArgs += " -x c";
      } else if ((hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cpp$")) ||
                 (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cxx$")) ||
                 (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cc$")) ||
                 (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.C$"))) {
        needCXXFLAGS = 1;
        hasCXX = 1;
      } else if (((hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cu$") ||
                   hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.cuh$")) &&
                   hip_compile_cxx_as_hip != "0") ||
                   (hipBinUtilPtr_->stringRegexMatch(arg, ".*\\.hip$"))) {
        needCXXFLAGS = 1;
        hasCU = 1;
      }
    }
    if (hasC) {
      needCFLAGS = 1;
    } else if (hasCXX || hasHIP) {
      needCXXFLAGS = 1;
    }
    inputs.push_back(arg);
    }
    // Windows needs different quoting, ignore for now
    if (os != windows && escapeArg) {
      regex reg("[^-a-zA-Z0-9_=+,.\\/]");
      arg = regex_replace(arg, reg, "\\$&");
    }
    if (!swallowArg)
      toolArgs += " " + arg;
    prevArg = arg;
  }  // end of for loop
  if (hasCXX) {
    HIPCXXFLAGS += " -x cu";
  }
  if (buildDeps) {
    HIPCXXFLAGS += " -M -D__CUDACC__";
    HIPCFLAGS += " -M -D__CUDACC__";
  }
  if (!var.hipccCompileFlagsAppendEnv_.empty()) {
    HIPCXXFLAGS += "\" " + var.hipccCompileFlagsAppendEnv_ + "\"";
    HIPCFLAGS += "\" " + var.hipccCompileFlagsAppendEnv_ + "\"";
  }
  if (!var.hipccLinkFlagsAppendEnv_.empty()) {
    HIPLDFLAGS += "\" " + var.hipccLinkFlagsAppendEnv_ + "\"";
  }
  string compiler;
  compiler = getHipCC();
  string CMD = compiler;
  if (needCFLAGS) {
    CMD += " " + HIPCFLAGS;
  }
  if (needCXXFLAGS) {
    CMD += " " + HIPCXXFLAGS;
  }
  if (needLDFLAGS && !compileOnly) {
    CMD += " " + HIPLDFLAGS;
  }
  CMD += " " + toolArgs;
  if (verbose & 0x1) {
    cout << "hipcc-cmd: " <<  CMD << "\n";
  }
  if (printHipVersion) {
    if (runCmd) {
      cout <<  "HIP version: ";
    }
    cout << hipVersion << endl;
  }
  if (printCXXFlags) {
    cout << HIPCXXFLAGS;
  }
  if (printLDFlags) {
    cout << HIPLDFLAGS;
  }
  if (runCmd) {
    SystemCmdOut sysOut;
    sysOut = hipBinUtilPtr_->exec(CMD.c_str(), true);
    string cmdOut = sysOut.out;
    int CMD_EXIT_CODE = sysOut.exitCode;
    if (CMD_EXIT_CODE !=0) {
      cout <<  "failed to execute:"  << CMD << std::endl;
    }
    exit(CMD_EXIT_CODE);
  }
}   // end of function


#endif  // SRC_HIPBIN_NVIDIA_H_
