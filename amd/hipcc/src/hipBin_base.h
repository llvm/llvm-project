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
#ifndef SRC_HIPBIN_BASE_H_
#define SRC_HIPBIN_BASE_H_


#include "hipBin_util.h"
#include <iostream>
#include <vector>
#include <string>

// All envirnoment variables used in the code
# define PATH                       "PATH"
# define HIP_ROCCLR_HOME            "HIP_ROCCLR_HOME"
# define HIP_PATH                   "HIP_PATH"
# define ROCM_PATH                  "ROCM_PATH"
# define CUDA_PATH                  "CUDA_PATH"
# define HSA_PATH                   "HSA_PATH"
# define HIP_CLANG_PATH             "HIP_CLANG_PATH"
# define HIP_PLATFORM               "HIP_PLATFORM"
# define HIP_COMPILER               "HIP_COMPILER"
# define HIP_RUNTIME                "HIP_RUNTIME"
# define LD_LIBRARY_PATH            "LD_LIBRARY_PATH"

// hipcc
# define HIPCC_COMPILE_FLAGS_APPEND     "HIPCC_COMPILE_FLAGS_APPEND"
# define HIPCC_LINK_FLAGS_APPEND        "HIPCC_LINK_FLAGS_APPEND"
# define HIP_LIB_PATH                   "HIP_LIB_PATH"
# define DEVICE_LIB_PATH                "DEVICE_LIB_PATH"
# define HIP_CLANG_HCC_COMPAT_MODE      "HIP_CLANG_HCC_COMPAT_MODE"
# define HIP_COMPILE_CXX_AS_HIP         "HIP_COMPILE_CXX_AS_HIP"
# define HIPCC_VERBOSE                  "HIPCC_VERBOSE"
# define HCC_AMDGPU_TARGET              "HCC_AMDGPU_TARGET"

# define HIP_BASE_VERSION_MAJOR     "4"
# define HIP_BASE_VERSION_MINOR     "4"
# define HIP_BASE_VERSION_PATCH     "0"
# define HIP_BASE_VERSION_GITHASH   "0"


enum PlatformType {
  amd = 0,
  nvidia,
  // add new platform types to be added here
};

string PlatformTypeStr(PlatformType platform) {
  switch (platform) {
  case amd:
    return "amd";
  case nvidia:
    return "nvidia";
  // add new platform types to be added here
  default:
    return "invalid platform";
  }
}

enum CompilerType {
  clang = 0,
  nvcc
  // add new compiler types to be added here
};


string CompilerTypeStr(CompilerType compiler) {
  switch (compiler) {
  case clang:
    return "clang";
  case nvcc:
    return "nvcc";
  // add new compiler types to be added here
  default:
    return "invalid CompilerType";
  }
}


enum RuntimeType {
  rocclr = 0,
  cuda
  // add new runtime types to be added here
};

string RuntimeTypeStr(RuntimeType runtime) {
  switch (runtime) {
  case rocclr:
    return "rocclr";
  case cuda:
    return "cuda";
  // add new runtime types to be added here
  default:
    return "invalid RuntimeType";
  }
}

enum OsType {
  lnx = 0,
  windows
  // add new OS types to be added here
};

string OsTypeStr(OsType os) {
  switch (os) {
  case lnx:
    return "linux";
  case windows:
    return "windows";
  // add new OS types to be added here
  default:
    return "invalid OsType";
  }
}

struct PlatformInfo {
  PlatformType platform;
  CompilerType compiler;
  RuntimeType runtime;
  OsType os;
};

struct EnvVariables {
  string path_ = "";
  string hipPathEnv_ = "";
  string hipRocclrPathEnv_ = "";
  string roccmPathEnv_ = "";
  string cudaPathEnv_ = "";
  string hsaPathEnv_ = "";
  string hipClangPathEnv_ = "";
  string hipPlatformEnv_ = "";
  string hipCompilerEnv_ = "";
  string hipRuntimeEnv_ = "";
  string ldLibraryPathEnv_ = "";
  string verboseEnv_ = "";
  string hipccCompileFlagsAppendEnv_ = "";
  string hipccLinkFlagsAppendEnv_ = "";
  string hipLibPathEnv_ = "";
  string deviceLibPathEnv_ = "";
  string hipClangHccCompactModeEnv_ = "";
  string hipCompileCxxAsHipEnv_ = "";
  string hccAmdGpuTargetEnv_ = "";
  friend std::ostream& operator <<(std::ostream& os, const EnvVariables& var) {
    os << "Path: "                           << var.path_ << endl;
    os << "Hip Path: "                       << var.hipPathEnv_ << endl;
    os << "Hip Rocclr Path: "                << var.hipRocclrPathEnv_ << endl;
    os << "Roccm Path: "                     << var.roccmPathEnv_ << endl;
    os << "Cuda Path: "                      << var.cudaPathEnv_ << endl;
    os << "Hsa Path: "                       << var.hsaPathEnv_ << endl;
    os << "Hip Clang Path: "                 << var.hipClangPathEnv_ << endl;
    os << "Hip Platform: "                   << var.hipPlatformEnv_ << endl;
    os << "Hip Compiler: "                   << var.hipCompilerEnv_ << endl;
    os << "Hip Runtime: "                    << var.hipRuntimeEnv_ << endl;
    os << "LD Library Path: "                << var.ldLibraryPathEnv_ << endl;
    os << "Verbose: "                        << var.verboseEnv_ << endl;
    os << "Hipcc Compile Flags Append: "     <<
           var.hipccCompileFlagsAppendEnv_ << endl;
    os << "Hipcc Link Flags Append: "        <<
           var.hipccLinkFlagsAppendEnv_ << endl;
    os << "Hip lib Path: "                   << var.hipLibPathEnv_ << endl;
    os << "Device lib Path: "                << var.deviceLibPathEnv_ << endl;
    os << "Hip Clang HCC Compact mode: "     <<
           var.hipClangHccCompactModeEnv_ << endl;
    os << "Hip Compile Cxx as Hip: "         <<
           var.hipCompileCxxAsHipEnv_ << endl;
    os << "Hcc Amd Gpu Target: "             << var.hccAmdGpuTargetEnv_ << endl;
    return os;
  }
};

enum HipBinCommand {
  unknown = -1,
  path,
  roccmpath,
  cpp_config,
  compiler,
  platform,
  runtime,
  hipclangpath,
  full,
  version,
  check,
  newline,
  help,
};



class HipBinBase {
 public:
  HipBinBase();
  // Interface functions
  virtual void constructCompilerPath() = 0;
  virtual void printFull() = 0;
  virtual bool detectPlatform() = 0;
  virtual const string& getCompilerPath() const = 0;
  virtual void printCompilerInfo() const = 0;
  virtual string getCompilerVersion() = 0;
  virtual const PlatformInfo& getPlatformInfo() const = 0;
  virtual string getCppConfig() = 0;
  virtual void checkHipconfig() = 0;
  virtual string getDeviceLibPath() const = 0;
  virtual string getHipLibPath() const = 0;
  virtual string getHipCC() const = 0;
  virtual string getHipInclude() const = 0;
  virtual void initializeHipCXXFlags() = 0;
  virtual void initializeHipCFlags() = 0;
  virtual void initializeHipLdFlags() = 0;
  virtual const string& getHipCXXFlags() const = 0;
  virtual const string& getHipCFlags() const = 0;
  virtual const string& getHipLdFlags() const = 0;
  virtual void executeHipCCCmd(vector<string> argv) = 0;
  // Common functions used by all platforms
  void getSystemInfo() const;
  void printEnvironmentVariables() const;
  const EnvVariables& getEnvVariables() const;
  const OsType& getOSInfo() const;
  const string& getHipPath() const;
  const string& getRoccmPath() const;
  const string& getHipVersion() const;
  void printUsage() const;
  bool canRunCompiler(string exeName, string& cmdOut);
  HipBinCommand gethipconfigCmd(string argument);
  const string& getrocm_pathOption() const;

 protected:
  // hipBinUtilPtr used by derived platforms
  // so therefore its protected
  HipBinUtil* hipBinUtilPtr_;
  string rocm_pathOption_ = "";
  void readOSInfo();
  void readEnvVariables();
  void constructHipPath();
  void constructRoccmPath();
  void readHipVersion();
  
 private:
  EnvVariables envVariables_, variables_;
  OsType osInfo_;
  string hipVersion_;

};

HipBinBase::HipBinBase() {
  hipBinUtilPtr_ = hipBinUtilPtr_->getInstance();
  readOSInfo();                 // detects if windows or linux
  readEnvVariables();           // reads the environment variables
}

// detects the OS information
void HipBinBase::readOSInfo() {
#if defined _WIN32 || defined  _WIN64
  osInfo_ = windows;
#elif  defined __unix || defined __linux__
  osInfo_ = lnx;
#endif
}


// reads envirnoment variables
void HipBinBase::readEnvVariables() {
  if (const char* path = std::getenv(PATH))
    envVariables_.path_ = path;
  if (const char* hip = std::getenv(HIP_PATH))
    envVariables_.hipPathEnv_ = hip;
  if (const char* hip_rocclr = std::getenv(HIP_ROCCLR_HOME))
    envVariables_.hipRocclrPathEnv_ = hip_rocclr;
  if (const char* roccm = std::getenv(ROCM_PATH))
    envVariables_.roccmPathEnv_ = roccm;
  if (const char* cuda = std::getenv(CUDA_PATH))
    envVariables_.cudaPathEnv_ = cuda;
  if (const char* hsa = std::getenv(HSA_PATH))
    envVariables_.hsaPathEnv_ = hsa;
  if (const char* hipClang = std::getenv(HIP_CLANG_PATH))
    envVariables_.hipClangPathEnv_ = hipClang;
  if (const char* hipPlatform = std::getenv(HIP_PLATFORM))
    envVariables_.hipPlatformEnv_ = hipPlatform;
  if (const char* hipCompiler = std::getenv(HIP_COMPILER))
    envVariables_.hipCompilerEnv_ = hipCompiler;
  if (const char* hipRuntime = std::getenv(HIP_RUNTIME))
    envVariables_.hipRuntimeEnv_ = hipRuntime;
  if (const char* ldLibaryPath = std::getenv(LD_LIBRARY_PATH))
    envVariables_.ldLibraryPathEnv_ = ldLibaryPath;
  if (const char* hccAmdGpuTarget = std::getenv(HCC_AMDGPU_TARGET))
    envVariables_.hccAmdGpuTargetEnv_ = hccAmdGpuTarget;
  if (const char* verbose = std::getenv(HIPCC_VERBOSE))
    envVariables_.verboseEnv_ = verbose;
  if (const char* hipccCompileFlagsAppend =
      std::getenv(HIPCC_COMPILE_FLAGS_APPEND))
    envVariables_.hipccCompileFlagsAppendEnv_ = hipccCompileFlagsAppend;
  if (const char* hipccLinkFlagsAppend = std::getenv(HIPCC_LINK_FLAGS_APPEND))
    envVariables_.hipccLinkFlagsAppendEnv_ = hipccLinkFlagsAppend;
  if (const char* hipLibPath = std::getenv(HIP_LIB_PATH))
    envVariables_.hipLibPathEnv_ = hipLibPath;
  if (const char* deviceLibPath = std::getenv(DEVICE_LIB_PATH))
    envVariables_.deviceLibPathEnv_ = deviceLibPath;
  if (const char* hipClangHccCompactMode =
      std::getenv(HIP_CLANG_HCC_COMPAT_MODE))
    envVariables_.hipClangHccCompactModeEnv_ = hipClangHccCompactMode;
  if (const char* hipCompileCxxAsHip = std::getenv(HIP_COMPILE_CXX_AS_HIP))
    envVariables_.hipCompileCxxAsHipEnv_ = hipCompileCxxAsHip;
}

// constructs the HIP path
void HipBinBase::constructHipPath() {
  fs::path full_path(hipBinUtilPtr_->getSelfPath());
  if (envVariables_.hipPathEnv_.empty())
    variables_.hipPathEnv_ = (full_path.parent_path()).string();
  else
    variables_.hipPathEnv_ = envVariables_.hipPathEnv_;
}


// constructs the ROCM path
void HipBinBase::constructRoccmPath() {
  // we need to use --rocm-path option
  string rocm_path_name = getrocm_pathOption();

  // chose the --rocm-path option first, if specified.
  if (!rocm_path_name.empty())
    variables_.roccmPathEnv_ = rocm_path_name;
  else if (envVariables_.roccmPathEnv_.empty()) {
    const string& hipPath = getHipPath();
    fs::path roccm_path(hipPath);
    roccm_path = roccm_path.parent_path();
    fs::path rocm_agent_enumerator_file(roccm_path);
    rocm_agent_enumerator_file /= "bin/rocm_agent_enumerator";
    if (!fs::exists(rocm_agent_enumerator_file)) {
      roccm_path = "/opt/rocm";
    }
  } else {
    variables_.roccmPathEnv_ = envVariables_.roccmPathEnv_;}
}

// reads the Hip Version
void HipBinBase::readHipVersion() {
  string hipVersion;
  const string& hipPath = getHipPath();
  fs::path hipVersionPath = hipPath;
  hipVersionPath /= "bin/.hipVersion";
  map<string, string> hipVersionMap;
  hipVersionMap = hipBinUtilPtr_->parseConfigFile(hipVersionPath);
  string hip_version_major, hip_version_minor,
         hip_version_patch, hip_version_githash;
  hip_version_major = hipBinUtilPtr_->readConfigMap(
                      hipVersionMap, "HIP_VERSION_MAJOR",
                      HIP_BASE_VERSION_MAJOR);
  hip_version_minor = hipBinUtilPtr_->readConfigMap(
                      hipVersionMap, "HIP_VERSION_MINOR",
                      HIP_BASE_VERSION_MINOR);
  hip_version_patch = hipBinUtilPtr_->readConfigMap(
                      hipVersionMap, "HIP_VERSION_PATCH",
                      HIP_BASE_VERSION_PATCH);
  hip_version_githash = hipBinUtilPtr_->readConfigMap(
                      hipVersionMap, "HIP_VERSION_GITHASH",
                      HIP_BASE_VERSION_GITHASH);
  hipVersion = hip_version_major + "." + hip_version_minor +
               "." + hip_version_patch + "-" + hip_version_githash;
  hipVersion_ = hipVersion;
}

// prints system information
void HipBinBase::getSystemInfo() const {
  const OsType& os = getOSInfo();
  if (os == windows) {
    cout << endl << "== Windows Display Drivers" << endl;
    cout << "Hostname      :";
    system("hostname");
    system("wmic path win32_VideoController get AdapterCompatibility,"
    "InstalledDisplayDrivers,Name | findstr /B /C:\"Advanced Micro Devices\"");
  } else {
    assert(os == lnx);
    cout << endl << "== Linux Kernel" << endl;
    cout << "Hostname      :" << endl;
    system("hostname");
    system("uname -a");
  }
}

// prints the envirnoment variables
void HipBinBase::printEnvironmentVariables() const {
  const OsType& os = getOSInfo();
  if (os == windows) {
    cout << "PATH=" << envVariables_.path_ << "\n" << endl;
    system("set | findstr"
    " /B /C:\"HIP\" /C:\"HSA\" /C:\"CUDA\" /C:\"LD_LIBRARY_PATH\"");
  } else {
    string cmd = "echo PATH =";
    cmd += envVariables_.path_;
    system(cmd.c_str());
    system("env | egrep '^HIP|^HSA|^CUDA|^LD_LIBRARY_PATH'");
  }
}

// returns envirnoment variables
const EnvVariables& HipBinBase::getEnvVariables() const {
  return envVariables_;
}


// returns the os information
const OsType& HipBinBase::getOSInfo() const {
  return osInfo_;
}

// returns the HIP path
const string& HipBinBase::getHipPath() const {
  return variables_.hipPathEnv_;
}

// returns the Roccm path
const string& HipBinBase::getRoccmPath() const {
  return variables_.roccmPathEnv_;
}

// returns the Hip Version
const string& HipBinBase::getHipVersion() const {
  return hipVersion_;
}

// prints the help text
void HipBinBase::printUsage() const {
  cout << "usage: hipconfig [OPTIONS]\n";
  cout << "  --path,  -p        :"
  " print HIP_PATH (use env var if set, else determine from hipconfig path)\n";
  cout << "  --rocmpath,  -R    :"
  " print ROCM_PATH (use env var if set,"
  " else determine from hip path or /opt/rocm)\n";
  cout << "  --cpp_config, -C   : print C++ compiler options\n";
  cout << "  --compiler, -c     : print compiler (clang or nvcc)\n";
  cout << "  --platform, -P     : print platform (amd or nvidia)\n";
  cout << "  --runtime, -r      : print runtime (rocclr or cuda)\n";
  cout << "  --hipclangpath, -l : print HIP_CLANG_PATH\n";
  cout << "  --full, -f         : print full config\n";
  cout << "  --version, -v      : print hip version\n";
  cout << "  --check            : check configuration\n";
  cout << "  --newline, -n      : print newline\n";
  cout << "  --help, -h         : print help message\n";
}



// compiler canRun or not
bool HipBinBase::canRunCompiler(string exeName, string& cmdOut) {
  string complierName = exeName;
  string temp_dir = hipBinUtilPtr_->getTempDir();
  fs::path templateFs = temp_dir;
  templateFs /= "canRunXXXXXX";
  string tmpFileName = hipBinUtilPtr_->mktempFile(templateFs.string());
  complierName += " --version > " + tmpFileName + " 2>&1";
  bool executable = false;
  if (system(const_cast<char*>(complierName.c_str()))) {
    executable = false;
  } else {
    string myline;
    ifstream fp;
    fp.open(tmpFileName);
    if (fp.is_open()) {
      while (std::getline(fp, myline)) {
        cmdOut += myline;
      }
    }
    fp.close();
    executable = true;
  }
  return executable;
}

HipBinCommand HipBinBase::gethipconfigCmd(string argument) {
  vector<string> pathStrs = { "-p", "--path", "-path", "--p" };
  if (hipBinUtilPtr_->checkCmd(pathStrs, argument))
    return path;
  vector<string> rocmPathStrs = { "-R", "--rocmpath", "-rocmpath", "--R" };
  if (hipBinUtilPtr_->checkCmd(rocmPathStrs, argument))
    return roccmpath;
  vector<string> cppConfigStrs = { "-C", "--cpp_config",
                                   "-cpp_config", "--C", };
  if (hipBinUtilPtr_->checkCmd(cppConfigStrs, argument))
    return cpp_config;
  vector<string> CompilerStrs = { "-c", "--compiler", "-compiler", "--c" };
  if (hipBinUtilPtr_->checkCmd(CompilerStrs, argument))
    return compiler;
  vector<string> platformStrs = { "-P", "--platform", "-platform", "--P" };
  if (hipBinUtilPtr_->checkCmd(platformStrs, argument))
    return platform;
  vector<string> runtimeStrs = { "-r", "--runtime", "-runtime", "--r" };
  if (hipBinUtilPtr_->checkCmd(runtimeStrs, argument))
    return runtime;
  vector<string> hipClangPathStrs = { "-l", "--hipclangpath",
                                      "-hipclangpath", "--l" };
  if (hipBinUtilPtr_->checkCmd(hipClangPathStrs, argument))
    return hipclangpath;
  vector<string> fullStrs = { "-f", "--full", "-full", "--f" };
  if (hipBinUtilPtr_->checkCmd(fullStrs, argument))
    return full;
  vector<string> versionStrs = { "-v", "--version", "-version", "--v" };
  if (hipBinUtilPtr_->checkCmd(versionStrs, argument))
    return version;
  vector<string> checkStrs = { "--check", "-check" };
  if (hipBinUtilPtr_->checkCmd(checkStrs, argument))
    return check;
  vector<string> newlineStrs = { "--n", "-n", "--newline", "-newline" };
  if (hipBinUtilPtr_->checkCmd(newlineStrs, argument))
    return newline;
  vector<string> helpStrs = { "-h", "--help", "-help", "--h" };
  if (hipBinUtilPtr_->checkCmd(helpStrs, argument))
    return help;
  return full;  // default is full. return full if no commands are matched
}

const  string& HipBinBase::getrocm_pathOption() const {
  return rocm_pathOption_;
}

#endif  // SRC_HIPBIN_BASE_H_
