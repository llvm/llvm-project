//===-- sanitizer_symbolizer_posix_libcdep.cc -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
// POSIX-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_POSIX
#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_linux.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_posix.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_symbolizer_internal.h"
#include "sanitizer_symbolizer_libbacktrace.h"
#include "sanitizer_symbolizer_mac.h"

#include <errno.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#if SANITIZER_MAC
#include <util.h>  // for forkpty()
#endif  // SANITIZER_MAC

// C++ demangling function, as required by Itanium C++ ABI. This is weak,
// because we do not require a C++ ABI library to be linked to a program
// using sanitizers; if it's not present, we'll just use the mangled name.
namespace __cxxabiv1 {
  extern "C" SANITIZER_WEAK_ATTRIBUTE
  char *__cxa_demangle(const char *mangled, char *buffer,
                                  size_t *length, int *status);
}

namespace __sanitizer {

// Attempts to demangle the name via __cxa_demangle from __cxxabiv1.
const char *DemangleCXXABI(const char *name) {
  // FIXME: __cxa_demangle aggressively insists on allocating memory.
  // There's not much we can do about that, short of providing our
  // own demangler (libc++abi's implementation could be adapted so that
  // it does not allocate). For now, we just call it anyway, and we leak
  // the returned value.
  if (__cxxabiv1::__cxa_demangle)
    if (const char *demangled_name =
          __cxxabiv1::__cxa_demangle(name, 0, 0, 0))
      return demangled_name;

  return name;
}

bool SymbolizerProcess::StartSymbolizerSubprocess() {
  if (!FileExists(path_)) {
    if (!reported_invalid_path_) {
      Report("WARNING: invalid path to external symbolizer!\n");
      reported_invalid_path_ = true;
    }
    return false;
  }

  int pid;
  if (use_forkpty_) {
#if SANITIZER_MAC
    fd_t fd = kInvalidFd;
    // Use forkpty to disable buffering in the new terminal.
    pid = internal_forkpty(&fd);
    if (pid == -1) {
      // forkpty() failed.
      Report("WARNING: failed to fork external symbolizer (errno: %d)\n",
             errno);
      return false;
    } else if (pid == 0) {
      // Child subprocess.
      const char *argv[kArgVMax];
      GetArgV(path_, argv);
      execv(path_, const_cast<char **>(&argv[0]));
      internal__exit(1);
    }

    // Continue execution in parent process.
    input_fd_ = output_fd_ = fd;

    // Disable echo in the new terminal, disable CR.
    struct termios termflags;
    tcgetattr(fd, &termflags);
    termflags.c_oflag &= ~ONLCR;
    termflags.c_lflag &= ~ECHO;
    tcsetattr(fd, TCSANOW, &termflags);
#else  // SANITIZER_MAC
    UNIMPLEMENTED();
#endif  // SANITIZER_MAC
  } else {
    int *infd = NULL;
    int *outfd = NULL;
    // The client program may close its stdin and/or stdout and/or stderr
    // thus allowing socketpair to reuse file descriptors 0, 1 or 2.
    // In this case the communication between the forked processes may be
    // broken if either the parent or the child tries to close or duplicate
    // these descriptors. The loop below produces two pairs of file
    // descriptors, each greater than 2 (stderr).
    int sock_pair[5][2];
    for (int i = 0; i < 5; i++) {
      if (pipe(sock_pair[i]) == -1) {
        for (int j = 0; j < i; j++) {
          internal_close(sock_pair[j][0]);
          internal_close(sock_pair[j][1]);
        }
        Report("WARNING: Can't create a socket pair to start "
               "external symbolizer (errno: %d)\n", errno);
        return false;
      } else if (sock_pair[i][0] > 2 && sock_pair[i][1] > 2) {
        if (infd == NULL) {
          infd = sock_pair[i];
        } else {
          outfd = sock_pair[i];
          for (int j = 0; j < i; j++) {
            if (sock_pair[j] == infd) continue;
            internal_close(sock_pair[j][0]);
            internal_close(sock_pair[j][1]);
          }
          break;
        }
      }
    }
    CHECK(infd);
    CHECK(outfd);

    // Real fork() may call user callbacks registered with pthread_atfork().
    pid = internal_fork();
    if (pid == -1) {
      // Fork() failed.
      internal_close(infd[0]);
      internal_close(infd[1]);
      internal_close(outfd[0]);
      internal_close(outfd[1]);
      Report("WARNING: failed to fork external symbolizer "
             " (errno: %d)\n", errno);
      return false;
    } else if (pid == 0) {
      // Child subprocess.
      internal_close(STDOUT_FILENO);
      internal_close(STDIN_FILENO);
      internal_dup2(outfd[0], STDIN_FILENO);
      internal_dup2(infd[1], STDOUT_FILENO);
      internal_close(outfd[0]);
      internal_close(outfd[1]);
      internal_close(infd[0]);
      internal_close(infd[1]);
      for (int fd = sysconf(_SC_OPEN_MAX); fd > 2; fd--)
        internal_close(fd);
      const char *argv[kArgVMax];
      GetArgV(path_, argv);
      execv(path_, const_cast<char **>(&argv[0]));
      internal__exit(1);
    }

    // Continue execution in parent process.
    internal_close(outfd[0]);
    internal_close(infd[1]);
    input_fd_ = infd[0];
    output_fd_ = outfd[1];
  }

  // Check that symbolizer subprocess started successfully.
  int pid_status;
  SleepForMillis(kSymbolizerStartupTimeMillis);
  int exited_pid = waitpid(pid, &pid_status, WNOHANG);
  if (exited_pid != 0) {
    // Either waitpid failed, or child has already exited.
    Report("WARNING: external symbolizer didn't start up correctly!\n");
    return false;
  }

  return true;
}

class Addr2LineProcess : public SymbolizerProcess {
 public:
  Addr2LineProcess(const char *path, const char *module_name)
      : SymbolizerProcess(path), module_name_(internal_strdup(module_name)) {}

  const char *module_name() const { return module_name_; }

 private:
  void GetArgV(const char *path_to_binary,
               const char *(&argv)[kArgVMax]) const override {
    int i = 0;
    argv[i++] = path_to_binary;
    argv[i++] = "-iCfe";
    argv[i++] = module_name_;
    argv[i++] = nullptr;
  }

  bool ReachedEndOfOutput(const char *buffer, uptr length) const override;

  bool ReadFromSymbolizer(char *buffer, uptr max_length) override {
    if (!SymbolizerProcess::ReadFromSymbolizer(buffer, max_length))
      return false;
    // We should cut out output_terminator_ at the end of given buffer,
    // appended by addr2line to mark the end of its meaningful output.
    // We cannot scan buffer from it's beginning, because it is legal for it
    // to start with output_terminator_ in case given offset is invalid. So,
    // scanning from second character.
    char *garbage = internal_strstr(buffer + 1, output_terminator_);
    // This should never be NULL since buffer must end up with
    // output_terminator_.
    CHECK(garbage);
    // Trim the buffer.
    garbage[0] = '\0';
    return true;
  }

  const char *module_name_;  // Owned, leaked.
  static const char output_terminator_[];
};

const char Addr2LineProcess::output_terminator_[] = "??\n??:0\n";

bool Addr2LineProcess::ReachedEndOfOutput(const char *buffer,
                                          uptr length) const {
  const size_t kTerminatorLen = sizeof(output_terminator_) - 1;
  // Skip, if we read just kTerminatorLen bytes, because Addr2Line output
  // should consist at least of two pairs of lines:
  // 1. First one, corresponding to given offset to be symbolized
  // (may be equal to output_terminator_, if offset is not valid).
  // 2. Second one for output_terminator_, itself to mark the end of output.
  if (length <= kTerminatorLen) return false;
  // Addr2Line output should end up with output_terminator_.
  return !internal_memcmp(buffer + length - kTerminatorLen,
                          output_terminator_, kTerminatorLen);
}

class Addr2LinePool : public SymbolizerTool {
 public:
  explicit Addr2LinePool(const char *addr2line_path,
                         LowLevelAllocator *allocator)
      : addr2line_path_(addr2line_path), allocator_(allocator),
        addr2line_pool_(16) {}

  bool SymbolizePC(uptr addr, SymbolizedStack *stack) override {
    if (const char *buf =
            SendCommand(stack->info.module, stack->info.module_offset)) {
      ParseSymbolizePCOutput(buf, stack);
      return true;
    }
    return false;
  }

  bool SymbolizeData(uptr addr, DataInfo *info) override {
    return false;
  }

 private:
  const char *SendCommand(const char *module_name, uptr module_offset) {
    Addr2LineProcess *addr2line = 0;
    for (uptr i = 0; i < addr2line_pool_.size(); ++i) {
      if (0 ==
          internal_strcmp(module_name, addr2line_pool_[i]->module_name())) {
        addr2line = addr2line_pool_[i];
        break;
      }
    }
    if (!addr2line) {
      addr2line =
          new(*allocator_) Addr2LineProcess(addr2line_path_, module_name);
      addr2line_pool_.push_back(addr2line);
    }
    CHECK_EQ(0, internal_strcmp(module_name, addr2line->module_name()));
    char buffer[kBufferSize];
    internal_snprintf(buffer, kBufferSize, "0x%zx\n0x%zx\n",
                      module_offset, dummy_address_);
    return addr2line->SendCommand(buffer);
  }

  static const uptr kBufferSize = 64;
  const char *addr2line_path_;
  LowLevelAllocator *allocator_;
  InternalMmapVector<Addr2LineProcess*> addr2line_pool_;
  static const uptr dummy_address_ =
      FIRST_32_SECOND_64(UINT32_MAX, UINT64_MAX);
};

#if SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
bool __sanitizer_symbolize_code(const char *ModuleName, u64 ModuleOffset,
                                char *Buffer, int MaxLength);
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
bool __sanitizer_symbolize_data(const char *ModuleName, u64 ModuleOffset,
                                char *Buffer, int MaxLength);
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void __sanitizer_symbolize_flush();
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
int __sanitizer_symbolize_demangle(const char *Name, char *Buffer,
                                   int MaxLength);
}  // extern "C"

class InternalSymbolizer : public SymbolizerTool {
 public:
  static InternalSymbolizer *get(LowLevelAllocator *alloc) {
    if (__sanitizer_symbolize_code != 0 &&
        __sanitizer_symbolize_data != 0) {
      return new(*alloc) InternalSymbolizer();
    }
    return 0;
  }

  bool SymbolizePC(uptr addr, SymbolizedStack *stack) override {
    bool result = __sanitizer_symbolize_code(
        stack->info.module, stack->info.module_offset, buffer_, kBufferSize);
    if (result) ParseSymbolizePCOutput(buffer_, stack);
    return result;
  }

  bool SymbolizeData(uptr addr, DataInfo *info) override {
    bool result = __sanitizer_symbolize_data(info->module, info->module_offset,
                                             buffer_, kBufferSize);
    if (result) {
      ParseSymbolizeDataOutput(buffer_, info);
      info->start += (addr - info->module_offset);  // Add the base address.
    }
    return result;
  }

  void Flush() override {
    if (__sanitizer_symbolize_flush)
      __sanitizer_symbolize_flush();
  }

  const char *Demangle(const char *name) override {
    if (__sanitizer_symbolize_demangle) {
      for (uptr res_length = 1024;
           res_length <= InternalSizeClassMap::kMaxSize;) {
        char *res_buff = static_cast<char*>(InternalAlloc(res_length));
        uptr req_length =
            __sanitizer_symbolize_demangle(name, res_buff, res_length);
        if (req_length > res_length) {
          res_length = req_length + 1;
          InternalFree(res_buff);
          continue;
        }
        return res_buff;
      }
    }
    return name;
  }

 private:
  InternalSymbolizer() { }

  static const int kBufferSize = 16 * 1024;
  static const int kMaxDemangledNameSize = 1024;
  char buffer_[kBufferSize];
};
#else  // SANITIZER_SUPPORTS_WEAK_HOOKS

class InternalSymbolizer : public SymbolizerTool {
 public:
  static InternalSymbolizer *get(LowLevelAllocator *alloc) { return 0; }
};

#endif  // SANITIZER_SUPPORTS_WEAK_HOOKS

const char *Symbolizer::PlatformDemangle(const char *name) {
  return DemangleCXXABI(name);
}

void Symbolizer::PlatformPrepareForSandboxing() {}

static SymbolizerTool *ChooseExternalSymbolizer(LowLevelAllocator *allocator) {
  const char *path = common_flags()->external_symbolizer_path;
  const char *binary_name = path ? StripModuleName(path) : "";
  if (path && path[0] == '\0') {
    VReport(2, "External symbolizer is explicitly disabled.\n");
    return nullptr;
  } else if (!internal_strcmp(binary_name, "llvm-symbolizer")) {
    VReport(2, "Using llvm-symbolizer at user-specified path: %s\n", path);
    return new(*allocator) LLVMSymbolizer(path, allocator);
  } else if (!internal_strcmp(binary_name, "atos")) {
#if SANITIZER_MAC
    VReport(2, "Using atos at user-specified path: %s\n", path);
    return new(*allocator) AtosSymbolizer(path, allocator);
#else  // SANITIZER_MAC
    Report("ERROR: Using `atos` is only supported on Darwin.\n");
    Die();
#endif  // SANITIZER_MAC
  } else if (!internal_strcmp(binary_name, "addr2line")) {
    VReport(2, "Using addr2line at user-specified path: %s\n", path);
    return new(*allocator) Addr2LinePool(path, allocator);
  } else if (path) {
    Report("ERROR: External symbolizer path is set to '%s' which isn't "
           "a known symbolizer. Please set the path to the llvm-symbolizer "
           "binary or other known tool.\n", path);
    Die();
  }

  // Otherwise symbolizer program is unknown, let's search $PATH
  CHECK(path == nullptr);
  if (const char *found_path = FindPathToBinary("llvm-symbolizer")) {
    VReport(2, "Using llvm-symbolizer found at: %s\n", found_path);
    return new(*allocator) LLVMSymbolizer(found_path, allocator);
  }
#if SANITIZER_MAC
  if (const char *found_path = FindPathToBinary("atos")) {
    VReport(2, "Using atos found at: %s\n", found_path);
    return new(*allocator) AtosSymbolizer(found_path, allocator);
  }
#endif  // SANITIZER_MAC
  if (common_flags()->allow_addr2line) {
    if (const char *found_path = FindPathToBinary("addr2line")) {
      VReport(2, "Using addr2line found at: %s\n", found_path);
      return new(*allocator) Addr2LinePool(found_path, allocator);
    }
  }
  return nullptr;
}

static void ChooseSymbolizerTools(IntrusiveList<SymbolizerTool> *list,
                                  LowLevelAllocator *allocator) {
  if (!common_flags()->symbolize) {
    VReport(2, "Symbolizer is disabled.\n");
    return;
  }
  if (SymbolizerTool *tool = InternalSymbolizer::get(allocator)) {
    VReport(2, "Using internal symbolizer.\n");
    list->push_back(tool);
    return;
  }
  if (SymbolizerTool *tool = LibbacktraceSymbolizer::get(allocator)) {
    VReport(2, "Using libbacktrace symbolizer.\n");
    list->push_back(tool);
    return;
  }

  if (SymbolizerTool *tool = ChooseExternalSymbolizer(allocator)) {
    list->push_back(tool);
  }

#if SANITIZER_MAC
  VReport(2, "Using dladdr symbolizer.\n");
  list->push_back(new(*allocator) DlAddrSymbolizer());
#endif  // SANITIZER_MAC
}

Symbolizer *Symbolizer::PlatformInit() {
  IntrusiveList<SymbolizerTool> list;
  list.clear();
  ChooseSymbolizerTools(&list, &symbolizer_allocator_);
  return new(symbolizer_allocator_) Symbolizer(list);
}

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
