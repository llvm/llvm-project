//===-- sanitizer_symbolizer_libcdep.cc -----------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator_internal.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_symbolizer_internal.h"

namespace __sanitizer {

const char *ExtractToken(const char *str, const char *delims, char **result) {
  uptr prefix_len = internal_strcspn(str, delims);
  *result = (char*)InternalAlloc(prefix_len + 1);
  internal_memcpy(*result, str, prefix_len);
  (*result)[prefix_len] = '\0';
  const char *prefix_end = str + prefix_len;
  if (*prefix_end != '\0') prefix_end++;
  return prefix_end;
}

const char *ExtractInt(const char *str, const char *delims, int *result) {
  char *buff;
  const char *ret = ExtractToken(str, delims, &buff);
  if (buff != 0) {
    *result = (int)internal_atoll(buff);
  }
  InternalFree(buff);
  return ret;
}

const char *ExtractUptr(const char *str, const char *delims, uptr *result) {
  char *buff;
  const char *ret = ExtractToken(str, delims, &buff);
  if (buff != 0) {
    *result = (uptr)internal_atoll(buff);
  }
  InternalFree(buff);
  return ret;
}

const char *ExtractTokenUpToDelimiter(const char *str, const char *delimiter,
                                      char **result) {
  const char *found_delimiter = internal_strstr(str, delimiter);
  uptr prefix_len =
      found_delimiter ? found_delimiter - str : internal_strlen(str);
  *result = (char *)InternalAlloc(prefix_len + 1);
  internal_memcpy(*result, str, prefix_len);
  (*result)[prefix_len] = '\0';
  const char *prefix_end = str + prefix_len;
  if (*prefix_end != '\0') prefix_end += internal_strlen(delimiter);
  return prefix_end;
}

SymbolizedStack *Symbolizer::SymbolizePC(uptr addr) {
  BlockingMutexLock l(&mu_);
  const char *module_name;
  uptr module_offset;
  SymbolizedStack *res = SymbolizedStack::New(addr);
  if (!FindModuleNameAndOffsetForAddress(addr, &module_name, &module_offset))
    return res;
  // Always fill data about module name and offset.
  res->info.FillModuleInfo(module_name, module_offset);
  for (auto &tool : tools_) {
    SymbolizerScope sym_scope(this);
    if (tool.SymbolizePC(addr, res)) {
      return res;
    }
  }
  return res;
}

bool Symbolizer::SymbolizeData(uptr addr, DataInfo *info) {
  BlockingMutexLock l(&mu_);
  const char *module_name;
  uptr module_offset;
  if (!FindModuleNameAndOffsetForAddress(addr, &module_name, &module_offset))
    return false;
  info->Clear();
  info->module = internal_strdup(module_name);
  info->module_offset = module_offset;
  for (auto &tool : tools_) {
    SymbolizerScope sym_scope(this);
    if (tool.SymbolizeData(addr, info)) {
      return true;
    }
  }
  return true;
}

bool Symbolizer::GetModuleNameAndOffsetForPC(uptr pc, const char **module_name,
                                             uptr *module_address) {
  BlockingMutexLock l(&mu_);
  const char *internal_module_name = nullptr;
  if (!FindModuleNameAndOffsetForAddress(pc, &internal_module_name,
                                         module_address))
    return false;

  if (module_name)
    *module_name = module_names_.GetOwnedCopy(internal_module_name);
  return true;
}

void Symbolizer::Flush() {
  BlockingMutexLock l(&mu_);
  for (auto &tool : tools_) {
    SymbolizerScope sym_scope(this);
    tool.Flush();
  }
}

const char *Symbolizer::Demangle(const char *name) {
  BlockingMutexLock l(&mu_);
  for (auto &tool : tools_) {
    SymbolizerScope sym_scope(this);
    if (const char *demangled = tool.Demangle(name))
      return demangled;
  }
  return PlatformDemangle(name);
}

void Symbolizer::PrepareForSandboxing() {
  BlockingMutexLock l(&mu_);
  PlatformPrepareForSandboxing();
}

bool Symbolizer::FindModuleNameAndOffsetForAddress(uptr address,
                                                   const char **module_name,
                                                   uptr *module_offset) {
  const LoadedModule *module = FindModuleForAddress(address);
  if (module == nullptr)
    return false;
  *module_name = module->full_name();
  *module_offset = address - module->base_address();
  return true;
}

const LoadedModule *Symbolizer::FindModuleForAddress(uptr address) {
  bool modules_were_reloaded = false;
  if (!modules_fresh_) {
    modules_.init();
    RAW_CHECK(modules_.size() > 0);
    modules_fresh_ = true;
    modules_were_reloaded = true;
  }
  for (uptr i = 0; i < modules_.size(); i++) {
    if (modules_[i].containsAddress(address)) {
      return &modules_[i];
    }
  }
  // Reload the modules and look up again, if we haven't tried it yet.
  if (!modules_were_reloaded) {
    // FIXME: set modules_fresh_ from dlopen()/dlclose() interceptors.
    // It's too aggressive to reload the list of modules each time we fail
    // to find a module for a given address.
    modules_fresh_ = false;
    return FindModuleForAddress(address);
  }
  return 0;
}

Symbolizer *Symbolizer::GetOrInit() {
  SpinMutexLock l(&init_mu_);
  if (symbolizer_)
    return symbolizer_;
  symbolizer_ = PlatformInit();
  CHECK(symbolizer_);
  return symbolizer_;
}

// For now we assume the following protocol:
// For each request of the form
//   <module_name> <module_offset>
// passed to STDIN, external symbolizer prints to STDOUT response:
//   <function_name>
//   <file_name>:<line_number>:<column_number>
//   <function_name>
//   <file_name>:<line_number>:<column_number>
//   ...
//   <empty line>
class LLVMSymbolizerProcess : public SymbolizerProcess {
 public:
  explicit LLVMSymbolizerProcess(const char *path) : SymbolizerProcess(path) {}

 private:
  bool ReachedEndOfOutput(const char *buffer, uptr length) const override {
    // Empty line marks the end of llvm-symbolizer output.
    return length >= 2 && buffer[length - 1] == '\n' &&
           buffer[length - 2] == '\n';
  }

  void GetArgV(const char *path_to_binary,
               const char *(&argv)[kArgVMax]) const override {
#if defined(__x86_64h__)
    const char* const kSymbolizerArch = "--default-arch=x86_64h";
#elif defined(__x86_64__)
    const char* const kSymbolizerArch = "--default-arch=x86_64";
#elif defined(__i386__)
    const char* const kSymbolizerArch = "--default-arch=i386";
#elif defined(__aarch64__)
    const char* const kSymbolizerArch = "--default-arch=arm64";
#elif defined(__arm__)
    const char* const kSymbolizerArch = "--default-arch=arm";
#elif defined(__powerpc64__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    const char* const kSymbolizerArch = "--default-arch=powerpc64";
#elif defined(__powerpc64__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    const char* const kSymbolizerArch = "--default-arch=powerpc64le";
#elif defined(__s390x__)
    const char* const kSymbolizerArch = "--default-arch=s390x";
#elif defined(__s390__)
    const char* const kSymbolizerArch = "--default-arch=s390";
#else
    const char* const kSymbolizerArch = "--default-arch=unknown";
#endif

    const char *const inline_flag = common_flags()->symbolize_inline_frames
                                        ? "--inlining=true"
                                        : "--inlining=false";
    int i = 0;
    argv[i++] = path_to_binary;
    argv[i++] = inline_flag;
    argv[i++] = kSymbolizerArch;
    argv[i++] = nullptr;
  }
};

LLVMSymbolizer::LLVMSymbolizer(const char *path, LowLevelAllocator *allocator)
    : symbolizer_process_(new(*allocator) LLVMSymbolizerProcess(path)) {}

// Parse a <file>:<line>[:<column>] buffer. The file path may contain colons on
// Windows, so extract tokens from the right hand side first. The column info is
// also optional.
static const char *ParseFileLineInfo(AddressInfo *info, const char *str) {
  char *file_line_info = 0;
  str = ExtractToken(str, "\n", &file_line_info);
  CHECK(file_line_info);
  // Parse the last :<int>, which must be there.
  char *last_colon = internal_strrchr(file_line_info, ':');
  CHECK(last_colon);
  int line_or_column = internal_atoll(last_colon + 1);
  // Truncate the string at the last colon and find the next-to-last colon.
  *last_colon = '\0';
  last_colon = internal_strrchr(file_line_info, ':');
  if (last_colon && IsDigit(last_colon[1])) {
    // If the second-to-last colon is followed by a digit, it must be the line
    // number, and the previous parsed number was a column.
    info->line = internal_atoll(last_colon + 1);
    info->column = line_or_column;
    *last_colon = '\0';
  } else {
    // Otherwise, we have line info but no column info.
    info->line = line_or_column;
    info->column = 0;
  }
  ExtractToken(file_line_info, "", &info->file);
  InternalFree(file_line_info);
  return str;
}

// Parses one or more two-line strings in the following format:
//   <function_name>
//   <file_name>:<line_number>[:<column_number>]
// Used by LLVMSymbolizer, Addr2LinePool and InternalSymbolizer, since all of
// them use the same output format.
void ParseSymbolizePCOutput(const char *str, SymbolizedStack *res) {
  bool top_frame = true;
  SymbolizedStack *last = res;
  while (true) {
    char *function_name = 0;
    str = ExtractToken(str, "\n", &function_name);
    CHECK(function_name);
    if (function_name[0] == '\0') {
      // There are no more frames.
      InternalFree(function_name);
      break;
    }
    SymbolizedStack *cur;
    if (top_frame) {
      cur = res;
      top_frame = false;
    } else {
      cur = SymbolizedStack::New(res->info.address);
      cur->info.FillModuleInfo(res->info.module, res->info.module_offset);
      last->next = cur;
      last = cur;
    }

    AddressInfo *info = &cur->info;
    info->function = function_name;
    str = ParseFileLineInfo(info, str);

    // Functions and filenames can be "??", in which case we write 0
    // to address info to mark that names are unknown.
    if (0 == internal_strcmp(info->function, "??")) {
      InternalFree(info->function);
      info->function = 0;
    }
    if (0 == internal_strcmp(info->file, "??")) {
      InternalFree(info->file);
      info->file = 0;
    }
  }
}

// Parses a two-line string in the following format:
//   <symbol_name>
//   <start_address> <size>
// Used by LLVMSymbolizer and InternalSymbolizer.
void ParseSymbolizeDataOutput(const char *str, DataInfo *info) {
  str = ExtractToken(str, "\n", &info->name);
  str = ExtractUptr(str, " ", &info->start);
  str = ExtractUptr(str, "\n", &info->size);
}

bool LLVMSymbolizer::SymbolizePC(uptr addr, SymbolizedStack *stack) {
  if (const char *buf = SendCommand(/*is_data*/ false, stack->info.module,
                                    stack->info.module_offset)) {
    ParseSymbolizePCOutput(buf, stack);
    return true;
  }
  return false;
}

bool LLVMSymbolizer::SymbolizeData(uptr addr, DataInfo *info) {
  if (const char *buf =
          SendCommand(/*is_data*/ true, info->module, info->module_offset)) {
    ParseSymbolizeDataOutput(buf, info);
    info->start += (addr - info->module_offset); // Add the base address.
    return true;
  }
  return false;
}

const char *LLVMSymbolizer::SendCommand(bool is_data, const char *module_name,
                                        uptr module_offset) {
  CHECK(module_name);
  internal_snprintf(buffer_, kBufferSize, "%s\"%s\" 0x%zx\n",
                    is_data ? "DATA " : "", module_name, module_offset);
  return symbolizer_process_->SendCommand(buffer_);
}

SymbolizerProcess::SymbolizerProcess(const char *path, bool use_forkpty)
    : path_(path),
      input_fd_(kInvalidFd),
      output_fd_(kInvalidFd),
      times_restarted_(0),
      failed_to_start_(false),
      reported_invalid_path_(false),
      use_forkpty_(use_forkpty) {
  CHECK(path_);
  CHECK_NE(path_[0], '\0');
}

const char *SymbolizerProcess::SendCommand(const char *command) {
  for (; times_restarted_ < kMaxTimesRestarted; times_restarted_++) {
    // Start or restart symbolizer if we failed to send command to it.
    if (const char *res = SendCommandImpl(command))
      return res;
    Restart();
  }
  if (!failed_to_start_) {
    Report("WARNING: Failed to use and restart external symbolizer!\n");
    failed_to_start_ = true;
  }
  return 0;
}

const char *SymbolizerProcess::SendCommandImpl(const char *command) {
  if (input_fd_ == kInvalidFd || output_fd_ == kInvalidFd)
      return 0;
  if (!WriteToSymbolizer(command, internal_strlen(command)))
      return 0;
  if (!ReadFromSymbolizer(buffer_, kBufferSize))
      return 0;
  return buffer_;
}

bool SymbolizerProcess::Restart() {
  if (input_fd_ != kInvalidFd)
    CloseFile(input_fd_);
  if (output_fd_ != kInvalidFd)
    CloseFile(output_fd_);
  return StartSymbolizerSubprocess();
}

bool SymbolizerProcess::ReadFromSymbolizer(char *buffer, uptr max_length) {
  if (max_length == 0)
    return true;
  uptr read_len = 0;
  while (true) {
    uptr just_read = 0;
    bool success = ReadFromFile(input_fd_, buffer + read_len,
                                max_length - read_len - 1, &just_read);
    // We can't read 0 bytes, as we don't expect external symbolizer to close
    // its stdout.
    if (!success || just_read == 0) {
      Report("WARNING: Can't read from symbolizer at fd %d\n", input_fd_);
      return false;
    }
    read_len += just_read;
    if (ReachedEndOfOutput(buffer, read_len))
      break;
  }
  buffer[read_len] = '\0';
  return true;
}

bool SymbolizerProcess::WriteToSymbolizer(const char *buffer, uptr length) {
  if (length == 0)
    return true;
  uptr write_len = 0;
  bool success = WriteToFile(output_fd_, buffer, length, &write_len);
  if (!success || write_len != length) {
    Report("WARNING: Can't write to symbolizer at fd %d\n", output_fd_);
    return false;
  }
  return true;
}

}  // namespace __sanitizer
