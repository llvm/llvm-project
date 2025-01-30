#ifndef LLVM_LIBC_SRC___SUPPORT_UEFI_FILE_H
#define LLVM_LIBC_SRC___SUPPORT_UEFI_FILE_H

#include "include/llvm-libc-types/EFI_SIMPLE_TEXT_INPUT_PROTOCOL.h"
#include "include/llvm-libc-types/EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL.h"
#include "src/__support/CPP/new.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

enum FileHandleType {
  FileHandleId,
  FileHandleSimpleTextInput,
  FileHandleSimpleTextOutput,
};

union FileHandle {
  int id;
  EFI_SIMPLE_TEXT_INPUT_PROTOCOL *simple_text_input;
  EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *simple_text_output;
};

class File {
  FileHandle handle;
  FileHandleType handle_type;

private:
  bool needsReset();

public:
  constexpr File(FileHandle handle, FileHandleType handle_type)
      : handle(handle), handle_type(handle_type) {}

  void reset();

  size_t read(void *data, size_t len);
  size_t write(const void *data, size_t len);
};

extern File stdin;
extern File stdout;
extern File stderr;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_UEFI_FILE_H
