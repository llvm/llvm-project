#include "file.h"
#include "hdr/types/FILE.h"
#include "src/__support/macros/config.h"
#include <Uefi.h>

#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2

namespace LIBC_NAMESPACE_DECL {
bool File::needsReset() { return handle_type == FileHandleId; }

void File::reset() {
  if (handle_type != FileHandleId)
    return;

  if (handle.id == STDIN_FILENO) {
    handle = (FileHandle){
        .simple_text_input = efi_system_table->ConIn,
    };
    handle_type = FileHandleSimpleTextInput;
  } else {
    handle = (FileHandle){
        .simple_text_output = handle.id == STDERR_FILENO
                                  ? efi_system_table->StdErr
                                  : efi_system_table->ConOut,
    };
    handle_type = FileHandleSimpleTextOutput;
  }
}

size_t File::read(void *, size_t) {
  if (needsReset())
    reset();

  // TODO: decode keys from simple text input
  return 0;
}

size_t File::write(const void *data, size_t data_len) {
  if (needsReset())
    reset();

  if (handle_type == FileHandleSimpleTextOutput) {
    for (size_t i = 0; i < data_len; i++) {
      char16_t e[2] = {((const char *)data)[i], 0};
      handle.simple_text_output->OutputString(
          handle.simple_text_output, reinterpret_cast<const char16_t *>(&e));
    }
    return data_len;
  }
  return 0;
}

File stdin(
    (FileHandle){
        .id = STDIN_FILENO,
    },
    FileHandleId);

File stdout(
    (FileHandle){
        .id = STDOUT_FILENO,
    },
    FileHandleId);

File stderr(
    (FileHandle){
        .id = STDERR_FILENO,
    },
    FileHandleId);
} // namespace LIBC_NAMESPACE_DECL

extern "C" {
FILE *stdin = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::stdin);
FILE *stdout = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::stdout);
FILE *stderr = reinterpret_cast<FILE *>(&LIBC_NAMESPACE::stderr);
}
