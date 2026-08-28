#ifndef LLVM_LIBC_SRC_STDIO_GETDELIM_H
#define LLVM_LIBC_SRC_STDIO_GETDELIM_H

#include "hdr/types/FILE.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

ssize_t getdelim(char **__restrict lineptr, size_t *__restrict n, int delimiter,
                 ::FILE *__restrict stream);
} // namespace LIBC_NAMESPACE_DECL
#endif
