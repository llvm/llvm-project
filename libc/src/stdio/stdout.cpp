#include "src/__support/File/file.h"

#include <stdio.h>

extern FILE *stdout = reinterpret_cast<FILE *>(__llvm_libc::stdout);
