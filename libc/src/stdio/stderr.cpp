#include "src/__support/File/file.h"

#include <stdio.h>

extern FILE *stderr = reinterpret_cast<FILE *>(__llvm_libc::stderr);
