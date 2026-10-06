#ifndef LLVM_LIBC_BENCHMARKS_LIBC_FUNCTION_PROTOTYPES_H
#define LLVM_LIBC_BENCHMARKS_LIBC_FUNCTION_PROTOTYPES_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace libc_benchmarks {

/// Memory function prototype and configuration.
using MemcpyFunction = void *(*)(void *__restrict, const void *__restrict,
                                 size_t);
struct MemcpyConfiguration {
  MemcpyFunction function;
  llvm::StringRef name;
};

using MemmoveFunction = void *(*)(void *, const void *, size_t);
struct MemmoveConfiguration {
  MemmoveFunction function;
  llvm::StringRef name;
};

using MemsetFunction = void *(*)(void *, int, size_t);
struct MemsetConfiguration {
  MemsetFunction function;
  llvm::StringRef name;
};

using BzeroFunction = void (*)(void *, size_t);
struct BzeroConfiguration {
  BzeroFunction function;
  llvm::StringRef name;
};

using MemcmpOrBcmpFunction = int (*)(const void *, const void *, size_t);
struct MemcmpOrBcmpConfiguration {
  MemcmpOrBcmpFunction function;
  llvm::StringRef name;
};

} // namespace libc_benchmarks
} // namespace llvm

#endif /* LLVM_LIBC_BENCHMARKS_LIBC_FUNCTION_PROTOTYPES_H */
