// Work-item and synchronization builtins as seen in clang spir64 LLVM IR
// (Itanium-mangled OpenCL C builtins). The single source of truth; passes
// and analyses reference these, never raw strings.

#ifndef INTER_SUPPORT_BUILTINS_H
#define INTER_SUPPORT_BUILTINS_H

#include "llvm/ADT/StringRef.h"

namespace inter::builtins {

inline constexpr llvm::StringRef kGetGlobalId = "_Z13get_global_idj";
inline constexpr llvm::StringRef kGetLocalId = "_Z12get_local_idj";
inline constexpr llvm::StringRef kBarrier = "_Z7barrierj";
// OpenCL C 1.2 atomic builtins (volatile pointer forms as emitted by clang).
inline constexpr llvm::StringRef kAtomicAdd = "_Z10atomic_addPU3AS1Vjj";

} // namespace inter::builtins

#endif // INTER_SUPPORT_BUILTINS_H
