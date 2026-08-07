// Work-item and synchronization builtins as seen in clang spir64 LLVM IR
// (Itanium-mangled OpenCL C builtins). The single source of truth; passes
// and analyses reference these, never raw strings.

#ifndef INTER_SUPPORT_BUILTINS_H
#define INTER_SUPPORT_BUILTINS_H

#include "llvm/ADT/StringRef.h"

namespace inter::builtins {

inline constexpr llvm::StringRef kGetGlobalId = "_Z13get_global_idj";

} // namespace inter::builtins

#endif // INTER_SUPPORT_BUILTINS_H
