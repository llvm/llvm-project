//== ProgramState_Fwd.h - Incomplete declarations of ProgramState -*- C++ -*--=/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_PROGRAMSTATE_FWD_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_PROGRAMSTATE_FWD_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace clang {
namespace ento {
  class ProgramState;
  class ProgramStateManager;
}
}

namespace llvm {
  template <> struct IntrusiveRefCntPtrInfo<const clang::ento::ProgramState> {
    // These are templates so that their bodies are only instantiated where
    // ProgramState is complete, which lets them be inlined. Destroying a
    // ProgramStateRef needs the complete type, including in a function that
    // takes one by value, as the callee destroys the arguments in some ABIs.
    template <typename T> static void retain(T *state) { state->retain(); }
    template <typename T> static void release(T *state) { state->release(); }
  };
}

namespace clang {
namespace ento {
  typedef IntrusiveRefCntPtr<const ProgramState> ProgramStateRef;
}
}

#endif

