//===--- MutexRegionExtractor.h - Modeling of mutexes ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines modeling checker for tracking mutex states.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_MUTEXMODELING_MUTEXREGIONEXTRACTOR_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_MUTEXMODELING_MUTEXREGIONEXTRACTOR_H

#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include <variant>

namespace clang::ento::mutex_modeling {

class FirstArgMutexExtractor {
  CallDescription CD;

public:
  template <typename T>
  FirstArgMutexExtractor(T &&CD) : CD(std::forward<T>(CD)) {}

  [[nodiscard]] bool matches(const CallEvent &Call) const {
    return CD.matches(Call);
  }

  [[nodiscard]] const MemRegion *getRegion(const CallEvent &Call) const {
    return Call.getArgSVal(0).getAsRegion();
  }
};

class MemberMutexExtractor {
  CallDescription CD;

public:
  template <typename T>
  MemberMutexExtractor(T &&CD) : CD(std::forward<T>(CD)) {}

  [[nodiscard]] bool matches(const CallEvent &Call) const {
    return CD.matches(Call);
  }

  [[nodiscard]] const MemRegion *getRegion(const CallEvent &Call) const {
    return llvm::cast<CXXMemberCall>(Call).getCXXThisVal().getAsRegion();
  }
};

template <bool IsLock> class RAIIMutexExtractor {
  mutable const clang::IdentifierInfo *Guard{};
  mutable bool IdentifierInfoInitialized{};
  mutable llvm::SmallString<32> GuardName{};

  void initIdentifierInfo(const CallEvent &Call) const {
    if (!IdentifierInfoInitialized) {
      // In case of checking C code, or when the corresponding headers are not
      // included, we might end up query the identifier table every time when
      // this function is called instead of early returning it. To avoid this,
      // a bool variable (IdentifierInfoInitialized) is used and the function
      // will be run only once.
      const auto &ASTCtx = Call.getState()->getStateManager().getContext();
      Guard = &ASTCtx.Idents.get(GuardName);
    }
  }

  template <typename T> bool matchesImpl(const CallEvent &Call) const {
    const T *C = llvm::dyn_cast<T>(&Call);
    if (!C)
      return false;
    const clang::IdentifierInfo *II =
        llvm::cast<clang::CXXRecordDecl>(C->getDecl()->getParent())
            ->getIdentifier();
    return II == Guard;
  }

public:
  RAIIMutexExtractor(llvm::StringRef GuardName) : GuardName(GuardName) {}
  [[nodiscard]] bool matches(const CallEvent &Call) const {
    initIdentifierInfo(Call);
    if constexpr (IsLock) {
      return matchesImpl<CXXConstructorCall>(Call);
    } else {
      return matchesImpl<CXXDestructorCall>(Call);
    }
  }
  [[nodiscard]] const MemRegion *getRegion(const CallEvent &Call) const {
    const MemRegion *MutexRegion = nullptr;
    if constexpr (IsLock) {
      if (std::optional<SVal> Object = Call.getReturnValueUnderConstruction()) {
        MutexRegion = Object->getAsRegion();
      }
    } else {
      MutexRegion =
          llvm::cast<CXXDestructorCall>(Call).getCXXThisVal().getAsRegion();
    }
    return MutexRegion;
  }
};

using RAIILockExtractor = RAIIMutexExtractor<true>;
using RAIIReleaseExtractor = RAIIMutexExtractor<false>;

using MutexRegionExtractor =
    std::variant<FirstArgMutexExtractor, MemberMutexExtractor,
                 RAIILockExtractor, RAIIReleaseExtractor>;

inline const MemRegion *getRegion(const MutexRegionExtractor &Extractor,
                                  const CallEvent &Call) {
  return std::visit(
      [&Call](auto &&Descriptor) { return Descriptor.getRegion(Call); },
      Extractor);
}

inline bool operator==(const MutexRegionExtractor &LHS,
                       const MutexRegionExtractor &RHS) {
  return std::visit([](auto &&LHS, auto &&RHS) { return LHS == RHS; }, LHS,
                    RHS);
}

inline bool matches(const MutexRegionExtractor &Extractor,
                    const CallEvent &Call) {
  return std::visit(
      [&Call](auto &&Extractor) { return Extractor.matches(Call); }, Extractor);
}

} // namespace clang::ento::mutex_modeling

#endif
