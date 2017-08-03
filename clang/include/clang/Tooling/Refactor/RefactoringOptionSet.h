//===--- RefactoringOptionSet.h - A container for the refactoring options -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPTION_SET_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPTION_SET_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace yaml {
class IO;
} // end namespace yaml
} // end namespace llvm

namespace clang {
namespace tooling {

struct RefactoringOption {
  virtual ~RefactoringOption() = default;

  struct SerializationContext {
    llvm::yaml::IO &IO;

    SerializationContext(llvm::yaml::IO &IO) : IO(IO) {}
  };

  virtual void serialize(const SerializationContext &Context);
};

/// \brief A set of refactoring options that can be given to a refactoring
/// operation.
class RefactoringOptionSet final {
  llvm::StringMap<std::unique_ptr<RefactoringOption>> Options;

public:
  RefactoringOptionSet() {}
  template <typename T> RefactoringOptionSet(const T &Option) { add(Option); }

  RefactoringOptionSet(RefactoringOptionSet &&) = default;
  RefactoringOptionSet &operator=(RefactoringOptionSet &&) = default;

  RefactoringOptionSet(const RefactoringOptionSet &) = delete;
  RefactoringOptionSet &operator=(const RefactoringOptionSet &) = delete;

  template <typename T> void add(const T &Option) {
    auto It = Options.try_emplace(StringRef(T::Name), nullptr);
    if (It.second)
      It.first->getValue().reset(new T(Option));
  }

  template <typename T> const T *get() const {
    auto It = Options.find(StringRef(T::Name));
    if (It == Options.end())
      return nullptr;
    return static_cast<const T *>(It->getValue().get());
  }

  template <typename T> const T &get(const T &Default) const {
    const auto *Ptr = get<T>();
    return Ptr ? *Ptr : Default;
  }

  void print(llvm::raw_ostream &OS) const;

  static llvm::Expected<RefactoringOptionSet> parse(StringRef Source);
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPTION_SET_H
