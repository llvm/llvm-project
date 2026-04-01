//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helper functions used to wrap kernel arguments to
/// typeless collection.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_ARG_WRAPPER_HPP
#define _LIBSYCL___IMPL_DETAIL_ARG_WRAPPER_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/exception.hpp>

#include <cassert>
#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// Base class is needed for unification, we pass arguments through ABI
/// boundary.
class ArgWrapperBase {
public:
  ArgWrapperBase(const ArgWrapperBase &) = delete;
  ArgWrapperBase &operator=(const ArgWrapperBase &) = delete;
  virtual ~ArgWrapperBase() = default;

  virtual bool deepCopy() = 0;
  virtual size_t getSize() const = 0;
  virtual const void *getPtr() const = 0;

protected:
  ArgWrapperBase() = default;
};

/// Helps to manage arguments in a typeless way.
template <typename Type> class ArgWrapper : public ArgWrapperBase {
public:
  ArgWrapper(Type &Arg) { Ptr = &Arg; }
  ArgWrapper(const ArgWrapper &) = delete;
  ArgWrapper &operator=(const ArgWrapper &) = delete;

  /// \return the size of the argument in bytes.
  size_t getSize() const override { return sizeof(Type); }

  /// Returns a raw pointer to the corresponding argument.
  /// No copy is done by this method. It works with pointer to the memory whose
  /// existence must be guaranteed by class user or with copy that must be
  /// explicitly requested by class user via deepCopy method.
  /// \return a pointer to the argument.
  const void *getPtr() const override {
    assert((!DeepCopy || (DeepCopy.get()) == Ptr) &&
           "Incorrect state of copied argument");
    return Ptr;
  }

  /// Copies the agrument to RT owned storage.
  /// \return true if argument was copied in this exact call.
  bool deepCopy() override {
    if (DeepCopy)
      return false;

    DeepCopy.reset(new Type(*Ptr));
    Ptr = DeepCopy.get();
    return true;
  }

private:
  Type *Ptr;
  std::unique_ptr<Type> DeepCopy;
};

/// Collection of arguments. Provides functionality to accumulate all arguments
/// data to pass through ABI boundary.
class ArgCollection {
public:
  /// Adds an argument to the collection. Doesn't own the memory, the argument
  /// lifetime must be guaranteed by the class user. If extended lifetime is
  /// needed (copy), deepCopy must be called.
  template <typename Type> void addArg(Type &Arg) {
    MArgs.emplace_back(new ArgWrapper(Arg));
  }

  /// \return array of argument pointers.
  const void **getArgPtrArray() {
    if (MPtrs.size() != MArgs.size()) {
      MPtrs.clear();
      MPtrs.reserve(MArgs.size());
      for (const auto &Argument : MArgs)
        MPtrs.push_back(Argument->getPtr());
    }
    return MPtrs.data();
  }

  /// \return array of argument sizes.
  int64_t *getSizesArray() {
    if (MSizes.size() != MArgs.size()) {
      MSizes.clear();
      MSizes.reserve(MArgs.size());
      for (const auto &Argument : MArgs)
        MSizes.push_back(static_cast<int64_t>(Argument->getSize()));
    }
    return MSizes.data();
  }

  /// \return count of arguments in collection.
  size_t getArgCount() { return MArgs.size(); }

  /// Extends arguments lifetime by doing copy of all arguments.
  void deepCopy() {
    bool CopiedAtLeastOne = false;
    for (auto &Arg : MArgs)
      CopiedAtLeastOne |= Arg->deepCopy();

    if (CopiedAtLeastOne) {
      MPtrs.clear();
      // MSizes must be the same. No changes here so no need to clean and
      // refill.
    }
  }

private:
  std::vector<std::unique_ptr<ArgWrapperBase>> MArgs;
  std::vector<int64_t> MSizes;
  std::vector<const void *> MPtrs;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_ARG_WRAPPER_HPP
