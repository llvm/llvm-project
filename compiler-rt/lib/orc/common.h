//===- common.h - Common utilities for the ORC runtime ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_COMMON_H
#define ORC_RT_COMMON_H

#include "compiler.h"
#include "error.h"
#include "executor_address.h"
#include "orc_rt/c_api.h"
#include <algorithm>
#include <type_traits>
#include <vector>

/// This macro should be used to define tags that will be associated with
/// handlers in the JIT process, and call can be used to define tags f
#define ORC_RT_JIT_DISPATCH_TAG(X) \
extern "C" char X; \
char X = 0;

/// Opaque struct for external symbols.
struct __orc_rt_Opaque {};

/// Error reporting function.
extern "C" void __orc_rt_log_error(const char *ErrMsg);

/// Context object for dispatching calls to the JIT object.
///
/// This is declared for use by the runtime, but should be implemented in the
/// executor or provided by a definition added to the JIT before the runtime
/// is loaded.
ORC_RT_IMPORT __orc_rt_Opaque __orc_rt_jit_dispatch_ctx ORC_RT_WEAK_IMPORT;

/// For dispatching calls to the JIT object.
///
/// This is declared for use by the runtime, but should be implemented in the
/// executor or provided by a definition added to the JIT before the runtime
/// is loaded.
ORC_RT_IMPORT orc_rt_CWrapperFunctionResult
__orc_rt_jit_dispatch(__orc_rt_Opaque *DispatchCtx, const void *FnTag,
                      const char *Data, size_t Size) ORC_RT_WEAK_IMPORT;

/// Used to manage sections of fixed-sized metadata records (e.g. pointer
/// sections, selector refs, etc.)
template <typename RecordElement> class RecordSectionsTracker {
public:
  /// Add a section to the "new" list.
  void add(orc_rt::span<RecordElement> Sec) { New.push_back(std::move(Sec)); }

  /// Returns true if there are new sections to process.
  bool hasNewSections() const { return !New.empty(); }

  /// Returns the number of new sections to process.
  size_t numNewSections() const { return New.size(); }

  /// Process all new sections.
  template <typename ProcessSectionFunc>
  std::enable_if_t<std::is_void_v<
      std::invoke_result_t<ProcessSectionFunc, orc_rt::span<RecordElement>>>>
  processNewSections(ProcessSectionFunc &&ProcessSection) {
    for (auto &Sec : New)
      ProcessSection(Sec);
    moveNewToProcessed();
  }

  /// Proces all new sections with a fallible handler.
  ///
  /// Successfully handled sections will be moved to the Processed
  /// list.
  template <typename ProcessSectionFunc>
  std::enable_if_t<
      std::is_same_v<orc_rt::Error,
                     std::invoke_result_t<ProcessSectionFunc,
                                          orc_rt::span<RecordElement>>>,
      orc_rt::Error>
  processNewSections(ProcessSectionFunc &&ProcessSection) {
    for (size_t I = 0; I != New.size(); ++I) {
      if (auto Err = ProcessSection(New[I])) {
        for (size_t J = 0; J != I; ++J)
          Processed.push_back(New[J]);
        New.erase(New.begin(), New.begin() + I);
        return Err;
      }
    }
    moveNewToProcessed();
    return orc_rt::Error::success();
  }

  /// Move all sections back to New for reprocessing.
  void reset() {
    moveNewToProcessed();
    New = std::move(Processed);
  }

  /// Remove the section with the given range.
  bool removeIfPresent(orc_rt::ExecutorAddrRange R) {
    if (removeIfPresent(New, R))
      return true;
    return removeIfPresent(Processed, R);
  }

private:
  void moveNewToProcessed() {
    if (Processed.empty())
      Processed = std::move(New);
    else {
      Processed.reserve(Processed.size() + New.size());
      std::copy(New.begin(), New.end(), std::back_inserter(Processed));
      New.clear();
    }
  }

  bool removeIfPresent(std::vector<orc_rt::span<RecordElement>> &V,
                       orc_rt::ExecutorAddrRange R) {
    auto RI = std::find_if(
        V.rbegin(), V.rend(),
        [RS = R.toSpan<RecordElement>()](const orc_rt::span<RecordElement> &E) {
          return E.data() == RS.data();
        });
    if (RI != V.rend()) {
      V.erase(std::next(RI).base());
      return true;
    }
    return false;
  }

  std::vector<orc_rt::span<RecordElement>> Processed;
  std::vector<orc_rt::span<RecordElement>> New;
};
#endif // ORC_RT_COMMON_H
