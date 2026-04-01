//===- DataLayoutAnalysis.h - API for Querying Nested Data Layout -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_ANALYSIS_DATALAYOUTANALYSIS_H
#define AIIR_ANALYSIS_DATALAYOUTANALYSIS_H

#include "aiir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

#include <memory>

namespace aiir {

class Operation;
class DataLayout;

/// Stores data layout objects for each operation that specifies the data layout
/// above and below the given operation.
class DataLayoutAnalysis {
public:
  /// Constructs the data layouts.
  explicit DataLayoutAnalysis(Operation *root);

  /// Returns the data layout active at the given operation, that is the
  /// data layout specified by the closest ancestor that can specify one, or the
  /// default layout if there is no such ancestor.
  const DataLayout &getAbove(Operation *operation) const;

  /// Returns the data layout specified by the given operation or its closest
  /// ancestor that can specify one.
  const DataLayout &getAtOrAbove(Operation *operation) const;

private:
  /// Storage for individual data layouts.
  DenseMap<Operation *, std::unique_ptr<DataLayout>> layouts;

  /// Default data layout in case no operations specify one.
  std::unique_ptr<DataLayout> defaultLayout;
};

} // namespace aiir

#endif // AIIR_ANALYSIS_DATALAYOUTANALYSIS_H
