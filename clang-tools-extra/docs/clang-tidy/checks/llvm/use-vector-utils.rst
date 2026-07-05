.. title:: clang-tidy - llvm-use-vector-utils

llvm-use-vector-utils
=====================

Finds calls to ``llvm::to_vector`` with ``llvm::map_range`` or
``llvm::make_filter_range`` that can be replaced with the more concise
``llvm::map_to_vector`` and ``llvm::filter_to_vector`` utilities from
``llvm/ADT/SmallVectorExtras.h``.

The check will add the necessary ``#include "llvm/ADT/SmallVectorExtras.h"``
directive when applying fixes.

Example
-------

.. code-block:: c++

  auto v1 = llvm::to_vector(llvm::map_range(container, func));
  auto v2 = llvm::to_vector(llvm::make_filter_range(container, pred));
  auto v3 = llvm::to_vector<4>(llvm::map_range(container, func));
  auto v4 = llvm::to_vector<4>(llvm::make_filter_range(container, pred));

Transforms to:

.. code-block:: c++

  auto v1 = llvm::map_to_vector(container, func);
  auto v2 = llvm::filter_to_vector(container, pred);
  auto v3 = llvm::map_to_vector<4>(container, func);
  auto v4 = llvm::filter_to_vector<4>(container, pred);
