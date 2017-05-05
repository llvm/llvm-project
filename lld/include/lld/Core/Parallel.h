//===- lld/Core/Parallel.h - Parallel utilities ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_PARALLEL_H
#define LLD_CORE_PARALLEL_H

#include "lld/Core/LLVM.h"
#include "lld/Core/TaskGroup.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Config/llvm-config.h"

#include <algorithm>

#if defined(_MSC_VER) && LLVM_ENABLE_THREADS
#include <concrt.h>
#include <ppl.h>
#endif

namespace lld {

#if !LLVM_ENABLE_THREADS
template <class RandomAccessIterator, class Comparator>
void parallel_sort(
    RandomAccessIterator Start, RandomAccessIterator End,
    const Comparator &Comp = std::less<
        typename std::iterator_traits<RandomAccessIterator>::value_type>()) {
  std::sort(Start, End, Comp);
}
#elif defined(_MSC_VER)
// Use ppl parallel_sort on Windows.
template <class RandomAccessIterator, class Comparator>
void parallel_sort(
    RandomAccessIterator Start, RandomAccessIterator End,
    const Comparator &Comp = std::less<
        typename std::iterator_traits<RandomAccessIterator>::value_type>()) {
  concurrency::parallel_sort(Start, End, Comp);
}
#else
namespace detail {
const ptrdiff_t MinParallelSize = 1024;

/// \brief Inclusive median.
template <class RandomAccessIterator, class Comparator>
RandomAccessIterator medianOf3(RandomAccessIterator Start,
                               RandomAccessIterator End,
                               const Comparator &Comp) {
  RandomAccessIterator Mid = Start + (std::distance(Start, End) / 2);
  return Comp(*Start, *(End - 1))
             ? (Comp(*Mid, *(End - 1)) ? (Comp(*Start, *Mid) ? Mid : Start)
                                       : End - 1)
             : (Comp(*Mid, *Start) ? (Comp(*(End - 1), *Mid) ? Mid : End - 1)
                                   : Start);
}

template <class RandomAccessIterator, class Comparator>
void parallel_quick_sort(RandomAccessIterator Start, RandomAccessIterator End,
                         const Comparator &Comp, TaskGroup &TG, size_t Depth) {
  // Do a sequential sort for small inputs.
  if (std::distance(Start, End) < detail::MinParallelSize || Depth == 0) {
    std::sort(Start, End, Comp);
    return;
  }

  // Partition.
  auto Pivot = medianOf3(Start, End, Comp);
  // Move Pivot to End.
  std::swap(*(End - 1), *Pivot);
  Pivot = std::partition(Start, End - 1, [&Comp, End](decltype(*Start) V) {
    return Comp(V, *(End - 1));
  });
  // Move Pivot to middle of partition.
  std::swap(*Pivot, *(End - 1));

  // Recurse.
  TG.spawn([=, &Comp, &TG] {
    parallel_quick_sort(Start, Pivot, Comp, TG, Depth - 1);
  });
  parallel_quick_sort(Pivot + 1, End, Comp, TG, Depth - 1);
}
}

template <class RandomAccessIterator, class Comparator>
void parallel_sort(
    RandomAccessIterator Start, RandomAccessIterator End,
    const Comparator &Comp = std::less<
        typename std::iterator_traits<RandomAccessIterator>::value_type>()) {
  TaskGroup TG;
  detail::parallel_quick_sort(Start, End, Comp, TG,
                              llvm::Log2_64(std::distance(Start, End)) + 1);
}
#endif

template <class T> void parallel_sort(T *Start, T *End) {
  parallel_sort(Start, End, std::less<T>());
}

#if !LLVM_ENABLE_THREADS
template <class IterTy, class FuncTy>
void parallel_for_each(IterTy Begin, IterTy End, FuncTy Fn) {
  std::for_each(Begin, End, Fn);
}

template <class IndexTy, class FuncTy>
void parallel_for(IndexTy Begin, IndexTy End, FuncTy Fn) {
  for (IndexTy I = Begin; I != End; ++I)
    Fn(I);
}
#elif defined(_MSC_VER)
// Use ppl parallel_for_each on Windows.
template <class IterTy, class FuncTy>
void parallel_for_each(IterTy Begin, IterTy End, FuncTy Fn) {
  concurrency::parallel_for_each(Begin, End, Fn);
}

template <class IndexTy, class FuncTy>
void parallel_for(IndexTy Begin, IndexTy End, FuncTy Fn) {
  concurrency::parallel_for(Begin, End, Fn);
}
#else
template <class IterTy, class FuncTy>
void parallel_for_each(IterTy Begin, IterTy End, FuncTy Fn) {
  // TaskGroup has a relatively high overhead, so we want to reduce
  // the number of spawn() calls. We'll create up to 1024 tasks here.
  // (Note that 1024 is an arbitrary number. This code probably needs
  // improving to take the number of available cores into account.)
  ptrdiff_t TaskSize = std::distance(Begin, End) / 1024;
  if (TaskSize == 0)
    TaskSize = 1;

  TaskGroup TG;
  while (TaskSize <= std::distance(Begin, End)) {
    TG.spawn([=, &Fn] { std::for_each(Begin, Begin + TaskSize, Fn); });
    Begin += TaskSize;
  }
  TG.spawn([=, &Fn] { std::for_each(Begin, End, Fn); });
}

template <class IndexTy, class FuncTy>
void parallel_for(IndexTy Begin, IndexTy End, FuncTy Fn) {
  ptrdiff_t TaskSize = (End - Begin) / 1024;
  if (TaskSize == 0)
    TaskSize = 1;

  TaskGroup TG;
  IndexTy I = Begin;
  for (; I + TaskSize < End; I += TaskSize) {
    TG.spawn([=, &Fn] {
      for (IndexTy J = I, E = I + TaskSize; J != E; ++J)
        Fn(J);
    });
  }
  TG.spawn([=, &Fn] {
    for (IndexTy J = I; J < End; ++J)
      Fn(J);
  });
}
#endif
} // End namespace lld

#endif // LLD_CORE_PARALLEL_H
