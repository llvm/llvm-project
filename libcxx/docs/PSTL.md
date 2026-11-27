# PSTL

_THIS IS A WORK IN PROGRESS_

## Overview

Libc++ provides an implementation of C++17 parallel algorithms
([P0024R2](https://wg21.link/p0024r2)) under `-fexperimental-library`. No other
build flags are required to enable parallelism; however, not every standard
algorithm has a parallel implementation at the moment.

When the `std::execution::par` policy is passed, libc++ will make its best
attempt to run the algorithm in parallel and, as a fallback in case parallelism
is not available on the system, libc++ will run it serially.

The underlying parallelism provider varies between platforms, as libc++ tries to
integrate well with the OS and other components of the program.
Currently these providers are used:
- `libdispatch` on Apple platforms ([doc](https://developer.apple.com/documentation/dispatch))
- `Thread Pool API` on Windows ([doc](https://learn.microsoft.com/en-us/windows/win32/procthread/thread-pool-api))
- A custom thread pool on platforms that support `std::thread`

The implementation of PSTL in libc++ aims to perform correctly in many usage scenarios, e.g.:
- Calling parallel algorithms concurrently.
- Nested (recursive) calls to parallel algorithms (however parallelism can be limited in this case).
- Invoking PSTL before the `main` function starts or after it returns.
- Targeting a platform with no support for threads.
- Running on a system that has only a single CPU core.

Generally, random-access iterators are required to enable parallel implementations
of the algorithms, however some algorithms like `std::for_each` and the ones
derived from it do run in parallel even with forward iterators.

## Algorithms

The following table shows which algorithms can run in parallel if the
`std::execution::par` execution policy is provided.
- ✅ - runs in parallel (given the iterator requirements described above).
- ❌ - accepts an execution policy, but runs serially.

| Algorithm | Parallel |
| :-------- | :------: |
| std::adjacent_difference | ✅ |
| std::adjacent_find | ✅ |
| std::all_of | ✅ |
| std::any_of | ✅ |
| std::copy | ✅ |
| std::copy_if | ✅ |
| std::copy_n | ✅ |
| std::count | ✅ |
| std::count_if | ✅ |
| std::destroy | ✅ |
| std::destroy_n | ✅ |
| std::equal | ✅ |
| std::exclusive_scan | ✅ |
| std::fill | ✅ |
| std::fill_n | ✅ |
| std::find | ✅ |
| std::find_end | ✅ |
| std::find_first_of | ✅ |
| std::find_if | ✅ |
| std::find_if_not | ✅ |
| std::for_each | ✅ |
| std::for_each_n | ✅ |
| std::generate | ✅ |
| std::generate_n | ✅ |
| std::includes | ❌ |
| std::inclusive_scan | ✅ |
| std::inplace_merge | ❌ |
| std::is_heap | ✅ |
| std::is_heap_until | ✅ |
| std::is_partitioned | ✅ |
| std::is_sorted | ✅ |
| std::is_sorted_until | ✅ |
| std::lexicographical_compare | ✅ |
| std::max_element | ✅ |
| std::merge | ✅ |
| std::min_element | ✅ |
| std::minmax_element | ✅ |
| std::mismatch | ✅ |
| std::move | ✅ |
| std::none_of | ✅ |
| std::nth_element |  |
| std::partial_sort |  |
| std::partial_sort_copy |  |
| std::partition | ❌ |
| std::partition_copy | Planned |
| std::reduce | ✅ |
| std::remove | Planned |
| std::remove_copy | ✅ |
| std::remove_copy_if | ✅ |
| std::remove_if | Planned |
| std::replace | ✅ |
| std::replace_copy | ✅ |
| std::replace_copy_if | ✅ |
| std::replace_if | ✅ |
| std::reverse | ✅ |
| std::reverse_copy | ✅ |
| std::rotate | ❌ |
| std::rotate_copy | ✅ |
| std::search | ✅ |
| std::search_n | ✅ |
| std::set_difference | ❌ |
| std::set_intersection | ❌ |
| std::set_symmetric_difference | ❌ |
| std::set_union | ❌ |
| std::sort | ✅ |
| std::stable_partition | ❌ |
| std::stable_sort | ✅ |
| std::swap_ranges | ✅ |
| std::transform | ✅ |
| std::transform_exclusive_scan | ✅ |
| std::transform_inclusive_scan | ✅ |
| std::transform_reduce | ✅ |
| std::uninitialized_copy | ✅ |
| std::uninitialized_copy_n | ✅ |
| std::uninitialized_default_construct | ✅ |
| std::uninitialized_default_construct_n | ✅ |
| std::uninitialized_fill | ✅ |
| std::uninitialized_fill_n | ✅ |
| std::uninitialized_move | ✅ |
| std::uninitialized_move_n | ✅ |
| std::uninitialized_value_construct | ✅ |
| std::uninitialized_value_construct_n | ✅ |
| std::unique | Planned |
| std::unique_copy | Planned |
