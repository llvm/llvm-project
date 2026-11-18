# PSTL

## Overview

Libc++ provides an implementation of C++17 parallel algorithms
([P0024R2](https://wg21.link/p0024r2)) under `-fexperimental-library`. No other
build flags are required to enable parallelism; however, not every standard
algorithm has a parallel implementation at the moment.

The underlying parallel executor varies between platforms, as libc++ tries to
integrate well with the OS and other components of the program.
Currently these executors are employed:
- `libdispatch` on Darwin platforms ([doc](https://developer.apple.com/documentation/dispatch))
- Thread Pool API on Windows platforms ([doc](https://learn.microsoft.com/en-us/windows/win32/procthread/thread-pool-api))
- A custom thread pool on platforms that support `std::thread`

The implementation of PSTL in libc++ aims to perform correctly in all usage scenarios, e.g.:
- Calling parallel algorithms concurrently.
- Nested (recursive) calls to parallel algorithms (however parallelism can be limited in this case).
- Invoking PSTL before or after the `main` function.
- Targeting a platform with no support for threads.
- Running on a system that has only a single CPU core.

Generally, random-access iterators are required to enable parallel
implementations of the algorithms. The exception is `std::for_each` and the
family of algorithms based on it (e.g. `replace`, `generate` etc). They
will run in parallel even with forward iterators.

## Implementation details

The implementation of PSTL is split into several backends that are stacked on
top of each other. When an algorithm is called with an execution policy,
dispatching logic picks the first backend that implements the algorithm in the
current configuration.

There are currently 3 different backends:
- `default` ("one-liners") backend (`__pstl/backends/default.h`).  
  It only reframes the calls in terms of other more general functions, e.g.
  `generate(execution::par, ...)` is reframed in terms of `for_each(execution::par, ...)`.
- `serial` backend (`__pstl/backends/serial.h`).  
  This backend only delegates the calls to the serial implementations of the algorithms.  
  For example `set_symmetric_difference(execution::par, ...)` will be forwarded to `set_symmetric_difference(...)`.
- `statpart` backend (`__pstl/backends/statpart.h`).  
  This backend provides real parallel implementations of the algorithms and uses
  an underlying parallel executor to run them. While implementations in this
  backend vary a lot depending on the algorithm nature, they share a common
  approach of statically partitioning the input range upfront.

There are 2 configurations of composing the PSTL backends (`__pstl/backend_fwd.h`):
- When the platform has support for threading:  
  `statpart` -> `default` -> `serial`.
- When there's no threading:  
  `serial` -> `default`.

## Algorithms

The following table shows which algorithms are implemented in the backends:

| Algorithm                              | default | serial | statpart |
| -------------------------------------- | ------- | ------ | -------- |
| std::adjacent_difference | ✅ |  |  |
| std::adjacent_find | ✅ |  |  |
| std::all_of | ✅ |  |  |
| std::any_of | ✅ |  | ✅ |
| std::copy | ✅ |  |  |
| std::copy_if |  | ✅ |  |
| std::copy_n | ✅ |  |  |
| std::count | ✅ |  |  |
| std::count_if | ✅ |  |  |
| std::destroy | ✅ |  |  |
| std::destroy_n | ✅ |  |  |
| std::equal | ✅ |  |  |
| std::exclusive_scan |  |  |  |
| std::fill | ✅ |  | ✅ |
| std::fill_n | ✅ |  |  |
| std::find | ✅ |  |  |
| std::find_end |  | ✅ | ✅ |
| std::find_first_of | ✅ |  |  |
| std::find_if |  | ✅ | ✅ |
| std::find_if_not | ✅ |  |  |
| std::for_each |  | ✅ | ✅ |
| std::for_each_n | ✅ |  |  |
| std::generate | ✅ |  |  |
| std::generate_n | ✅ |  |  |
| std::includes |  | ✅ |  |
| std::inclusive_scan | ✅ |  |  |
| std::inplace_merge |  |  |  |
| std::is_heap | ✅ |  |  |
| std::is_heap_until |  | ✅ | ✅ |
| std::is_partitioned | ✅ |  |  |
| std::is_sorted | ✅ |  |  |
| std::is_sorted_until | ✅ |  |  |
| std::lexicographical_compare | ✅ |  |  |
| std::max_element | ✅ |  |  |
| std::merge |  | ✅ | ✅ |
| std::min_element |  | ✅ | ✅ |
| std::minmax_element |  | ✅ | ✅ |
| std::mismatch | ✅ | ✅ | ✅ |
| std::move | ✅ |  |  |
| std::none_of | ✅ |  |  |
| std::nth_element |  |  |  |
| std::partial_sort |  |  |  |
| std::partial_sort_copy |  |  |  |
| std::partition |  | ✅ |  |
| std::partition_copy |  | ✅ |  |
| std::reduce | ✅ |  |  |
| std::remove |  |  |  |
| std::remove_copy |  |  |  |
| std::remove_copy_if |  |  |  |
| std::remove_if |  |  |  |
| std::replace | ✅ |  |  |
| std::replace_copy | ✅ |  |  |
| std::replace_copy_if | ✅ |  |  |
| std::replace_if | ✅ |  |  |
| std::reverse |  | ✅ | ✅ |
| std::reverse_copy | ✅ |  |  |
| std::rotate |  |  |  |
| std::rotate_copy | ✅ |  |  |
| std::search |  | ✅ | ✅ |
| std::search_n |  | ✅ | ✅ |
| std::set_difference |  | ✅ |  |
| std::set_intersection |  | ✅ |  |
| std::set_symmetric_difference |  | ✅ |  |
| std::set_union |  | ✅ |  |
| std::sort | ✅ |  |  |
| std::stable_partition |  | ✅ |  |
| std::stable_sort |  | ✅ | ✅ |
| std::swap_ranges |  | ✅ | ✅ |
| std::transform |  | ✅ | ✅ |
| std::transform_exclusive_scan |  |  |  |
| std::transform_inclusive_scan | ✅ | ✅ | ✅ |
| std::transform_reduce |  | ✅ | ✅ |
| std::uninitialized_copy |  | ✅ | ✅ |
| std::uninitialized_copy_n | ✅ |  |  |
| std::uninitialized_default_construct | ✅ |  |  |
| std::uninitialized_default_construct_n | ✅ |  |  |
| std::uninitialized_fill | ✅ |  |  |
| std::uninitialized_fill_n | ✅ |  |  |
| std::uninitialized_move |  | ✅ | ✅ |
| std::uninitialized_move_n | ✅ |  |  |
| std::uninitialized_value_construct | ✅ |  |  |
| std::uninitialized_value_construct_n | ✅ |  |  |
| std::unique |  |  |  |
| std::unique_copy |  |  |  |
