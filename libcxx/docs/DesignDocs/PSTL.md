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

The following table shows which algorithms are implemented in the backends.
The values in the `default` backend show which algorithms are used to reframe as.
The `parallel` column shows whether the algorithm runs in parallel when
`std::execution::par` is passed on a platform with threading support:
- ✅ - runs in parallel (given the iterator requirements described above).
- ❌ - accepts an execution policy, but runs serially.

| Algorithm                              | default | serial | statpart | parallel? |
| -------------------------------------- | ------- | :----: | :------: | :------: |
| adjacent_difference | → transform |  |  | ✅ |
| adjacent_find | → mismatch |  |  | ✅ |
| all_of | → any_of |  |  | ✅ |
| any_of | → find_if |  | ● | ✅ |
| copy | → transform |  |  | ✅ |
| copy_if |  | ● |  | ❌ |
| copy_n | → copy |  |  | ✅ |
| count | → count_if |  |  | ✅ |
| count_if | → transform_reduce |  |  | ✅ |
| destroy | → for_each |  |  | ✅ |
| destroy_n | → for_each_n |  |  | ✅ |
| equal | → transform_reduce |  |  | ✅ |
| exclusive_scan |  |  |  |  |
| fill | → for_each |  | ● | ✅ |
| fill_n | → for_each_n |  |  | ✅ |
| find | → find_if |  |  | ✅ |
| find_end |  | ● | ● | ✅ |
| find_first_of | → find_if |  |  | ✅ |
| find_if |  | ● | ● | ✅ |
| find_if_not | → find_if |  |  | ✅ |
| for_each |  | ● | ● | ✅ |
| for_each_n | → for_each |  |  | ✅ |
| generate | → for_each |  |  | ✅ |
| generate_n | → for_each_n |  |  | ✅ |
| includes |  | ● |  | ❌ |
| inclusive_scan | → transform_inclusive_scan |  |  | ✅ |
| inplace_merge |  |  |  |  |
| is_heap | → is_heap_until |  |  | ✅ |
| is_heap_until |  | ● | ● | ✅ |
| is_partitioned | → find_if_not, none_of |  |  | ✅ |
| is_sorted | → is_sorted_until |  |  | ✅ |
| is_sorted_until | → adjacent_find |  |  | ✅ |
| lexicographical_compare | → mismatch |  |  | ✅ |
| max_element | → min_element |  |  | ✅ |
| merge |  | ● | ● | ✅ |
| min_element |  | ● | ● | ✅ |
| minmax_element |  | ● | ● | ✅ |
| mismatch | → mismatch | ● | ● | ✅ |
| move | → transform |  |  | ✅ |
| none_of | → any_of |  |  | ✅ |
| nth_element |  |  |  |  |
| partial_sort |  |  |  |  |
| partial_sort_copy |  |  |  |  |
| partition |  | ● |  | ❌ |
| partition_copy |  | ● |  | ❌ |
| reduce | → transform_reduce |  |  | ✅ |
| remove |  |  |  |  |
| remove_copy |  |  |  |  |
| remove_copy_if |  |  |  |  |
| remove_if |  |  |  |  |
| replace | → replace_if |  |  | ✅ |
| replace_copy | → replace_copy_if |  |  | ✅ |
| replace_copy_if | → transform |  |  | ✅ |
| replace_if | → for_each |  |  | ✅ |
| reverse |  | ● | ● | ✅ |
| reverse_copy | → copy |  |  | ✅ |
| rotate |  |  |  |  |
| rotate_copy | → copy |  |  | ✅ |
| search |  | ● | ● | ✅ |
| search_n |  | ● | ● | ✅ |
| set_difference |  | ● |  | ❌ |
| set_intersection |  | ● |  | ❌ |
| set_symmetric_difference |  | ● |  | ❌ |
| set_union |  | ● |  | ❌ |
| sort | → stable_sort |  |  | ✅ |
| stable_partition |  | ● |  | ❌ |
| stable_sort |  | ● | ● | ✅ |
| swap_ranges |  | ● | ● | ✅ |
| transform |  | ● | ● | ✅ |
| transform_exclusive_scan |  |  |  |  |
| transform_inclusive_scan | → transform_inclusive_scan | ● | ● | ✅ |
| transform_reduce |  | ● | ● | ✅ |
| uninitialized_copy |  | ● | ● | ✅ |
| uninitialized_copy_n | → uninitialized_copy |  |  | ✅ |
| uninitialized_default_construct | → for_each |  |  | ✅ |
| uninitialized_default_construct_n | → for_each_n |  |  | ✅ |
| uninitialized_fill | → for_each |  |  | ✅ |
| uninitialized_fill_n | → for_each_n |  |  | ✅ |
| uninitialized_move |  | ● | ● | ✅ |
| uninitialized_move_n | → uninitialized_move |  |  | ✅ |
| uninitialized_value_construct | → for_each |  |  | ✅ |
| uninitialized_value_construct_n | → for_each_n |  |  | ✅ |
| unique |  |  |  |  |
| unique_copy |  |  |  |  |
