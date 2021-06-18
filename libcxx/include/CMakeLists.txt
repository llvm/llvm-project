set(files
  __algorithm/adjacent_find.h
  __algorithm/all_of.h
  __algorithm/any_of.h
  __algorithm/binary_search.h
  __algorithm/clamp.h
  __algorithm/comp.h
  __algorithm/comp_ref_type.h
  __algorithm/copy.h
  __algorithm/count.h
  __algorithm/count_if.h
  __algorithm/equal.h
  __algorithm/equal_range.h
  __algorithm/fill.h
  __algorithm/find.h
  __algorithm/find_end.h
  __algorithm/find_first_of.h
  __algorithm/find_if.h
  __algorithm/find_if_not.h
  __algorithm/for_each.h
  __algorithm/for_each_n.h
  __algorithm/generate.h
  __algorithm/half_positive.h
  __algorithm/includes.h
  __algorithm/inplace_merge.h
  __algorithm/is_heap.h
  __algorithm/is_heap_until.h
  __algorithm/is_partitioned.h
  __algorithm/is_permutation.h
  __algorithm/is_sorted.h
  __algorithm/lexicographical_compare.h
  __algorithm/lower_bound.h
  __algorithm/make_heap.h
  __algorithm/max.h
  __algorithm/max_element.h
  __algorithm/merge.h
  __algorithm/min.h
  __algorithm/min_element.h
  __algorithm/minmax.h
  __algorithm/minmax_element.h
  __algorithm/mismatch.h
  __algorithm/move.h
  __algorithm/next_permutation.h
  __algorithm/none_of.h
  __algorithm/nth_element.h
  __algorithm/partial_sort.h
  __algorithm/partition.h
  __algorithm/partition_point.h
  __algorithm/pop_heap.h
  __algorithm/prev_permutation.h
  __algorithm/push_heap.h
  __algorithm/remove.h
  __algorithm/replace.h
  __algorithm/reverse.h
  __algorithm/rotate.h
  __algorithm/sample.h
  __algorithm/search.h
  __algorithm/search_n.h
  __algorithm/set_difference.h
  __algorithm/set_intersection.h
  __algorithm/set_symmetric_difference.h
  __algorithm/set_union.h
  __algorithm/shift.h
  __algorithm/shuffle.h
  __algorithm/sift_down.h
  __algorithm/sort.h
  __algorithm/sort_heap.h
  __algorithm/stable_partition.h
  __algorithm/stable_sort.h
  __algorithm/transform.h
  __algorithm/uniform_int_distribution.h
  __algorithm/unique.h
  __algorithm/unwrap_iter.h
  __algorithm/upper_bound.h
  __availability
  __bit_reference
  __bits
  __bsd_locale_defaults.h
  __bsd_locale_fallbacks.h
  __config
  __debug
  __errc
  __format/format_error.h
  __format/format_parse_context.h
  __function_like.h
  __functional/search.h
  __functional_03
  __functional_base
  __functional_base_03
  __hash_table
  __iterator/advance.h
  __iterator/concepts.h
  __iterator/default_sentinel.h
  __iterator/incrementable_traits.h
  __iterator/indirect_concepts.h
  __iterator/iter_move.h
  __iterator/iterator_traits.h
  __iterator/next.h
  __iterator/prev.h
  __iterator/projected.h
  __iterator/readable_traits.h
  __libcpp_version
  __locale
  __memory/addressof.h
  __memory/allocation_guard.h
  __memory/allocator.h
  __memory/allocator_traits.h
  __memory/auto_ptr.h
  __memory/compressed_pair.h
  __memory/construct_at.h
  __memory/pointer_safety.h
  __memory/pointer_traits.h
  __memory/raw_storage_iterator.h
  __memory/shared_ptr.h
  __memory/temporary_buffer.h
  __memory/uninitialized_algorithms.h
  __memory/unique_ptr.h
  __mutex_base
  __node_handle
  __nullptr
  __ranges/access.h
  __ranges/concepts.h
  __ranges/data.h
  __ranges/empty.h
  __ranges/empty_view.h
  __ranges/enable_borrowed_range.h
  __ranges/view_interface.h
  __ranges/view.h
  __ranges/size.h
  __split_buffer
  __std_stream
  __string
  __threading_support
  __tree
  __tuple
  __undef_macros
  __utility/to_underlying.h
  algorithm
  any
  array
  atomic
  barrier
  bit
  bitset
  cassert
  ccomplex
  cctype
  cerrno
  cfenv
  cfloat
  charconv
  chrono
  cinttypes
  ciso646
  climits
  clocale
  cmath
  codecvt
  compare
  complex
  complex.h
  concepts
  condition_variable
  csetjmp
  csignal
  cstdarg
  cstdbool
  cstddef
  cstdint
  cstdio
  cstdlib
  cstring
  ctgmath
  ctime
  ctype.h
  cwchar
  cwctype
  deque
  errno.h
  exception
  execution
  experimental/__config
  experimental/__memory
  experimental/algorithm
  experimental/coroutine
  experimental/deque
  experimental/filesystem
  experimental/forward_list
  experimental/functional
  experimental/iterator
  experimental/list
  experimental/map
  experimental/memory_resource
  experimental/propagate_const
  experimental/regex
  experimental/set
  experimental/simd
  experimental/string
  experimental/type_traits
  experimental/unordered_map
  experimental/unordered_set
  experimental/utility
  experimental/vector
  ext/__hash
  ext/hash_map
  ext/hash_set
  fenv.h
  filesystem
  float.h
  format
  forward_list
  fstream
  functional
  future
  initializer_list
  inttypes.h
  iomanip
  ios
  iosfwd
  iostream
  istream
  iterator
  latch
  limits
  limits.h
  list
  locale
  locale.h
  map
  math.h
  memory
  module.modulemap
  mutex
  new
  numbers
  numeric
  optional
  ostream
  queue
  random
  ranges
  ratio
  ranges
  regex
  scoped_allocator
  semaphore
  set
  setjmp.h
  shared_mutex
  span
  sstream
  stack
  stdbool.h
  stddef.h
  stdexcept
  stdint.h
  stdio.h
  stdlib.h
  streambuf
  string
  string.h
  string_view
  strstream
  __support/android/locale_bionic.h
  __support/fuchsia/xlocale.h
  __support/ibm/gettod_zos.h
  __support/ibm/limits.h
  __support/ibm/locale_mgmt_aix.h
  __support/ibm/locale_mgmt_zos.h
  __support/ibm/nanosleep.h
  __support/ibm/support.h
  __support/ibm/xlocale.h
  __support/musl/xlocale.h
  __support/newlib/xlocale.h
  __support/nuttx/xlocale.h
  __support/openbsd/xlocale.h
  __support/solaris/floatingpoint.h
  __support/solaris/wchar.h
  __support/solaris/xlocale.h
  __support/win32/limits_msvc_win32.h
  __support/win32/locale_win32.h
  __support/xlocale/__nop_locale_mgmt.h
  __support/xlocale/__posix_l_fallback.h
  __support/xlocale/__strtonum_fallback.h
  system_error
  tgmath.h
  thread
  tuple
  type_traits
  typeindex
  typeinfo
  unordered_map
  unordered_set
  utility
  valarray
  variant
  vector
  version
  wchar.h
  wctype.h
  )

configure_file("__config_site.in" "${LIBCXX_GENERATED_INCLUDE_TARGET_DIR}/__config_site" @ONLY)

set(_all_includes "${LIBCXX_GENERATED_INCLUDE_TARGET_DIR}/__config_site")
foreach(f ${files})
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
  set(dst "${LIBCXX_GENERATED_INCLUDE_DIR}/${f}")
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying CXX header ${f}")
  list(APPEND _all_includes "${dst}")
endforeach()

add_custom_target(generate-cxx-headers ALL DEPENDS ${_all_includes})

add_library(cxx-headers INTERFACE)
add_dependencies(cxx-headers generate-cxx-headers ${LIBCXX_CXX_ABI_HEADER_TARGET})
# TODO: Use target_include_directories once we figure out why that breaks the runtimes build
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
  target_compile_options(cxx-headers INTERFACE /I${LIBCXX_GENERATED_INCLUDE_DIR}
                                     INTERFACE /I${LIBCXX_GENERATED_INCLUDE_TARGET_DIR})
else()
  target_compile_options(cxx-headers INTERFACE -I${LIBCXX_GENERATED_INCLUDE_DIR}
                                     INTERFACE -I${LIBCXX_GENERATED_INCLUDE_TARGET_DIR})
endif()

if (LIBCXX_INSTALL_HEADERS)
  foreach(file ${files})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file}
      DESTINATION ${LIBCXX_INSTALL_INCLUDE_DIR}/${dir}
      COMPONENT cxx-headers
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    )
  endforeach()

  # Install the generated __config_site.
  install(FILES ${LIBCXX_GENERATED_INCLUDE_TARGET_DIR}/__config_site
    DESTINATION ${LIBCXX_INSTALL_INCLUDE_TARGET_DIR}
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    COMPONENT cxx-headers)

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_custom_target(install-cxx-headers
                      DEPENDS cxx-headers
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=cxx-headers
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    # Stripping is a no-op for headers
    add_custom_target(install-cxx-headers-stripped DEPENDS install-cxx-headers)
  endif()
endif()
