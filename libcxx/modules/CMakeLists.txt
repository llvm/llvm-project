if (CMAKE_VERSION VERSION_LESS 3.26)
  message(WARNING "The libc++ modules won't be available because the CMake version is too old. Update to CMake 3.26 or later.")
  return()
endif()

# The headers of Table 24: C++ library headers [tab:headers.cpp]
# and the headers of Table 25: C++ headers for C library facilities [tab:headers.cpp.c]
set(LIBCXX_SOURCES_MODULE_STD
  std.cppm
  std/algorithm.cppm
  std/any.cppm
  std/array.cppm
  std/atomic.cppm
  std/barrier.cppm
  std/bit.cppm
  std/bitset.cppm
  std/cassert.cppm
  std/cctype.cppm
  std/cerrno.cppm
  std/cfenv.cppm
  std/cfloat.cppm
  std/charconv.cppm
  std/chrono.cppm
  std/cinttypes.cppm
  std/climits.cppm
  std/clocale.cppm
  std/cmath.cppm
  std/codecvt.cppm
  std/compare.cppm
  std/complex.cppm
  std/concepts.cppm
  std/condition_variable.cppm
  std/coroutine.cppm
  std/csetjmp.cppm
  std/csignal.cppm
  std/cstdarg.cppm
  std/cstddef.cppm
  std/cstdint.cppm
  std/cstdio.cppm
  std/cstdlib.cppm
  std/cstring.cppm
  std/ctime.cppm
  std/cuchar.cppm
  std/cwchar.cppm
  std/cwctype.cppm
  std/deque.cppm
  std/exception.cppm
  std/execution.cppm
  std/expected.cppm
  std/filesystem.cppm
  std/flat_map.cppm
  std/flat_set.cppm
  std/format.cppm
  std/forward_list.cppm
  std/fstream.cppm
  std/functional.cppm
  std/future.cppm
  std/generator.cppm
  std/hazard_pointer.cppm
  std/initializer_list.cppm
  std/iomanip.cppm
  std/ios.cppm
  std/iosfwd.cppm
  std/iostream.cppm
  std/iostream.cppm
  std/istream.cppm
  std/iterator.cppm
  std/latch.cppm
  std/limits.cppm
  std/list.cppm
  std/locale.cppm
  std/map.cppm
  std/mdspan.cppm
  std/memory.cppm
  std/memory_resource.cppm
  std/mutex.cppm
  std/new.cppm
  std/numbers.cppm
  std/numeric.cppm
  std/optional.cppm
  std/ostream.cppm
  std/print.cppm
  std/queue.cppm
  std/random.cppm
  std/ranges.cppm
  std/ratio.cppm
  std/rcu.cppm
  std/regex.cppm
  std/scoped_allocator.cppm
  std/semaphore.cppm
  std/set.cppm
  std/shared_mutex.cppm
  std/source_location.cppm
  std/span.cppm
  std/spanstream.cppm
  std/sstream.cppm
  std/stack.cppm
  std/stacktrace.cppm
  std/stdexcept.cppm
  std/stdexcept.cppm
  std/stdfloat.cppm
  std/stop_token.cppm
  std/streambuf.cppm
  std/string.cppm
  std/string_view.cppm
  std/strstream.cppm
  std/syncstream.cppm
  std/system_error.cppm
  std/text_encoding.cppm
  std/thread.cppm
  std/tuple.cppm
  std/type_traits.cppm
  std/typeindex.cppm
  std/typeinfo.cppm
  std/unordered_map.cppm
  std/unordered_set.cppm
  std/utility.cppm
  std/valarray.cppm
  std/variant.cppm
  std/vector.cppm
  std/version.cppm
)

# TODO MODULES the CMakeLists.txt in the install directory is only temporary
# When that is removed the configured file can use the substitution
# LIBCXX_GENERATED_INCLUDE_TARGET_DIR avoiding this set.
# Also clean up the parts needed to generate the install version.
set(LIBCXX_CONFIGURED_INCLUDE_DIR ${LIBCXX_GENERATED_INCLUDE_TARGET_DIR})
configure_file(
  "CMakeLists.txt.in"
  "${LIBCXX_GENERATED_MODULE_DIR}/CMakeLists.txt"
  @ONLY
)

set(_all_modules "${LIBCXX_GENERATED_MODULE_DIR}/CMakeLists.txt")
foreach(file ${LIBCXX_SOURCES_MODULE_STD})
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
  set(dst "${LIBCXX_GENERATED_MODULE_DIR}/${file}")
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying CXX module ${file}")
  list(APPEND _all_modules "${dst}")
endforeach()

add_custom_target(generate-cxx-modules
  ALL DEPENDS
    ${_all_modules}
)
