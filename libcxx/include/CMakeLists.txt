set(files
  __availability
  __bit_reference
  __bits
  __bsd_locale_defaults.h
  __bsd_locale_fallbacks.h
  __debug
  __errc
  __functional_03
  __functional_base
  __functional_base_03
  __hash_table
  __libcpp_version
  __locale
  __memory/allocator_traits.h
  __memory/base.h
  __memory/pointer_traits.h
  __memory/utilities.h
  __mutex_base
  __node_handle
  __nullptr
  __split_buffer
  __sso_allocator
  __std_stream
  __string
  __threading_support
  __tree
  __tuple
  __undef_macros
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
  ratio
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

configure_file("__config_site.in"
               "${LIBCXX_BINARY_DIR}/__config_site"
               @ONLY)

# Generate a custom __config header. The new header is created
# by prepending __config_site to the current __config header.
add_custom_command(OUTPUT ${LIBCXX_BINARY_DIR}/__generated_config
  COMMAND ${Python3_EXECUTABLE} ${LIBCXX_SOURCE_DIR}/utils/cat_files.py
    ${LIBCXX_BINARY_DIR}/__config_site
    ${LIBCXX_SOURCE_DIR}/include/__config
    -o ${LIBCXX_BINARY_DIR}/__generated_config
  DEPENDS ${LIBCXX_SOURCE_DIR}/include/__config
          ${LIBCXX_BINARY_DIR}/__config_site
)
# Add a target that executes the generation commands.
add_custom_target(cxx-generated-config ALL
  DEPENDS ${LIBCXX_BINARY_DIR}/__generated_config)

if(LIBCXX_HEADER_DIR)
  set(output_dir ${LIBCXX_HEADER_DIR}/include/c++/v1)

  set(out_files)
  foreach(f ${files})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    set(dst ${output_dir}/${f})
    add_custom_command(OUTPUT ${dst}
      DEPENDS ${src}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
      COMMENT "Copying CXX header ${f}")
    list(APPEND out_files ${dst})
  endforeach()

  # Copy the generated header as __config into build directory.
  set(src ${LIBCXX_BINARY_DIR}/__generated_config)
  set(dst ${output_dir}/__config)
  add_custom_command(OUTPUT ${dst}
      DEPENDS ${src} cxx-generated-config
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
      COMMENT "Copying CXX __config")
  list(APPEND out_files ${dst})
  add_custom_target(generate-cxx-headers ALL DEPENDS ${out_files})

  add_library(cxx-headers INTERFACE)
  add_dependencies(cxx-headers generate-cxx-headers ${LIBCXX_CXX_ABI_HEADER_TARGET})
  # TODO: Use target_include_directories once we figure out why that breaks the runtimes build
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
    target_compile_options(cxx-headers INTERFACE /I "${output_dir}")
  else()
    target_compile_options(cxx-headers INTERFACE -I "${output_dir}")
  endif()

  # Make sure the generated __config_site header is included when we build the library.
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
    target_compile_options(cxx-headers INTERFACE /FI "${LIBCXX_BINARY_DIR}/__config_site")
  else()
    target_compile_options(cxx-headers INTERFACE -include "${LIBCXX_BINARY_DIR}/__config_site")
  endif()
else()
  add_library(cxx-headers INTERFACE)
endif()

if (LIBCXX_INSTALL_HEADERS)
  foreach(file ${files})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file}
      DESTINATION ${LIBCXX_INSTALL_HEADER_PREFIX}include/c++/v1/${dir}
      COMPONENT cxx-headers
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    )
  endforeach()

  # Install the generated header as __config.
  install(FILES ${LIBCXX_BINARY_DIR}/__generated_config
    DESTINATION ${LIBCXX_INSTALL_HEADER_PREFIX}include/c++/v1
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    RENAME __config
    COMPONENT cxx-headers)

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_custom_target(install-cxx-headers
                      DEPENDS cxx-headers cxx-generated-config
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=cxx-headers
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    # Stripping is a no-op for headers
    add_custom_target(install-cxx-headers-stripped DEPENDS install-cxx-headers)
  endif()
endif()
