# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for details.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===----------------------------------------------------------------------===//
  
  

if(LIBBOLT_OMPT_SUPPORT)
  include_directories(${LIBBOLT_INCLUDE_DIR})

  add_library(bolt-archer SHARED ompt-tsan.cpp)
  add_library(bolt-archer_static STATIC ompt-tsan.cpp)

  install(TARGETS bolt-archer bolt-archer_static
    LIBRARY DESTINATION ${OPENMP_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${OPENMP_INSTALL_LIBDIR})

  # install aliases for BOLT
  add_custom_command(TARGET bolt-archer bolt-archer_static POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SHARED_LIBRARY_PREFIX}bolt-archer${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${CMAKE_SHARED_LIBRARY_PREFIX}archer${CMAKE_SHARED_LIBRARY_SUFFIX}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_STATIC_LIBRARY_PREFIX}bolt-archer_static${CMAKE_STATIC_LIBRARY_SUFFIX}
      ${CMAKE_STATIC_LIBRARY_PREFIX}archer_static${CMAKE_STATIC_LIBRARY_SUFFIX}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink \"${CMAKE_SHARED_LIBRARY_PREFIX}bolt-archer${CMAKE_SHARED_LIBRARY_SUFFIX}\"
    \"${CMAKE_SHARED_LIBRARY_PREFIX}archer${CMAKE_SHARED_LIBRARY_SUFFIX}\" WORKING_DIRECTORY
    \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENMP_INSTALL_LIBDIR})")
  install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink \"${CMAKE_STATIC_LIBRARY_PREFIX}bolt-archer_static${CMAKE_STATIC_LIBRARY_SUFFIX}\"
    \"${CMAKE_STATIC_LIBRARY_PREFIX}archer_static${LIBOMP_LIBRARY_SUFFIX}\" WORKING_DIRECTORY
    \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENMP_INSTALL_LIBDIR})")

  add_subdirectory(tests)
endif()
