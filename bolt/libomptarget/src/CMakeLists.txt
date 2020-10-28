##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build offloading library libomptarget.so.
#
##===----------------------------------------------------------------------===##

libomptarget_say("Building offloading runtime library libomptarget.")

set(src_files
  api.cpp
  device.cpp
  interface.cpp
  MemoryManager.cpp
  rtl.cpp
  omptarget.cpp
)

# Build libomptarget library with libdl dependency.
add_library(bolt-omptarget SHARED ${src_files})
target_link_libraries(bolt-omptarget
  ${CMAKE_DL_LIBS}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")

# Install libomptarget under the lib destination folder.
install(TARGETS bolt-omptarget LIBRARY COMPONENT omptarget
  DESTINATION "${OPENMP_INSTALL_LIBDIR}")

# Install aliases
get_target_property(BOLT_LIBOMPTARGET_LIBRARY_DIR bolt-omptarget LIBRARY_OUTPUT_DIRECTORY)
if(BOLT_LIBOMPTARGET_LIBRARY_DIR)
  add_custom_command(TARGET bolt-omptarget POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SHARED_LIBRARY_PREFIX}bolt-omptarget${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${CMAKE_SHARED_LIBRARY_PREFIX}omptarget${CMAKE_SHARED_LIBRARY_SUFFIX}
    WORKING_DIRECTORY ${BOLT_LIBOMPTARGET_LIBRARY_DIR}
  )
endif()
install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink \"${CMAKE_SHARED_LIBRARY_PREFIX}bolt-omptarget${CMAKE_SHARED_LIBRARY_SUFFIX}\"
  \"${CMAKE_SHARED_LIBRARY_PREFIX}omptarget${CMAKE_SHARED_LIBRARY_SUFFIX}\" WORKING_DIRECTORY
  \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENMP_INSTALL_LIBDIR})")
