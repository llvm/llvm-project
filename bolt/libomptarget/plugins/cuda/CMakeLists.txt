##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a CUDA machine if available.
#
##===----------------------------------------------------------------------===##
if (NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building CUDA offloading plugin: only support CUDA in Linux x86_64, ppc64le, or aarch64 hosts.")
  return()
elseif (NOT LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building CUDA offloading plugin: libelf dependency not found.")
  return()
elseif(NOT LIBOMPTARGET_DEP_CUDA_FOUND)
  libomptarget_say("Not building CUDA offloading plugin: CUDA not found in system.")
  return()
elseif(NOT LIBOMPTARGET_DEP_CUDA_DRIVER_FOUND)
  libomptarget_say("Not building CUDA offloading plugin: CUDA Driver API not found in system.")
  return()
endif()

libomptarget_say("Building CUDA offloading plugin.")

# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=CUDA)

include_directories(${LIBOMPTARGET_DEP_CUDA_INCLUDE_DIRS})
include_directories(${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS})

add_library(bolt-omptarget.rtl.cuda SHARED src/rtl.cpp)

# Install plugin under the lib destination folder.
install(TARGETS bolt-omptarget.rtl.cuda LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")

# Install aliases
get_target_property(BOLT_LIBOMPTARGET_LIBRARY_DIR bolt-omptarget.rtl.cuda LIBRARY_OUTPUT_DIRECTORY)
if(BOLT_LIBOMPTARGET_LIBRARY_DIR)
  add_custom_command(TARGET bolt-omptarget.rtl.cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SHARED_LIBRARY_PREFIX}bolt-omptarget.rtl.cuda${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${CMAKE_SHARED_LIBRARY_PREFIX}omptarget.rtl.cuda${CMAKE_SHARED_LIBRARY_SUFFIX}
      WORKING_DIRECTORY ${BOLT_LIBOMPTARGET_LIBRARY_DIR}
  )
endif()
install(CODE "execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink \"${CMAKE_SHARED_LIBRARY_PREFIX}bolt-omptarget.rtl.cuda${CMAKE_SHARED_LIBRARY_SUFFIX}\"
  \"${CMAKE_SHARED_LIBRARY_PREFIX}omptarget.rtl.cuda${CMAKE_SHARED_LIBRARY_SUFFIX}\" WORKING_DIRECTORY
  \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${OPENMP_INSTALL_LIBDIR})")

target_link_libraries(bolt-omptarget.rtl.cuda
  ${LIBOMPTARGET_DEP_CUDA_DRIVER_LIBRARIES}
  ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")

# Report to the parent scope that we are building a plugin for CUDA.
set(LIBBOLTTARGET_SYSTEM_TARGETS "${LIBBOLTTARGET_SYSTEM_TARGETS} nvptx64-nvidia-cuda" PARENT_SCOPE)
