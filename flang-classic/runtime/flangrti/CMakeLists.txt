
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

enable_language(C ASM) # Enable assembly

include(CheckTypeSize)

set(CMAKE_EXTRA_INCLUDE_FILES "sys/ucontext.h")
CHECK_TYPE_SIZE("gregset_t" GREGSET_T LANGUAGE C)
set(CMAKE_EXTRA_INCLUDE_FILES )

configure_file(flangrti_config.h.in flangrti_config.h)

SET(ASM_OPTIONS "-DLINUX_ELF")
SET(CMAKE_ASM_FLAGS "${CFLAGS} ${ASM_OPTIONS}" )

set(ARCH_DEP_FILES
  ${TARGET_ARCHITECTURE}/dumpregs.c
)

if( ${TARGET_ARCHITECTURE} STREQUAL "x86_64" )
  set(ARCH_DEP_FILES ${ARCH_DEP_FILES}
    x86_64/x86_daz.c
    x86_64/x86_flushz.c
  )
endif()

SET(PGC_SRC_FILES
  ${ARCH_DEP_FILES}
  bessel_tjn.c
  bessel_tyn.c
  dbessel_tjn.c
  dbessel_tyn.c
  f2cmain.c
  ktrap.c
  mcopy1.c
  mcopy2.c
  mcopy4.c
  mcopy8.c
  mset1.c
  mset2.c
  mset4.c
  mset8.c
  mzero1.c
  mzero2.c
  mzero4.c
  mzero8.c
  ioargs.c
  memalign.c
  iostdinit.c
  tempnam.c
  s_erf.c
  trace_lin.c
  trace.c
  mpmalloc.c
  # src-mp files
  llcrit.c
  init_nomp.c
  )

SET(PGC_SRC_FILES_QUADFP
  mcopy16.c
  mset16.c
  mzero16.c
)

list(APPEND LLVM_OPTIONAL_SOURCES ${PGC_SRC_FILES_QUADFP})

if(TARGET_SUPPORTS_QUADFP)
  LIST(APPEND PGC_SRC_FILES ${PGC_SRC_FILES_QUADFP})
endif()

add_flang_library(flangrti_static
  ${PGC_SRC_FILES}
  ${SHARED_SOURCES}
  )
if (MSVC)
  set_property(TARGET flangrti_static PROPERTY OUTPUT_NAME libflangrti)
else()
  set_property(TARGET flangrti_static PROPERTY OUTPUT_NAME flangrti)
endif()


set(SHARED_LIBRARY TRUE)
add_flang_library(flangrti_shared
  ${PGC_SRC_FILES}
  ${SHARED_SOURCES}
  )

# Resolve symbols against libm
if (NOT MSVC)
  target_link_libraries(flangrti_shared PRIVATE m)
endif()


# Resolve symbols against libpthread
find_package(Threads REQUIRED)
if (CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(flangrti_shared PRIVATE "${CMAKE_THREAD_LIBS_INIT}")
endif()

# Import OpenMP
if (NOT DEFINED LIBOMP_EXPORT_DIR)
  find_library(
    FLANG_LIBOMP
    NAMES omp libomp
    HINTS ${CMAKE_BINARY_DIR}/lib)
  target_link_libraries(flangrti_shared PUBLIC ${FLANG_LIBOMP})
endif()

find_library(
  LIBPGMATH
  NAMES pgmath libpgmath
  HINTS ${CMAKE_BINARY_DIR}/lib)
target_link_libraries(flangrti_shared PUBLIC ${LIBPGMATH})

if( ${TARGET_ARCHITECTURE} STREQUAL "aarch64" )
  target_compile_definitions(flangrti_static PRIVATE TARGET_LINUX_ARM)
  target_compile_definitions(flangrti_shared PRIVATE TARGET_LINUX_ARM)
elseif( ${TARGET_ARCHITECTURE} STREQUAL "ppc64le" )
  target_compile_definitions(flangrti_static PRIVATE TARGET_LINUX_POWER)
  target_compile_definitions(flangrti_shared PRIVATE TARGET_LINUX_POWER)
endif()

set_property(TARGET flangrti_shared PROPERTY OUTPUT_NAME flangrti)
set(SHARED_LIBRARY FALSE)

target_include_directories(flangrti_static
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  )

target_include_directories(flangrti_shared
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  )

set_target_properties(flangrti_shared flangrti_static
                      PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${FLANG_RTE_LIB_DIR})

if (NOT MSVC)
  target_compile_options(flangrti_static PRIVATE -fPIC)
  target_compile_options(flangrti_shared PRIVATE -fPIC)
else()
  set_target_properties(flangrti_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

target_compile_options(flangrti_static PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:-Mreentrant>)

target_compile_options(flangrti_shared PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:-Mreentrant>)

set_target_properties(flangrti_shared PROPERTIES INSTALL_RPATH "$ORIGIN")
