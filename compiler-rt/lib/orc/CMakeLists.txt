# Build for all components of the ORC runtime support library.

# ORC runtime library implementation files.
set(ORC_SOURCES
  extensible_rtti.cpp
  log_error_to_stderr.cpp
  macho_platform.cpp
  elfnix_platform.cpp
  run_program_wrapper.cpp
  )

# Implementation files for all ORC architectures.
set(ALL_ORC_ASM_SOURCES
  macho_tlv.x86-64.S
  elfnix_tls.x86-64.S
)

set(ORC_IMPL_HEADERS
# Implementation headers will go here.
  adt.h
  c_api.h
  common.h
  compiler.h
  endianness.h
  error.h
  executor_address.h
  extensible_rtti.h
  macho_platform.h
  simple_packed_serialization.h
  stl_extras.h
  wrapper_function_utils.h
)

# Create list of all source files for
# consumption by tests.
set(ORC_ALL_SOURCE_FILES
  ${ORC_SOURCES}
  ${ALL_ORC_ASM_SOURCES}
  ${ORC_IMPL_HEADERS}
  )

list(REMOVE_DUPLICATES ORC_ALL_SOURCE_FILES)

# Now put it all together...
include_directories(..)
include_directories(../../include)

set(ORC_CFLAGS ${COMPILER_RT_COMMON_CFLAGS})

# Allow the ORC runtime to reference LLVM headers.
foreach (DIR ${LLVM_INCLUDE_DIR} ${LLVM_MAIN_INCLUDE_DIR})
  list(APPEND ORC_CFLAGS -I${DIR})
endforeach()

add_compiler_rt_component(orc)

# ORC uses C++ standard library headers.
if (TARGET cxx-headers OR HAVE_LIBCXX)
  set(ORC_DEPS cxx-headers)
endif()

if (APPLE)
  add_asm_sources(ORC_ASM_SOURCES macho_tlv.x86-64.S)

  add_compiler_rt_object_libraries(RTOrc
    OS ${ORC_SUPPORTED_OS}
    ARCHS ${ORC_SUPPORTED_ARCH}
    SOURCES ${ORC_SOURCES} ${ORC_ASM_SOURCES}
    ADDITIONAL_HEADERS ${ORC_IMPL_HEADERS}
    CFLAGS ${ORC_CFLAGS}
    DEPS ${ORC_DEPS})

  add_compiler_rt_runtime(clang_rt.orc
    STATIC
    OS ${ORC_SUPPORTED_OS}
    ARCHS ${ORC_SUPPORTED_ARCH}
    OBJECT_LIBS RTOrc
    CFLAGS ${ORC_CFLAGS}
    LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS}
    LINK_LIBS ${ORC_LINK_LIBS}
    PARENT_TARGET orc)
else() # not Apple
  add_asm_sources(ORC_ASM_SOURCES elfnix_tls.x86-64.S)

  foreach(arch ${ORC_SUPPORTED_ARCH})
    if(NOT CAN_TARGET_${arch})
      continue()
    endif()

    add_compiler_rt_object_libraries(RTOrc
      ARCHS ${arch}
      SOURCES ${ORC_SOURCES} ${ORC_ASM_SOURCES}
      ADDITIONAL_HEADERS ${ORC_IMPL_HEADERS}
      CFLAGS ${ORC_CFLAGS}
      DEPS ${ORC_DEPS})

    # Common ORC archive for instrumented binaries.
    add_compiler_rt_runtime(clang_rt.orc
     STATIC
     ARCHS ${arch}
     CFLAGS ${ORC_CFLAGS}
     OBJECT_LIBS ${ORC_COMMON_RUNTIME_OBJECT_LIBS} RTOrc
     PARENT_TARGET orc)
  endforeach()
endif() # not Apple

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(unittests)
endif()
