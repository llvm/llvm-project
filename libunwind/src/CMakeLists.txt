# Get sources

set(LIBUNWIND_CXX_SOURCES
    libunwind.cpp
    Unwind-EHABI.cpp
    Unwind-seh.cpp
    )
if(APPLE)
  list(APPEND LIBUNWIND_CXX_SOURCES
    Unwind_AppleExtras.cpp
    )
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "AIX")
  list(APPEND LIBUNWIND_CXX_SOURCES
    Unwind_AIXExtras.cpp
    )
endif()

set(LIBUNWIND_C_SOURCES
    UnwindLevel1.c
    UnwindLevel1-gcc-ext.c
    Unwind-sjlj.c
    )
set_source_files_properties(${LIBUNWIND_C_SOURCES}
                            PROPERTIES
                              COMPILE_FLAGS "-std=c99")

set(LIBUNWIND_ASM_SOURCES
    UnwindRegistersRestore.S
    UnwindRegistersSave.S
    )

# See add_asm_sources() in compiler-rt for explanation of this workaround.
# CMake doesn't work correctly with assembly on AIX. Workaround by compiling
# as C files as well.
if((APPLE AND CMAKE_VERSION VERSION_LESS 3.19) OR
   (MINGW AND CMAKE_VERSION VERSION_LESS 3.17) OR
   (${CMAKE_SYSTEM_NAME} MATCHES "AIX"))
  set_source_files_properties(${LIBUNWIND_ASM_SOURCES} PROPERTIES LANGUAGE C)
endif()

set(LIBUNWIND_HEADERS
    AddressSpace.hpp
    assembly.h
    CompactUnwinder.hpp
    cet_unwind.h
    config.h
    dwarf2.h
    DwarfInstructions.hpp
    DwarfParser.hpp
    EHHeaderParser.hpp
    FrameHeaderCache.hpp
    libunwind_ext.h
    Registers.hpp
    RWMutex.hpp
    Unwind-EHABI.h
    UnwindCursor.hpp
    ../include/libunwind.h
    ../include/unwind.h
    ../include/unwind_itanium.h
    ../include/unwind_arm_ehabi.h
    )
if(APPLE)
  list(APPEND LIBUNWIND_HEADERS
    ../include/mach-o/compact_unwind_encoding.h
    )
endif()

if (MSVC_IDE)
  # Force them all into the headers dir on MSVC, otherwise they end up at
  # project scope because they don't have extensions.
  source_group("Header Files" FILES ${LIBUNWIND_HEADERS})
endif()

set(LIBUNWIND_SOURCES
    ${LIBUNWIND_CXX_SOURCES}
    ${LIBUNWIND_C_SOURCES}
    ${LIBUNWIND_ASM_SOURCES})

# Generate library list.
add_library_flags_if(LIBUNWIND_HAS_C_LIB c)
if (LIBUNWIND_USE_COMPILER_RT)
  add_library_flags("${LIBUNWIND_BUILTINS_LIBRARY}")
else()
  add_library_flags_if(LIBUNWIND_HAS_GCC_S_LIB gcc_s)
  add_library_flags_if(LIBUNWIND_HAS_GCC_LIB gcc)
endif()
add_library_flags_if(LIBUNWIND_HAS_DL_LIB dl)
if (LIBUNWIND_ENABLE_THREADS)
  add_library_flags_if(LIBUNWIND_HAS_PTHREAD_LIB pthread)
  add_compile_flags_if(LIBUNWIND_WEAK_PTHREAD_LIB -DLIBUNWIND_USE_WEAK_PTHREAD=1)
endif()

# Setup flags.
add_link_flags_if(CXX_SUPPORTS_UNWINDLIB_EQ_NONE_FLAG --unwindlib=none)
if (CXX_SUPPORTS_NOSTDLIBXX_FLAG)
  add_link_flags_if_supported(-nostdlib++)
else()
  add_link_flags_if_supported(-nodefaultlibs)
endif()

# MINGW_LIBRARIES is defined in config-ix.cmake
add_library_flags_if(MINGW "${MINGW_LIBRARIES}")

if (LIBUNWIND_ENABLE_SHARED AND
    NOT (CXX_SUPPORTS_FNO_EXCEPTIONS_FLAG AND
         CXX_SUPPORTS_FUNWIND_TABLES_FLAG))
  message(FATAL_ERROR
          "Compiler doesn't support generation of unwind tables if exception "
          "support is disabled.  Building libunwind DSO with runtime dependency "
          "on C++ ABI library is not supported.")
endif()

if (APPLE)
  add_compile_flags("-U__STRICT_ANSI__")
  add_link_flags("-compatibility_version 1" "-install_name /usr/lib/libunwind.1.dylib")

  if (CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "10.6")
    add_link_flags("-current_version ${LIBUNWIND_VERSION}" "/usr/lib/libSystem.B.dylib")
  endif ()
endif ()

string(REPLACE ";" " " LIBUNWIND_COMPILE_FLAGS "${LIBUNWIND_COMPILE_FLAGS}")
string(REPLACE ";" " " LIBUNWIND_CXX_FLAGS "${LIBUNWIND_CXX_FLAGS}")
string(REPLACE ";" " " LIBUNWIND_C_FLAGS "${LIBUNWIND_C_FLAGS}")
string(REPLACE ";" " " LIBUNWIND_LINK_FLAGS "${LIBUNWIND_LINK_FLAGS}")

set_property(SOURCE ${LIBUNWIND_CXX_SOURCES}
             APPEND_STRING PROPERTY COMPILE_FLAGS " ${LIBUNWIND_CXX_FLAGS}")
set_property(SOURCE ${LIBUNWIND_C_SOURCES}
             APPEND_STRING PROPERTY COMPILE_FLAGS " ${LIBUNWIND_C_FLAGS}")

# NOTE: avoid implicit dependencies on C++ runtimes.  libunwind uses C++ for
# ease, but does not rely on C++ at runtime.
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "")

# Build the shared library.
add_library(unwind_shared_objects OBJECT EXCLUDE_FROM_ALL ${LIBUNWIND_SOURCES} ${LIBUNWIND_HEADERS})
if(CMAKE_C_COMPILER_ID STREQUAL MSVC)
  target_compile_options(unwind_shared_objects PRIVATE /GR-)
else()
  target_compile_options(unwind_shared_objects PRIVATE -fno-rtti)
endif()
target_link_libraries(unwind_shared_objects PRIVATE unwind-headers ${LIBUNWIND_LIBRARIES})
set_target_properties(unwind_shared_objects
  PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    COMPILE_FLAGS "${LIBUNWIND_COMPILE_FLAGS}"
)
if (CMAKE_POSITION_INDEPENDENT_CODE OR NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set_target_properties(unwind_shared_objects PROPERTIES POSITION_INDEPENDENT_CODE ON) # must set manually because it's an object library
endif()

if (LIBUNWIND_ENABLE_SHARED)
  add_library(unwind_shared SHARED)
  target_link_libraries(unwind_shared PUBLIC unwind_shared_objects)
  set_target_properties(unwind_shared
    PROPERTIES
      LINK_FLAGS "${LIBUNWIND_LINK_FLAGS}"
      LINKER_LANGUAGE C
      OUTPUT_NAME "${LIBUNWIND_SHARED_OUTPUT_NAME}"
      VERSION "1.0"
      SOVERSION "1"
  )

  list(APPEND LIBUNWIND_BUILD_TARGETS "unwind_shared")
  if (LIBUNWIND_INSTALL_SHARED_LIBRARY)
    list(APPEND LIBUNWIND_INSTALL_TARGETS "unwind_shared")
  endif()
endif()

# Build the static library.
add_library(unwind_static_objects OBJECT EXCLUDE_FROM_ALL ${LIBUNWIND_SOURCES} ${LIBUNWIND_HEADERS})
if(CMAKE_C_COMPILER_ID STREQUAL MSVC)
  target_compile_options(unwind_static_objects PRIVATE /GR-)
else()
  target_compile_options(unwind_static_objects PRIVATE -fno-rtti)
endif()
target_link_libraries(unwind_static_objects PRIVATE unwind-headers ${LIBUNWIND_LIBRARIES})
set_target_properties(unwind_static_objects
  PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    COMPILE_FLAGS "${LIBUNWIND_COMPILE_FLAGS}"
)

if(LIBUNWIND_HIDE_SYMBOLS)
  target_add_compile_flags_if_supported(unwind_static_objects PRIVATE -fvisibility=hidden)
  target_add_compile_flags_if_supported(unwind_static_objects PRIVATE -fvisibility-global-new-delete-hidden)
  target_compile_definitions(unwind_static_objects PRIVATE _LIBUNWIND_HIDE_SYMBOLS)
endif()

if (LIBUNWIND_ENABLE_STATIC)
  add_library(unwind_static STATIC)
  target_link_libraries(unwind_static PUBLIC unwind_static_objects)
  set_target_properties(unwind_static
    PROPERTIES
      LINK_FLAGS "${LIBUNWIND_LINK_FLAGS}"
      LINKER_LANGUAGE C
      OUTPUT_NAME "${LIBUNWIND_STATIC_OUTPUT_NAME}"
  )

  list(APPEND LIBUNWIND_BUILD_TARGETS "unwind_static")
  if (LIBUNWIND_INSTALL_STATIC_LIBRARY)
    list(APPEND LIBUNWIND_INSTALL_TARGETS "unwind_static")
  endif()
endif()

# Add a meta-target for both libraries.
add_custom_target(unwind DEPENDS ${LIBUNWIND_BUILD_TARGETS})

if (LIBUNWIND_INSTALL_LIBRARY)
  install(TARGETS ${LIBUNWIND_INSTALL_TARGETS}
    LIBRARY DESTINATION ${LIBUNWIND_INSTALL_LIBRARY_DIR} COMPONENT unwind
    ARCHIVE DESTINATION ${LIBUNWIND_INSTALL_LIBRARY_DIR} COMPONENT unwind
    RUNTIME DESTINATION ${LIBUNWIND_INSTALL_RUNTIME_DIR} COMPONENT unwind)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES AND LIBUNWIND_INSTALL_LIBRARY)
  add_custom_target(install-unwind
    DEPENDS unwind
    COMMAND "${CMAKE_COMMAND}"
            -DCMAKE_INSTALL_COMPONENT=unwind
            -P "${LIBUNWIND_BINARY_DIR}/cmake_install.cmake")
  add_custom_target(install-unwind-stripped
    DEPENDS unwind
    COMMAND "${CMAKE_COMMAND}"
            -DCMAKE_INSTALL_COMPONENT=unwind
            -DCMAKE_INSTALL_DO_STRIP=1
            -P "${LIBUNWIND_BINARY_DIR}/cmake_install.cmake")
  if(LIBUNWIND_INSTALL_HEADERS)
    add_dependencies(install-unwind install-unwind-headers)
    add_dependencies(install-unwind-stripped install-unwind-headers-stripped)
  endif()
endif()
