# Get sources

set(LIBUNWIND_CXX_SOURCES
    libunwind.cpp
    Unwind-EHABI.cpp)
append_if(LIBUNWIND_CXX_SOURCES APPLE Unwind_AppleExtras.cpp)

set(LIBUNWIND_C_SOURCES
    UnwindLevel1.c
    UnwindLevel1-gcc-ext.c
    Unwind-sjlj.c)
set_source_files_properties(${LIBUNWIND_C_SOURCES}
                            PROPERTIES
                              COMPILE_FLAGS "-std=c99")

set(LIBUNWIND_ASM_SOURCES
    UnwindRegistersRestore.S
    UnwindRegistersSave.S)
set_source_files_properties(${LIBUNWIND_ASM_SOURCES}
                            PROPERTIES
                              LANGUAGE C)

set(LIBUNWIND_HEADERS
    AddressSpace.hpp
    assembly.h
    CompactUnwinder.hpp
    config.h
    dwarf2.h
    DwarfInstructions.hpp
    DwarfParser.hpp
    libunwind_ext.h
    Registers.hpp
    RWMutex.hpp
    UnwindCursor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/libunwind.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/unwind.h)

append_if(LIBUNWIND_HEADERS APPLE
          "${CMAKE_CURRENT_SOURCE_DIR}/../include/mach-o/compact_unwind_encoding.h")

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
set(libraries ${LIBUNWINDCXX_ABI_LIBRARIES})
append_if(libraries LIBUNWIND_HAS_C_LIB c)
append_if(libraries LIBUNWIND_HAS_DL_LIB dl)
if (LIBUNWIND_ENABLE_THREADS)
  append_if(libraries LIBUNWIND_HAS_PTHREAD_LIB pthread)
endif()

# Setup flags.
append_if(LIBUNWIND_CXX_FLAGS LIBUNWIND_HAS_NO_RTTI_FLAG -fno-rtti)

append_if(LIBUNWIND_LINK_FLAGS LIBUNWIND_HAS_NODEFAULTLIBS_FLAG -nodefaultlibs)

if (LIBUNWIND_HAS_NO_EXCEPTIONS_FLAG AND LIBUNWIND_HAS_FUNWIND_TABLES)
  list(APPEND LIBUNWIND_COMPILE_FLAGS -fno-exceptions)
  list(APPEND LIBUNWIND_COMPILE_FLAGS -funwind-tables)
elseif (LIBUNWIND_ENABLE_SHARED)
  message(FATAL_ERROR
          "Compiler doesn't support generation of unwind tables if exception "
          "support is disabled.  Building libunwind DSO with runtime dependency "
          "on C++ ABI library is not supported.")
endif()

if (APPLE)
  list(APPEND LIBUNWIND_COMPILE_FLAGS "-U__STRICT_ANSI__")
  list(APPEND LIBUNWIND_LINK_FLAGS
       "-compatibility_version 1"
       "-install_name /usr/lib/libunwind.1.dylib")

  if (CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "10.6")
    list(APPEND LIBUNWIND_LINK_FLAGS
         "-current_version ${LIBUNWIND_VERSION}"
         "/usr/lib/libSystem.B.dylib")
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

# Add a object library that contains the compiled source files.
add_library(unwind_objects OBJECT ${LIBUNWIND_SOURCES} ${LIBUNWIND_HEADERS})

set_target_properties(unwind_objects
                      PROPERTIES
                        COMPILE_FLAGS "${LIBUNWIND_COMPILE_FLAGS}"
                        POSITION_INDEPENDENT_CODE ON)

set(LIBUNWIND_TARGETS)

# Build the shared library.
if (LIBUNWIND_ENABLE_SHARED)
  add_library(unwind_shared SHARED $<TARGET_OBJECTS:unwind_objects>)
  if(COMMAND llvm_setup_rpath)
    llvm_setup_rpath(unwind_shared)
  endif()
  target_link_libraries(unwind_shared ${libraries})
  set_target_properties(unwind_shared
                        PROPERTIES
                          LINK_FLAGS    "${LIBUNWIND_LINK_FLAGS}"
                          OUTPUT_NAME   "unwind"
                          VERSION       "1.0"
                          SOVERSION     "1")
  list(APPEND LIBUNWIND_TARGETS "unwind_shared")
endif()

# Build the static library.
if (LIBUNWIND_ENABLE_STATIC)
  add_library(unwind_static STATIC $<TARGET_OBJECTS:unwind_objects>)
  target_link_libraries(unwind_static ${libraries})
  set_target_properties(unwind_static
                        PROPERTIES
                          LINK_FLAGS    "${LIBUNWIND_LINK_FLAGS}"
                          OUTPUT_NAME   "unwind")
  list(APPEND LIBUNWIND_TARGETS "unwind_static")
endif()

# Add a meta-target for both libraries.
add_custom_target(unwind DEPENDS ${LIBUNWIND_TARGETS})

if (LIBUNWIND_INSTALL_LIBRARY)
  install(TARGETS ${LIBUNWIND_TARGETS}
    LIBRARY DESTINATION ${LIBUNWIND_INSTALL_PREFIX}lib${LIBUNWIND_LIBDIR_SUFFIX} COMPONENT unwind
    ARCHIVE DESTINATION ${LIBUNWIND_INSTALL_PREFIX}lib${LIBUNWIND_LIBDIR_SUFFIX} COMPONENT unwind)
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
endif()
