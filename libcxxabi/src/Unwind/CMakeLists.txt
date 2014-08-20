# Get sources
set(LIBUNWIND_SOURCES
  libunwind.cpp
  Unwind-EHABI.cpp
  UnwindLevel1.c
  UnwindLevel1-gcc-ext.c
  Unwind-sjlj.c
)

append_if(LIBUNWIND_SOURCES APPLE Unwind_AppleExtras.cpp)

set(LIBUNWIND_ASM_SOURCES
  UnwindRegistersRestore.S
  UnwindRegistersSave.S
)

set_source_files_properties(${LIBUNWIND_ASM_SOURCES} PROPERTIES LANGUAGE C)

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
  UnwindCursor.hpp
  unwind_ext.h
  ../../include/libunwind.h
  ../../include/unwind.h
)

append_if(LIBCXXABI_HEADERS APPLE ../../include/mach-o/compact_unwind_encoding.h)

if (MSVC_IDE)
  # Force them all into the headers dir on MSVC, otherwise they end up at
  # project scope because they don't have extensions.
  source_group("Header Files" FILES ${LIBUNWIND_HEADERS})
endif()

if (LIBUNWIND_ENABLE_SHARED)
  add_library(unwind SHARED
    ${LIBUNWIND_SOURCES}
    ${LIBUNWIND_ASM_SOURCES}
    ${LIBUNWIND_HEADERS}
    )
else()
  add_library(unwind STATIC
    ${LIBUNWIND_SOURCES}
    ${LIBUNWIND_ASM_SOURCES}
    ${LIBUNWIND_HEADERS}
    )
endif()

include_directories("${LIBCXXABI_LIBCXX_INCLUDES}")

# Generate library list.
set(libraries ${LIBCXXABI_CXX_ABI_LIBRARIES})
append_if(libraries LIBCXXABI_HAS_C_LIB c)
append_if(libraries LIBCXXABI_HAS_DL_LIB dl)
append_if(libraries LIBCXXABI_HAS_PTHREAD_LIB pthread)

target_link_libraries(unwind ${libraries})

# Setup flags.
append_if(compile_flags LIBCXXABI_HAS_FPIC_FLAG -fPIC)
append_if(link_flags LIBCXXABI_HAS_NODEFAULTLIBS_FLAG -nodefaultlibs)

set(LIBCXXABI_UNWINDER_NAME "unwind")

if ( APPLE )
  if ( CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "10.6" )
    list(APPEND compile_flags "-U__STRICT_ANSI__")
    list(APPEND link_flags
      "-compatibility_version 1"
      "-current_version ${LIBCXXABI_VERSION}"
      "-install_name /usr/lib/lib${LIBCXXABI_UNWINDER_NAME}.1.dylib"
      "/usr/lib/libSystem.B.dylib")
  else()
    list(APPEND link_flags
      "-compatibility_version 1"
      "-install_name /usr/lib/lib${LIBCXXABI_UNWINDER_NAME}.1.dylib")
  endif()
endif()

string(REPLACE ";" " " compile_flags "${compile_flags}")
string(REPLACE ";" " " link_flags "${link_flags}")

set_target_properties(unwind
  PROPERTIES
    COMPILE_FLAGS "${compile_flags}"
    LINK_FLAGS    "${link_flags}"
    OUTPUT_NAME   "${LIBCXXABI_UNWINDER_NAME}"
    VERSION       "1.0"
    SOVERSION     "1"
  )

install(TARGETS unwind
  LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  )
