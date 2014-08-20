# Get sources
set(LIBCXXABI_SOURCES
  abort_message.cpp
  cxa_aux_runtime.cpp
  cxa_default_handlers.cpp
  cxa_demangle.cpp
  cxa_exception.cpp
  cxa_exception_storage.cpp
  cxa_guard.cpp
  cxa_handlers.cpp
  cxa_new_delete.cpp
  cxa_personality.cpp
  cxa_unexpected.cpp
  cxa_vector.cpp
  cxa_virtual.cpp
  exception.cpp
  private_typeinfo.cpp
  stdexcept.cpp
  typeinfo.cpp
)

set(LIBCXXABI_HEADERS ../include/cxxabi.h)

# Add all the headers to the project for IDEs.
if (MSVC_IDE OR XCODE)
  # Force them all into the headers dir on MSVC, otherwise they end up at
  # project scope because they don't have extensions.
  if (MSVC_IDE)
    source_group("Header Files" FILES ${LIBCXXABI_HEADERS})
  endif()
endif()

if (LIBCXXABI_ENABLE_SHARED)
  add_library(cxxabi SHARED
    ${LIBCXXABI_SOURCES}
    ${LIBCXXABI_HEADERS}
    )
else()
  add_library(cxxabi STATIC
    ${LIBCXXABI_SOURCES}
    ${LIBCXXABI_HEADERS}
    )
endif()

include_directories("${LIBCXXABI_LIBCXX_INCLUDES}")

# Generate library list.
set(libraries ${LIBCXXABI_CXX_ABI_LIBRARIES})
append_if(libraries LIBCXXABI_HAS_C_LIB c)
append_if(libraries LIBCXXABI_HAS_PTHREAD_LIB pthread)

if (LIBCXXABI_USE_LLVM_UNWINDER)
  list(APPEND libraries unwind)
else()
  append_if(libraries LIBCXXABI_HAS_GCC_EH_LIB gcc_eh)
endif()

target_link_libraries(cxxabi ${libraries})

# Setup flags.
append_if(compile_flags LIBCXXABI_HAS_FPIC_FLAG -fPIC)
append_if(link_flags LIBCXXABI_HAS_NODEFAULTLIBS_FLAG -nodefaultlibs)

if ( APPLE )
  if ( CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "10.6" )
    list(APPEND compile_flags "-U__STRICT_ANSI__")
    list(APPEND link_flags
      "-compatibility_version 1"
      "-current_version 1"
      "-install_name /usr/lib/libc++abi.1.dylib"
      "/usr/lib/libSystem.B.dylib")
  else()
    list(APPEND link_flags
      "-compatibility_version 1"
      "-install_name /usr/lib/libc++abi.1.dylib")
  endif()
endif()

string(REPLACE ";" " " compile_flags "${compile_flags}")
string(REPLACE ";" " " link_flags "${link_flags}")

set_target_properties(cxxabi
  PROPERTIES
    COMPILE_FLAGS "${compile_flags}"
    LINK_FLAGS    "${link_flags}"
    OUTPUT_NAME   "c++abi"
    VERSION       "1.0"
    SOVERSION     "1"
  )

install(TARGETS cxxabi
  LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  )
