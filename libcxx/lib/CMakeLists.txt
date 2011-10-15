# Get sources
file(GLOB sources ../src/*.cpp)
if(WIN32)
  file(GLOB win32_sources ../src/support/win32/*.cpp)
  list(APPEND sources ${win32_sources})
endif()

# Add all the headers to the project for IDEs.
if (MSVC_IDE OR XCODE)
  file(GLOB_RECURSE headers ../include/*)
  if(WIN32)
    file( GLOB win32_headers ../include/support/win32/*.h)
    list(APPEND headers ${win32_headers})
  endif()
  # Force them all into the headers dir on MSVC, otherwise they end up at
  # project scope because they don't have extensions.
  if (MSVC_IDE)
    source_group("Header Files" FILES ${headers})
  endif()
endif()

if (LIBCXX_ENABLE_SHARED)
  add_library(cxx SHARED
    ${sources}
    ${headers}
    )
else()
  add_library(cxx STATIC
    ${sources}
    ${headers}
    )
endif()

# Generate library list.
append_if(libraries LIBCXX_HAS_PTHREAD_LIB pthread)
append_if(libraries LIBCXX_HAS_C_LIB c)
append_if(libraries LIBCXX_HAS_M_LIB m)
append_if(libraries LIBCXX_HAS_RT_LIB rt)
append_if(libraries LIBCXX_HAS_GCC_S_LIB gcc_s)

target_link_libraries(cxx ${libraries})

# Setup flags.
append_if(compile_flags LIBCXX_HAS_FPIC_FLAG -fPIC)
append_if(link_flags LIBCXX_HAS_NODEFAULTLIBS_FLAG -nodefaultlibs)

set_target_properties(cxx
  PROPERTIES
    COMPILE_FLAGS "${compile_flags}"
    LINK_FLAGS    "${link_flags}"
    OUTPUT_NAME   "c++"
    VERSION       "1.0"
    SOVERSION     "1"
  )

install(TARGETS cxx
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

install(DIRECTORY ../include/
  DESTINATION include/c++/v1
  FILES_MATCHING
  PATTERN "*"
  PATTERN ".svn" EXCLUDE
  )
