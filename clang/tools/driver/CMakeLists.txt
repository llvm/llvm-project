set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  IPO
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
  )

option(CLANG_PLUGIN_SUPPORT "Build clang with plugin support" ON)

# Support plugins. This must be before add_clang_executable as it reads
# LLVM_NO_DEAD_STRIP.
if(CLANG_PLUGIN_SUPPORT)
  set(LLVM_NO_DEAD_STRIP 1)
endif()

add_clang_tool(clang
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  )

target_link_libraries(clang
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  )

if(WIN32 AND NOT CYGWIN)
  # Prevent versioning if the buildhost is targeting for Win32.
else()
  set_target_properties(clang PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})
endif()

# Support plugins.
if(CLANG_PLUGIN_SUPPORT)
  export_executable_symbols(clang)
endif()

add_dependencies(clang clang-headers)

if(NOT CLANG_LINKS_TO_CREATE)
  set(CLANG_LINKS_TO_CREATE clang++ clang-cl)

  if (WIN32)
    list(APPEND CLANG_LINKS_TO_CREATE ../msbuild-bin/cl)
  endif()
endif()

foreach(link ${CLANG_LINKS_TO_CREATE})
  add_clang_symlink(${link} clang)
endforeach()

# Configure plist creation for OS X.
set (TOOL_INFO_PLIST "Info.plist" CACHE STRING "Plist name")
if (APPLE)  
  if (CLANG_VENDOR)
    set(TOOL_INFO_NAME "${CLANG_VENDOR} clang")
  else()
    set(TOOL_INFO_NAME "clang")
  endif()

  set(TOOL_INFO_UTI "${CLANG_VENDOR_UTI}")
  set(TOOL_INFO_VERSION "${CLANG_VERSION}")
  set(TOOL_INFO_BUILD_VERSION "${LLVM_MAJOR_VERSION}.${LLVM_MINOR_VERSION}")
  
  set(TOOL_INFO_PLIST_OUT "${CMAKE_CURRENT_BINARY_DIR}/${TOOL_INFO_PLIST}")
  target_link_libraries(clang
    "-Wl,-sectcreate,__TEXT,__info_plist,${TOOL_INFO_PLIST_OUT}")
  configure_file("${TOOL_INFO_PLIST}.in" "${TOOL_INFO_PLIST_OUT}" @ONLY)
  
  set(TOOL_INFO_UTI)
  set(TOOL_INFO_NAME)
  set(TOOL_INFO_VERSION)
  set(TOOL_INFO_BUILD_VERSION)
endif()

# the linker -order_file flag is only supported by ld64
if(LD64_EXECUTABLE AND CLANG_ORDER_FILE)
  include(CMakePushCheckState)

  function(check_linker_flag flag out_var)
    cmake_push_check_state()
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${flag}")
    check_cxx_compiler_flag("" ${out_var})
    cmake_pop_check_state()
  endfunction()

  # This is a test to ensure the actual order file works with the linker.
  check_linker_flag("-Wl,-order_file,${CLANG_ORDER_FILE}"
    LINKER_ORDER_FILE_WORKS)

  # Passing an empty order file disables some linker layout optimizations.
  # To work around this and enable workflows for re-linking when the order file
  # changes we check during configuration if the file is empty, and make it a
  # configuration dependency.
  file(READ ${CLANG_ORDER_FILE} ORDER_FILE LIMIT 20)
  if("${ORDER_FILE}" STREQUAL "\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CLANG_ORDER_FILE})
  elseif(LINKER_ORDER_FILE_WORKS)
    target_link_libraries(clang "-Wl,-order_file,${CLANG_ORDER_FILE}")
    set_target_properties(clang PROPERTIES LINK_DEPENDS ${CLANG_ORDER_FILE})
  endif()
endif()

if(WITH_POLLY AND LINK_POLLY_INTO_TOOLS)
  target_link_libraries(clang Polly)
  if(POLLY_LINK_LIBS)
    foreach(lib ${POLLY_LINK_LIBS})
      target_link_libraries(clang ${lib})
    endforeach(lib)
  endif(POLLY_LINK_LIBS)
endif(WITH_POLLY AND LINK_POLLY_INTO_TOOLS)
