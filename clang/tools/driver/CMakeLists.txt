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

add_clang_executable(clang
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

install(TARGETS clang
  RUNTIME DESTINATION bin
  COMPONENT clang)
add_custom_target(install-clang
  DEPENDS clang
  COMMAND "${CMAKE_COMMAND}"
          -DCMAKE_INSTALL_COMPONENT=clang
          -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")

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

if(CLANG_ORDER_FILE)
  target_link_libraries(clang "-Wl,-order_file,${CLANG_ORDER_FILE}")
endif()

if(WITH_POLLY AND LINK_POLLY_INTO_TOOLS)
  target_link_libraries(clang Polly)
  if(POLLY_LINK_LIBS)
    foreach(lib ${POLLY_LINK_LIBS})
      target_link_libraries(clang ${lib})
    endforeach(lib)
  endif(POLLY_LINK_LIBS)
endif(WITH_POLLY AND LINK_POLLY_INTO_TOOLS)
