set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  IPO
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TargetParser
  TransformUtils
  Vectorize
  )

# Support plugins.
if(CLANG_PLUGIN_SUPPORT)
  set(support_plugins SUPPORT_PLUGINS)
endif()

set(CLANG_BOLT OFF CACHE STRING "Apply BOLT optimization to Clang. \
  May be specified as Instrument or Perf or LBR to use a particular profiling \
  mechanism.")
string(TOUPPER "${CLANG_BOLT}" CLANG_BOLT)

if (CLANG_BOLT AND NOT LLVM_BUILD_INSTRUMENTED)
  set(CLANG_BOLT_DEPS clear-bolt-fdata llvm-bolt llvm-readobj)
  if (NOT CLANG_BOLT STREQUAL "INSTRUMENT")
    list(APPEND CLANG_BOLT_DEPS clear-perf-data)
  endif()
endif()

add_clang_tool(clang
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  cc1gen_reproducer_main.cpp

  DEPENDS
  intrinsics_gen
  # These generated headers are included transitively.
  ARMTargetParserTableGen
  AArch64TargetParserTableGen
  ${support_plugins}
  ${CLANG_BOLT_DEPS}
  GENERATE_DRIVER
  )

setup_host_tool(clang CLANG clang_exe clang_target)

clang_target_link_libraries(clang
  PRIVATE
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangSerialization
  )

if(WIN32 AND NOT CYGWIN)
  # Prevent versioning if the buildhost is targeting for Win32.
else()
  set_target_properties(clang PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})
endif()

# Support plugins.
if(CLANG_PLUGIN_SUPPORT)
  export_executable_symbols_for_plugins(clang)
endif()

add_dependencies(clang clang-resource-headers)

if(NOT CLANG_LINKS_TO_CREATE)
  set(CLANG_LINKS_TO_CREATE clang++ clang-cl clang-cpp)
endif()

if (CLANG_ENABLE_HLSL)
  set(HLSL_LINK clang-dxc)
endif()

foreach(link ${CLANG_LINKS_TO_CREATE} ${HLSL_LINK})
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
  set(TOOL_INFO_BUILD_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")

  set(TOOL_INFO_PLIST_OUT "${CMAKE_CURRENT_BINARY_DIR}/${TOOL_INFO_PLIST}")

  if(LLVM_TOOL_LLVM_DRIVER_BUILD AND clang IN_LIST LLVM_DRIVER_TOOLS)
    set(TARGET_NAME llvm-driver)
  else()
    set(TARGET_NAME clang)
  endif()

  target_link_libraries(${TARGET_NAME}
    PRIVATE
    "-Wl,-sectcreate,__TEXT,__info_plist,\"${TOOL_INFO_PLIST_OUT}\"")
  configure_file("${TOOL_INFO_PLIST}.in" "${TOOL_INFO_PLIST_OUT}" @ONLY)

  set(TOOL_INFO_UTI)
  set(TOOL_INFO_NAME)
  set(TOOL_INFO_VERSION)
  set(TOOL_INFO_BUILD_VERSION)
endif()

if(CLANG_ORDER_FILE AND
    (LLVM_LINKER_IS_APPLE OR LLVM_LINKER_IS_GOLD OR LLVM_LINKER_IS_LLD))
  include(CheckLinkerFlag)

  if (LLVM_LINKER_IS_APPLE OR (LLVM_LINKER_IS_LLD AND APPLE))
    set(LINKER_ORDER_FILE_OPTION "-Wl,-order_file,${CLANG_ORDER_FILE}")
  elseif (LLVM_LINKER_IS_GOLD)
    set(LINKER_ORDER_FILE_OPTION "-Wl,--section-ordering-file,${CLANG_ORDER_FILE}")
  elseif (LLVM_LINKER_IS_LLD)
    set(LINKER_ORDER_FILE_OPTION "-Wl,--symbol-ordering-file,${CLANG_ORDER_FILE}")
  endif()

  # This is a test to ensure the actual order file works with the linker.
  check_linker_flag(CXX ${LINKER_ORDER_FILE_OPTION} LINKER_ORDER_FILE_WORKS)

  # Passing an empty order file disables some linker layout optimizations.
  # To work around this and enable workflows for re-linking when the order file
  # changes we check during configuration if the file is empty, and make it a
  # configuration dependency.
  file(READ ${CLANG_ORDER_FILE} ORDER_FILE LIMIT 20)
  if("${ORDER_FILE}" STREQUAL "\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CLANG_ORDER_FILE})
  elseif(LINKER_ORDER_FILE_WORKS)
    target_link_libraries(clang PRIVATE ${LINKER_ORDER_FILE_OPTION})
    set_target_properties(clang PROPERTIES LINK_DEPENDS ${CLANG_ORDER_FILE})
  endif()
endif()

if (CLANG_BOLT AND NOT LLVM_BUILD_INSTRUMENTED)
  # Add a clang-bolt target for backwards compatibility.
  add_custom_target(clang-bolt DEPENDS clang)

  set(CLANG_BOLT_INSTRUMENTED "clang-bolt.inst" CACHE STRING
    "Name of BOLT-instrumented Clang binary")
  set(CLANG_INSTRUMENTED ${LLVM_RUNTIME_OUTPUT_INTDIR}/${CLANG_BOLT_INSTRUMENTED})
  set(PERF_TRAINING_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../utils/perf-training)
  set(BOLT_FDATA ${PERF_TRAINING_BINARY_DIR}/prof.fdata)
  get_llvm_lit_path(
    lit_base_dir
    lit_file_name
    ALLOW_EXTERNAL
  )
  set(LIT_COMMAND "${lit_base_dir}/${lit_file_name}")

  # This POST_BUILD command is executed unconditionally even if the clang target
  # is already built.  We need to wrap the whole bolt optimization process in
  # a single python wrapper, so that we can first check if the binary has
  # already been optimized and then exit early with a 0 status if it has.
  add_custom_command(
    TARGET clang POST_BUILD
    COMMAND  "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/perf-training/perf-helper.py
             bolt-optimize
	     --method ${CLANG_BOLT}
	     --input $<TARGET_FILE:clang>
	     --instrumented-output ${CLANG_INSTRUMENTED}
	     --fdata ${BOLT_FDATA}
	     --perf-training-binary-dir ${PERF_TRAINING_BINARY_DIR}
	     --readelf $<TARGET_FILE:llvm-readobj>
	     --bolt $<TARGET_FILE:llvm-bolt>
	     --lit "${LIT_COMMAND}"
	     --merge-fdata $<TARGET_FILE:merge-fdata>
    COMMENT "Optimizing Clang with BOLT"
    USES_TERMINAL
    VERBATIM
  )
endif()
