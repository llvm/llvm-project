# GTest discovery for Comgr unit tests. Three sources, in priority order:
#
# 1. In-tree llvm_gtest target (LLVM monorepo build with utils/unittest/).
# 2. LLVM-installed gtest (-DLLVM_INSTALL_GTEST=ON on the LLVM build),
#    discovered via find_library in LLVM_LIBRARY_DIRS with NO_DEFAULT_PATH.
#    Preferred over find_package(GTest) for standalone Comgr builds against
#    an installed ROCm/LLVM, because find_package(GTest CONFIG) searches
#    CMake's default paths and can pick up a system gtest at /usr/local
#    that wasn't built with -fPIC, breaking PIE link of the test binaries
#    (https://github.com/ROCm/llvm-project/issues/2505).
# 3. Vanilla GTest CMake package (find_package(GTest CONFIG)). Used by
#    superproject builds (e.g. TheRock) that supply googletest via the
#    standard GTest:: imported targets. This avoids forcing the LLVM
#    build to enable LLVM_INSTALL_GTEST just to test Comgr.
#
# If none is available, skip the test-unit suite with a warning rather
# than failing configure.
if(TARGET llvm_gtest)
  set(COMGR_GTEST_LIBS llvm_gtest_main llvm_gtest)
  set(COMGR_GTEST_INCLUDE_DIRS "")
else()
  find_library(COMGR_LLVM_GTEST_LIB llvm_gtest
    PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
  find_library(COMGR_LLVM_GTEST_MAIN_LIB llvm_gtest_main
    PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
  find_path(COMGR_LLVM_GTEST_INCLUDE_DIR gtest/gtest.h
    PATHS ${LLVM_INCLUDE_DIRS} PATH_SUFFIXES llvm-gtest NO_DEFAULT_PATH)
  if(COMGR_LLVM_GTEST_LIB AND COMGR_LLVM_GTEST_MAIN_LIB
      AND COMGR_LLVM_GTEST_INCLUDE_DIR)
    set(COMGR_GTEST_LIBS
      ${COMGR_LLVM_GTEST_MAIN_LIB} ${COMGR_LLVM_GTEST_LIB})
    set(COMGR_GTEST_INCLUDE_DIRS ${COMGR_LLVM_GTEST_INCLUDE_DIR})
  else()
    find_package(GTest CONFIG QUIET)
    if(GTest_FOUND)
      set(COMGR_GTEST_LIBS GTest::gtest_main GTest::gtest)
      set(COMGR_GTEST_INCLUDE_DIRS "")
    else()
      message(WARNING
        "Comgr test-unit skipped: no llvm_gtest target, no LLVM-installed "
        "gtest at ${LLVM_INCLUDE_DIRS}/llvm-gtest, and no GTest CMake "
        "package found. Provide one of: in-tree LLVM with utils/unittest/, "
        "-DLLVM_INSTALL_GTEST=ON on the LLVM build, or find_package(GTest).")
      return()
    endif()
  endif()
endif()

# Pin C++17 and mirror LLVM's RTTI setting (avoids type_info link errors
# against LLVM libs built with RTTI off).
function(comgr_configure_test_target target)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED Yes
    CXX_EXTENSIONS No)
  if(NOT LLVM_ENABLE_RTTI)
    if(MSVC)
      target_compile_options(${target} PRIVATE /GR-)
    else()
      target_compile_options(${target} PRIVATE -fno-rtti)
    endif()
  endif()
endfunction()

# -- DemangleSymbolNameTest ---------------------------------------------------

add_executable(DemangleSymbolNameTest
  DemangleSymbolNameTest.cpp)

target_link_libraries(DemangleSymbolNameTest PRIVATE
  ${COMGR_GTEST_LIBS}
  amd_comgr
  LLVMSupport
  ${LLVM_PTHREAD_LIB})

# Disable LLVM printers in this test. LLVM's prebuilt GTest main still needs
# LLVMSupport for its ABI-check reference when assertions are enabled.
target_compile_definitions(DemangleSymbolNameTest PRIVATE GTEST_NO_LLVM_SUPPORT=true)

target_include_directories(DemangleSymbolNameTest PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(DemangleSymbolNameTest)

# amd_comgr is a shared lib; the test-unit target runs the exe directly
# without the PATH setup the lit suite uses, so place the DLL next to it.
if(WIN32 AND COMGR_BUILD_SHARED_LIBS)
  add_custom_command(TARGET DemangleSymbolNameTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:amd_comgr> $<TARGET_FILE_DIR:DemangleSymbolNameTest>)
endif()

# -- LoggerTests --------------------------------------------------------------
#
# Tests for the global COMGR::Logger (comgr-logger.cpp). Needs only llvm
# Support for raw_ostream / file handling; comgr-env.cpp supplies the
# environment lookups the Logger constructor consumes.

add_executable(LoggerTests
  LoggerTest.cpp
  ../src/comgr-logger.cpp
  ../src/comgr-env.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_SUPPORT_LIBS Support)

target_link_libraries(LoggerTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${COMGR_TEST_UNIT_SUPPORT_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(LoggerTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(LoggerTests)

# -- MetadataCacheTest --------------------------------------------------------
#
# The DataObject metadata cache is only observable as pointer identity between
# two amd_comgr_metadata_node_t handles, so this reaches into src/comgr.h rather
# than going through the public API alone. Reads the code object straight out of
# the CTest suite's source tree; it is checked in, so there is no build-order
# dependency on that suite.

add_executable(MetadataCacheTest
  MetadataCacheTest.cpp)

target_link_libraries(MetadataCacheTest PRIVATE
  ${COMGR_GTEST_LIBS}
  amd_comgr
  ${COMGR_TEST_UNIT_SUPPORT_LIBS}
  ${LLVM_PTHREAD_LIB})

target_compile_definitions(MetadataCacheTest PRIVATE
  GTEST_NO_LLVM_SUPPORT=true
  TEST_CODE_OBJECT="${CMAKE_CURRENT_SOURCE_DIR}/../test/source/legacy/shared-v3.so")

target_include_directories(MetadataCacheTest PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(MetadataCacheTest)

if(WIN32 AND COMGR_BUILD_SHARED_LIBS)
  add_custom_command(TARGET MetadataCacheTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:amd_comgr> $<TARGET_FILE_DIR:MetadataCacheTest>)
endif()

# -- SymbolIndexTest ----------------------------------------------------------
#
# The cached object-file parse and the symbol name index are only observable as
# state on the DataObject, so this reaches into src/comgr.h rather than going
# through the public API alone. Reads the code object straight out of the CTest
# suite's source tree; it is checked in, so there is no build-order dependency
# on that suite.

add_executable(SymbolIndexTest
  SymbolIndexTest.cpp)

target_link_libraries(SymbolIndexTest PRIVATE
  ${COMGR_GTEST_LIBS}
  amd_comgr
  ${COMGR_TEST_UNIT_SUPPORT_LIBS}
  ${LLVM_PTHREAD_LIB})

target_compile_definitions(SymbolIndexTest PRIVATE
  GTEST_NO_LLVM_SUPPORT=true
  TEST_CODE_OBJECT="${CMAKE_CURRENT_SOURCE_DIR}/../test/source/legacy/shared-v3.so")

target_include_directories(SymbolIndexTest PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(SymbolIndexTest)

if(WIN32 AND COMGR_BUILD_SHARED_LIBS)
  add_custom_command(TARGET SymbolIndexTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:amd_comgr> $<TARGET_FILE_DIR:SymbolIndexTest>)
endif()

# -- IncludeMaterializationTest -----------------------------------------------
#
# Includes are served out of the in-memory VFS rather than real temp files, so
# this runs the preprocessor with the filesystem pinned both ways via
# amd_comgr_action_info_set_vfs. Public API only; AMDGPUCompiler's VFS state is
# private and not exported from the shared library.

add_executable(IncludeMaterializationTest
  IncludeMaterializationTest.cpp)

target_link_libraries(IncludeMaterializationTest PRIVATE
  ${COMGR_GTEST_LIBS}
  amd_comgr
  ${LLVM_PTHREAD_LIB})

target_compile_definitions(IncludeMaterializationTest PRIVATE
  GTEST_NO_LLVM_SUPPORT=true)

target_include_directories(IncludeMaterializationTest PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(IncludeMaterializationTest)

if(WIN32 AND COMGR_BUILD_SHARED_LIBS)
  add_custom_command(TARGET IncludeMaterializationTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:amd_comgr> $<TARGET_FILE_DIR:IncludeMaterializationTest>)
endif()

# Register every test binary with the test-unit / check-comgr plumbing.
add_custom_target(test-unit
  COMMAND $<TARGET_FILE:DemangleSymbolNameTest>
  COMMAND $<TARGET_FILE:LoggerTests>
  COMMAND $<TARGET_FILE:MetadataCacheTest>
  COMMAND $<TARGET_FILE:SymbolIndexTest>
  COMMAND $<TARGET_FILE:IncludeMaterializationTest>)
add_dependencies(test-unit
  DemangleSymbolNameTest LoggerTests MetadataCacheTest SymbolIndexTest
  IncludeMaterializationTest)

add_dependencies(check-comgr test-unit)
