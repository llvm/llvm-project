# GTest discovery: prefer the in-tree llvm_gtest target (LLVM monorepo
# build); otherwise look for the gtest libraries / headers installed
# alongside LLVM when LLVM is built with -DLLVM_INSTALL_GTEST=ON. Comgr
# already depends on a Comgr-compatible LLVM, so we deliberately do not
# fall back to a separate system GTest install -- the test infrastructure
# tracks the LLVM we link against. If LLVM was built without
# LLVM_INSTALL_GTEST, skip the test-unit suite with a warning rather than
# failing the configure.
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
    message(WARNING
      "Comgr test-unit skipped: no llvm_gtest target and no "
      "LLVM-installed gtest found at ${LLVM_INCLUDE_DIRS}/llvm-gtest. "
      "Rebuild LLVM with -DLLVM_INSTALL_GTEST=ON to enable.")
    return()
  endif()
endif()

add_executable(HotswapElfTests
  HotswapElfTest.cpp
  ../src/comgr-hotswap-elf.cpp
  # COMGR::env::shouldEmitVerboseLogs() is referenced by the inline
  # COMGR::hotswap::log() helper in comgr-hotswap-internal.h; link its
  # definition so non-inlined builds (e.g. -O0 / ASan) resolve.
  ../src/comgr-env.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_LLVM_LIBS
  Object Support TargetParser)

target_link_libraries(HotswapElfTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${COMGR_TEST_UNIT_LLVM_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(HotswapElfTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

# LLVM is built with -fno-rtti by default; matching that here avoids
# undefined references to LLVM type_info symbols when linking against
# libLLVM*.a / .so. gtest itself does not require user-side RTTI
# (-DGTEST_HAS_RTTI=0 is set when gtest is built).
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(HotswapElfTests PRIVATE /GR-)
  else()
    target_compile_options(HotswapElfTests PRIVATE -fno-rtti)
  endif()
endif()

add_custom_target(test-unit COMMAND $<TARGET_FILE:HotswapElfTests>)
add_dependencies(test-unit HotswapElfTests)
add_dependencies(check-comgr test-unit)
