set(LLVM_LINK_COMPONENTS
  Object
  Support
  TargetParser
  )

add_unittest(check-comgr HotswapElfTests
  HotswapElfTest.cpp
  ../src/comgr-hotswap-elf.cpp
  # COMGR::env::shouldEmitVerboseLogs() is referenced by the inline
  # COMGR::hotswap::log() helper in comgr-hotswap-internal.h; link its
  # definition so non-inlined builds (e.g. -O0 / ASan) resolve.
  ../src/comgr-env.cpp)

target_include_directories(HotswapElfTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)
