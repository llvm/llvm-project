if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  add_distinct_clang_unittest(OffloadArchTests
    OffloadArchTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/offload-arch/AMDGPUArchByHIP.cpp
    CLANG_LIBS
      clangBasic
    LLVM_COMPONENTS
      Support
    )
endif()
