if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  add_distinct_clang_unittest(OffloadArchTests
    OffloadArchTest.cpp
    ${CMAKE_SOURCE_DIR}/clang/tools/offload-arch/AMDGPUArchByHIP.cpp
    CLANG_LIBS
      clangBasic
    LLVM_COMPONENTS
      Support
    )
endif()
