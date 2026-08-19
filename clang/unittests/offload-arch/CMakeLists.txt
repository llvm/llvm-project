set(OffloadArchTestSources
  OffloadArchTest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/offload-arch/AMDGPUArchByKFD.cpp
  )

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  list(APPEND OffloadArchTestSources
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/offload-arch/AMDGPUArchByHIP.cpp)
endif()

add_distinct_clang_unittest(OffloadArchTests
  ${OffloadArchTestSources}
  CLANG_LIBS
    clangBasic
  LINK_LIBS
    LLVMTestingSupport
  LLVM_COMPONENTS
    Support
  )
