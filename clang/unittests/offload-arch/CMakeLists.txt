if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(LLVM_LINK_COMPONENTS Support)

  add_clang_unittest(OffloadArchTests
    OffloadArchTest.cpp
    ${CMAKE_SOURCE_DIR}/clang/tools/offload-arch/AMDGPUArchByHIP.cpp
    )

  target_include_directories(OffloadArchTests PRIVATE
    ${CMAKE_SOURCE_DIR}/clang/tools/offload-arch
    )

  target_link_libraries(OffloadArchTests PRIVATE
    clangBasic
    )
endif()
