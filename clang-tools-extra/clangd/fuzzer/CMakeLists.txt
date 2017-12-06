include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

set(LLVM_LINK_COMPONENTS support)

if(LLVM_USE_SANITIZE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")
endif()

add_clang_executable(clangd-fuzzer
  EXCLUDE_FROM_ALL
  ClangdFuzzer.cpp
  )

target_link_libraries(clangd-fuzzer
  PRIVATE
  clangBasic
  clangDaemon
  clangFormat
  clangFrontend
  clangSema
  clangTooling
  clangToolingCore
  ${LLVM_LIB_FUZZING_ENGINE}
  )
