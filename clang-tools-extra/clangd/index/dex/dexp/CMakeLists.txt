include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../)

set(LLVM_LINK_COMPONENTS
  LineEditor
  Support
  )

add_clang_executable(dexp
  Dexp.cpp
  )

clang_target_link_libraries(dexp
  PRIVATE
  clangBasic
  )
target_link_libraries(dexp
  PRIVATE
  clangDaemon
  )
