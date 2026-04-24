add_library(ol_mock STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/mock.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/helpers.cpp"
)
  add_dependencies(ol_mock
    LLVMOffload
    llvm_gtest
  )
add_definitions(-g -O0)
target_include_directories(ol_mock
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    $<TARGET_PROPERTY:LLVMOffload,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:llvm_gtest,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(ol_mock
  PRIVATE
  llvm_gtest
)
