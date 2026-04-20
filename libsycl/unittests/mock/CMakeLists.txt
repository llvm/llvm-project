add_library(ol_mock STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/mock.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/helpers.cpp"
)
  add_dependencies(ol_mock
    LLVMOffload
  )
add_definitions(-g -O0)
target_include_directories(ol_mock
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    $<TARGET_PROPERTY:LLVMOffload,INTERFACE_INCLUDE_DIRECTORIES>
)
