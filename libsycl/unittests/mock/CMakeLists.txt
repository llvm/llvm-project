add_library(ol_mock STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mock.cpp"
)
  add_dependencies(ol_mock
    LLVMOffload
  )

target_include_directories(ol_mock
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    $<TARGET_PROPERTY:LLVMOffload,INTERFACE_INCLUDE_DIRECTORIES>
)
