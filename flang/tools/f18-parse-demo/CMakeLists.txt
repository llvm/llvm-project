include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../FortranRuntime/include")

set(LLVM_LINK_COMPONENTS
  FrontendOpenACC
  FrontendOpenMP
  Support
  )

add_flang_tool(f18-parse-demo
  f18-parse-demo.cpp
  stub-evaluate.cpp
)

target_link_libraries(f18-parse-demo
  PRIVATE
  FortranCommon
  FortranParser
  )
