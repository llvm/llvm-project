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
  FortranSupport
  FortranParser
  )

# stub-evaluate.cpp uses struct/class mismatches with other TU definitions
# that trigger -Werror,-Wmismatched-tags in unity builds.
set_source_files_properties(stub-evaluate.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
