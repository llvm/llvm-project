add_llvm_tool(f18-parse-demo
  f18-parse-demo.cpp
  stub-evaluate.cpp
  )
set_property(TARGET f18-parse-demo PROPERTY CXX_STANDARD 17)
target_compile_features(f18-parse-demo PRIVATE cxx_std_17)

target_link_libraries(f18-parse-demo
  PRIVATE
  FortranParser
  )

#install(TARGETS f18-parse-demo DESTINATION bin)
