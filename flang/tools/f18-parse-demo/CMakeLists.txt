add_flang_tool(f18-parse-demo
  f18-parse-demo.cpp
  stub-evaluate.cpp
)

target_link_libraries(f18-parse-demo
  PRIVATE
  FortranParser
  )
