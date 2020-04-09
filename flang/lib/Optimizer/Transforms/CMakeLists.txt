add_llvm_library(FIRTransforms
  CSE.cpp
  MemToReg.cpp
  RewriteLoop.cpp
)

add_dependencies(FIRTransforms FIROpsIncGen)

target_link_libraries(FIRTransforms FIROptimizer)

install (TARGETS FIRTransforms
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
