# For a better template to copy, see examples/standalone
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(toyc-ch2
  toyc.cpp
  parser/AST.cpp
  aiir/AIIRGen.cpp
  aiir/Dialect.cpp

  DEPENDS
  ToyCh2OpsIncGen

  )
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch2
  PRIVATE
    AIIRAnalysis
    AIIRFunctionInterfaces
    AIIRIR
    AIIRParser
    AIIRSideEffectInterfaces
    AIIRTransforms)
