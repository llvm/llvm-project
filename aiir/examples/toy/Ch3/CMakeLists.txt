# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS aiir/ToyCombine.td)
aiir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh3CombineIncGen)

add_toy_chapter(toyc-ch3
  toyc.cpp
  parser/AST.cpp
  aiir/AIIRGen.cpp
  aiir/Dialect.cpp
  aiir/ToyCombine.cpp

  DEPENDS
  ToyCh3OpsIncGen
  ToyCh3CombineIncGen
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch3
  PRIVATE
    AIIRAnalysis
    AIIRFunctionInterfaces
    AIIRIR
    AIIRParser
    AIIRPass
    AIIRSideEffectInterfaces
    AIIRTransforms)

