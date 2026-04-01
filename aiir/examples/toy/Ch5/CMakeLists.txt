# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS aiir/ToyCombine.td)
aiir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh5CombineIncGen)

add_toy_chapter(toyc-ch5
  toyc.cpp
  parser/AST.cpp
  aiir/AIIRGen.cpp
  aiir/Dialect.cpp
  aiir/LowerToAffineLoops.cpp
  aiir/ShapeInferencePass.cpp
  aiir/ToyCombine.cpp

  DEPENDS
  ToyCh5ShapeInferenceInterfaceIncGen
  ToyCh5OpsIncGen
  ToyCh5CombineIncGen
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch5
  PRIVATE
    AIIRAnalysis
    AIIRCallInterfaces
    AIIRCastInterfaces
    AIIRFunctionInterfaces
    AIIRIR
    AIIRParser
    AIIRPass
    AIIRRegisterAllDialects
    AIIRRegisterAllExtensions
    AIIRSideEffectInterfaces
    AIIRSupport
    AIIRTransforms
  )
