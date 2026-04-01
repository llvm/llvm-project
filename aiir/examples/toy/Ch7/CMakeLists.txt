# This chapter depends on JIT support enabled.
if(NOT AIIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

set(LLVM_TARGET_DEFINITIONS aiir/ToyCombine.td)
aiir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh7CombineIncGen)

add_toy_chapter(toyc-ch7
  toyc.cpp
  parser/AST.cpp
  aiir/AIIRGen.cpp
  aiir/Dialect.cpp
  aiir/LowerToAffineLoops.cpp
  aiir/LowerToLLVM.cpp
  aiir/ShapeInferencePass.cpp
  aiir/ToyCombine.cpp

  DEPENDS
  ToyCh7ShapeInferenceInterfaceIncGen
  ToyCh7OpsIncGen
  ToyCh7CombineIncGen
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch7
  PRIVATE
    AIIRAnalysis
    AIIRBuiltinToLLVMIRTranslation
    AIIRCallInterfaces
    AIIRCastInterfaces
    AIIRExecutionEngine
    AIIRFunctionInterfaces
    AIIRIR
    AIIRLLVMCommonConversion
    AIIRLLVMToLLVMIRTranslation
    AIIRMemRefDialect
    AIIRParser
    AIIRPass
    AIIRRegisterAllDialects
    AIIRRegisterAllExtensions
    AIIRRegisterAllPasses
    AIIRSideEffectInterfaces
    AIIRTargetLLVMIRExport
    AIIRTransforms
  )
