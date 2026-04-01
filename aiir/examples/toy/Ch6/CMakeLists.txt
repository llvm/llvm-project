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
add_public_tablegen_target(ToyCh6CombineIncGen)

add_toy_chapter(toyc-ch6
  toyc.cpp
  parser/AST.cpp
  aiir/AIIRGen.cpp
  aiir/Dialect.cpp
  aiir/LowerToAffineLoops.cpp
  aiir/LowerToLLVM.cpp
  aiir/ShapeInferencePass.cpp
  aiir/ToyCombine.cpp

  DEPENDS
  ToyCh6ShapeInferenceInterfaceIncGen
  ToyCh6OpsIncGen
  ToyCh6CombineIncGen
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch6
  PRIVATE
    AIIRAnalysis
    AIIRBuiltinToLLVMIRTranslation
    AIIRCallInterfaces
    AIIRCastInterfaces
    AIIRExecutionEngine
    AIIRFunctionInterfaces
    AIIRIR
    AIIRLLVMCommonConversion
    AIIRLLVMDialect
    AIIRLLVMToLLVMIRTranslation
    AIIRMemRefDialect
    AIIRParser
    AIIRPass
    AIIRRegisterAllDialects
    AIIRRegisterAllExtensions
    AIIRRegisterAllPasses
    AIIRSideEffectInterfaces
    AIIRSupport
    AIIRTargetLLVMIRExport
    AIIRTransforms
  )
