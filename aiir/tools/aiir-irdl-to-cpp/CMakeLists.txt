add_llvm_executable(aiir-irdl-to-cpp
    aiir-irdl-to-cpp.cpp
  )
aiir_target_link_libraries(aiir-irdl-to-cpp
  PRIVATE
  AIIRTargetIRDLToCpp
  )

setup_host_tool(aiir-irdl-to-cpp AIIR_IRDL_TO_CPP AIIR_IRDL_TO_CPP_EXE AIIR_IRDL_TO_CPP_TARGET)
