set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  MC
  Object
  Support
  )

add_bolt_tool(llvm-bolt-heatmap
  heatmap.cpp
  )

target_link_libraries(llvm-bolt-heatmap
  PRIVATE
  LLVMBOLTProfile
  LLVMBOLTRewrite
  LLVMBOLTUtils
  )

set_target_properties(llvm-bolt-heatmap PROPERTIES FOLDER "BOLT")
