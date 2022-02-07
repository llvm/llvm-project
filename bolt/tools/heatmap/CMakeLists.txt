set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BOLTProfile
  BOLTRewrite
  BOLTUtils
  MC
  Object
  Support
  )

add_llvm_tool(llvm-bolt-heatmap
  heatmap.cpp
  )

set_target_properties(llvm-bolt-heatmap PROPERTIES FOLDER "BOLT")
