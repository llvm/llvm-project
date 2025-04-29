set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Object
  Support
  )

add_bolt_tool(bolt-linux-instr
  bolt-linux-instr.cpp
  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

add_dependencies(bolt bolt-linux-instr)
