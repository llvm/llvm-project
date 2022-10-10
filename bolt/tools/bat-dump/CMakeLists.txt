set(LLVM_LINK_COMPONENTS
  Object
  Support
  )

add_llvm_tool(llvm-bat-dump
  bat-dump.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

target_link_libraries(llvm-bat-dump
  PRIVATE
  LLVMBOLTProfile
  )

set_target_properties(llvm-bat-dump PROPERTIES FOLDER "BOLT")
