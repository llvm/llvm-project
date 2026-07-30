set(_llvm_runtime_compiler_dependency
  "${CMAKE_CURRENT_LIST_DIR}/../../llvm/cmake/modules/LLVMRuntimeCompilerDependency.cmake")

if(EXISTS "${_llvm_runtime_compiler_dependency}")
  include("${_llvm_runtime_compiler_dependency}")
elseif(LLVM_CMAKE_DIR AND EXISTS "${LLVM_CMAKE_DIR}/LLVMRuntimeCompilerDependency.cmake")
  include("${LLVM_CMAKE_DIR}/LLVMRuntimeCompilerDependency.cmake")
elseif(LLVM_DIR AND EXISTS "${LLVM_DIR}/LLVMRuntimeCompilerDependency.cmake")
  include("${LLVM_DIR}/LLVMRuntimeCompilerDependency.cmake")
else()
  message(FATAL_ERROR "Could not find LLVMRuntimeCompilerDependency.cmake")
endif()

unset(_llvm_runtime_compiler_dependency)
