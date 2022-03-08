set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BOLTProfile
  BOLTRewrite
  BOLTUtils
  MC
  Object
  Support
  )

if (BOLT_ENABLE_RUNTIME)
  set(BOLT_DRIVER_DEPS "bolt_rt")
else()
  set(BOLT_DRIVER_DEPS "")
endif()

add_llvm_tool(llvm-bolt
  llvm-bolt.cpp

  DEPENDS
  ${BOLT_DRIVER_DEPS}
  )

add_llvm_tool_symlink(perf2bolt llvm-bolt)
add_llvm_tool_symlink(llvm-boltdiff llvm-bolt)

set(BOLT_DEPENDS
  llvm-bolt
  perf2bolt
  llvm-boltdiff
  )

add_custom_target(bolt DEPENDS ${BOLT_DEPENDS})
install(PROGRAMS
  ${CMAKE_BINARY_DIR}/bin/llvm-bolt
  ${CMAKE_BINARY_DIR}/bin/perf2bolt
  ${CMAKE_BINARY_DIR}/bin/llvm-boltdiff
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT bolt
  )
add_llvm_install_targets(install-bolt DEPENDS bolt COMPONENT bolt)
set_target_properties(bolt PROPERTIES FOLDER "BOLT")
set_target_properties(install-bolt PROPERTIES FOLDER "BOLT")

include_directories( ${BOLT_SOURCE_DIR}/lib )
