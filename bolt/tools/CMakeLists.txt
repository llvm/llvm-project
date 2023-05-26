set(BOLT_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")
mark_as_advanced(BOLT_TOOLS_INSTALL_DIR)

# Move these macros to AddBolt if such a CMake module is ever created.

macro(add_bolt_tool name)
  llvm_add_tool(BOLT ${ARGV})
  install(TARGETS ${name}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${name}
    )
  add_llvm_install_targets(install-${name} DEPENDS ${name} COMPONENT ${name})
endmacro()

macro(add_bolt_tool_symlink name dest)
  llvm_add_tool_symlink(BOLT ${name} ${dest})
  llvm_install_symlink(BOLT ${name} ${dest} ALWAYS_GENERATE)
endmacro()

add_subdirectory(driver)
add_subdirectory(llvm-bolt-fuzzer)
add_subdirectory(bat-dump)
add_subdirectory(merge-fdata)
add_subdirectory(heatmap)
