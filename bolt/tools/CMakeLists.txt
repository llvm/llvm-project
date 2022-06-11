set(BOLT_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")
mark_as_advanced(BOLT_TOOLS_INSTALL_DIR)

# Move these macros to AddBolt if such a CMake module is ever created.

macro(add_bolt_tool name)
  llvm_add_tool(BOLT ${ARGV})
endmacro()

macro(add_bolt_tool_symlink name)
  llvm_add_tool_symlink(BOLT ${ARGV})
endmacro()

add_subdirectory(driver)
add_subdirectory(llvm-bolt-fuzzer)
add_subdirectory(merge-fdata)
add_subdirectory(heatmap)
