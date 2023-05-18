set(BOLT_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")
mark_as_advanced(BOLT_TOOLS_INSTALL_DIR)

add_subdirectory(driver)
add_subdirectory(llvm-bolt-fuzzer)
add_subdirectory(bat-dump)
add_subdirectory(merge-fdata)
add_subdirectory(heatmap)
