file(GLOB files ${CPACK_TEMPORARY_INSTALL_DIRECTORY}/lib/*.a)

foreach(file ${files})
  execute_process(COMMAND ${CPACK_TEMPORARY_INSTALL_DIRECTORY}/bin/llvm-strip --no-strip-all -R .llvm.lto ${file})
endforeach()
