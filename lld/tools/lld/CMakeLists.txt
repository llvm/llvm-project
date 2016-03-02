add_llvm_executable(lld
  lld.cpp
  )

target_link_libraries(lld
  lldDriver
  lldCOFF
  lldELF
  LLVMSupport
  )

install(TARGETS lld
  RUNTIME DESTINATION bin)

if(NOT LLD_SYMLINKS_TO_CREATE)
  set(LLD_SYMLINKS_TO_CREATE lld-link ld.lld)
endif()

foreach(link ${LLD_SYMLINKS_TO_CREATE})
  add_llvm_tool_symlink(${link} lld ALWAYS_GENERATE)
  # Always generate install targets
  llvm_install_symlink(${link} lld ALWAYS_GENERATE)
endforeach()

