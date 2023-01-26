set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_lld_tool(lld
  lld.cpp

  SUPPORT_PLUGINS
  GENERATE_DRIVER
  )
export_executable_symbols_for_plugins(lld)

function(lld_target_link_libraries target type)
  target_link_libraries(${target} ${type} ${ARGN})
  if (TARGET obj.${target})
    target_link_libraries(obj.${target} ${ARGN})
  endif()
endfunction()

lld_target_link_libraries(lld
  PRIVATE
  lldCommon
  lldCOFF
  lldELF
  lldMachO
  lldMinGW
  lldWasm
  )

install(TARGETS lld
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

if(NOT LLD_SYMLINKS_TO_CREATE)
  set(LLD_SYMLINKS_TO_CREATE
      lld-link ld.lld ld64.lld wasm-ld)
endif()

foreach(link ${LLD_SYMLINKS_TO_CREATE})
  add_lld_symlink(${link} lld)
endforeach()
