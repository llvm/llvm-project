include(GNUInstallDirs)
include(LLVMDistributionSupport)

macro(add_bolt_executable name)
  add_llvm_executable(${name} ${ARGN})
endmacro()

macro(add_bolt_tool name)
  if (NOT BOLT_BUILD_TOOLS)
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_bolt_executable(${name} ${ARGN})

  if (BOLT_BUILD_TOOLS)
    get_target_export_arg(${name} BOLT export_to_bolttargets)
    install(TARGETS ${name}
      ${export_to_bolttargets}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT bolt)

    if(NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${name}
                               DEPENDS ${name}
                               COMPONENT bolt)
    endif()
    set_property(GLOBAL APPEND PROPERTY BOLT_EXPORTS ${name})
  endif()
endmacro()

macro(add_bolt_tool_symlink name dest)
  llvm_add_tool_symlink(BOLT ${name} ${dest} ALWAYS_GENERATE)
  # Always generate install targets
  llvm_install_symlink(BOLT ${name} ${dest} ALWAYS_GENERATE COMPONENT bolt)
endmacro()
