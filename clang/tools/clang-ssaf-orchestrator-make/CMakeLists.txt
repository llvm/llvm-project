set(SsafOrchestratorMakeLib
    "__init__.py"
    "build_graph.py"
    "config.py"
    "config_orchestrator.py"
    "config_shims.py"
    "failure_logs.py"
    "interposition.py"
    "logging_diagnostics.py"
    "options.py"
    "orchestrator.py"
    "ssaf_artifact_repository.py"
    "utils.py")

set(SsafOrchestratorMakeShimLib
    "__init__.py"
    "ar.py"
    "clang.py"
    "commands.py"
    "driver_jobs.py"
    "ld.py"
    "run.py"
    "shim.py")

set(SsafOrchestratorMakePackageDir "lib/clang_ssaf_orchestrator_make")
set(SsafOrchestratorMakeShimDir "libexec/ssaf-orchestrator-shims")

# Copy one file into the build tree and install it, recording the copy so the
# tool's target can depend on it. kind is FILES or PROGRAMS.
function(add_ssaf_orchestrator_make_file source build_dir destination kind)
  get_filename_component(name "${source}" NAME)
  set(copied "${LLVM_BINARY_DIR}/${build_dir}/${name}")
  add_custom_command(OUTPUT "${copied}"
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       "${LLVM_BINARY_DIR}/${build_dir}"
                     COMMAND ${CMAKE_COMMAND} -E copy
                       "${CMAKE_CURRENT_SOURCE_DIR}/${source}" "${copied}"
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  set(SSAF_ORCHESTRATOR_MAKE_DEPENDS
      ${SSAF_ORCHESTRATOR_MAKE_DEPENDS} "${copied}" PARENT_SCOPE)
  install(${kind} "${source}"
          DESTINATION "${destination}"
          COMPONENT clang-ssaf-orchestrator-make)
endfunction()

add_ssaf_orchestrator_make_file("bin/clang-ssaf-orchestrator-make"
                                "bin" "${CMAKE_INSTALL_BINDIR}" PROGRAMS)

foreach(lib ${SsafOrchestratorMakeLib})
  add_ssaf_orchestrator_make_file(
    "${SsafOrchestratorMakePackageDir}/${lib}"
    "${SsafOrchestratorMakePackageDir}"
    "${SsafOrchestratorMakePackageDir}" FILES)
endforeach()

# Substituted at configure time into this directory's own build area, then
# copied like every other file, so that deleting the package from the build
# tree leaves a rule to put it back.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/${SsafOrchestratorMakePackageDir}/version.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/version.py")
set(version_py "${LLVM_BINARY_DIR}/${SsafOrchestratorMakePackageDir}/version.py")
add_custom_command(OUTPUT "${version_py}"
                   COMMAND ${CMAKE_COMMAND} -E make_directory
                     "${LLVM_BINARY_DIR}/${SsafOrchestratorMakePackageDir}"
                   COMMAND ${CMAKE_COMMAND} -E copy
                     "${CMAKE_CURRENT_BINARY_DIR}/version.py" "${version_py}"
                   DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/version.py")
list(APPEND SSAF_ORCHESTRATOR_MAKE_DEPENDS "${version_py}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.py"
        DESTINATION "${SsafOrchestratorMakePackageDir}"
        COMPONENT clang-ssaf-orchestrator-make)

foreach(lib ${SsafOrchestratorMakeShimLib})
  add_ssaf_orchestrator_make_file(
    "${SsafOrchestratorMakePackageDir}/shims/${lib}"
    "${SsafOrchestratorMakePackageDir}/shims"
    "${SsafOrchestratorMakePackageDir}/shims" FILES)
endforeach()

# One program, symlinked under every name a build might spell.
add_ssaf_orchestrator_make_file(
  "shims/shim"
  "${SsafOrchestratorMakeShimDir}"
  "${CMAKE_INSTALL_LIBEXECDIR}/ssaf-orchestrator-shims" PROGRAMS)

add_custom_target(clang-ssaf-orchestrator-make ALL
                  DEPENDS ${SSAF_ORCHESTRATOR_MAKE_DEPENDS})
set_target_properties(clang-ssaf-orchestrator-make
                      PROPERTIES FOLDER "Clang/Tools")
if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-clang-ssaf-orchestrator-make"
                           DEPENDS clang-ssaf-orchestrator-make
                           COMPONENT clang-ssaf-orchestrator-make)
endif()
