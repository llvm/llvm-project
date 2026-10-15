set(SsafDriverLib
    "__init__.py"
    "driver.py"
    "logging_diagnostics.py"
    "make_orchestrator.py"
    "options.py"
    "patch.py"
    "report.py"
    "source_edits.py"
    "transformations.py")

set(SsafDriverPackageDir "lib/clang_ssaf_driver")

# Copy one file into the build tree and install it, recording the copy so the
# tool's target can depend on it. kind is FILES or PROGRAMS.
function(add_ssaf_driver_file source build_dir destination kind)
  get_filename_component(name "${source}" NAME)
  set(copied "${LLVM_BINARY_DIR}/${build_dir}/${name}")
  add_custom_command(OUTPUT "${copied}"
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       "${LLVM_BINARY_DIR}/${build_dir}"
                     COMMAND ${CMAKE_COMMAND} -E copy
                       "${CMAKE_CURRENT_SOURCE_DIR}/${source}" "${copied}"
                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  set(SSAF_DRIVER_DEPENDS ${SSAF_DRIVER_DEPENDS} "${copied}" PARENT_SCOPE)
  install(${kind} "${source}"
          DESTINATION "${destination}"
          COMPONENT clang-ssaf-driver)
endfunction()

add_ssaf_driver_file("bin/clang-ssaf-driver"
                     "bin" "${CMAKE_INSTALL_BINDIR}" PROGRAMS)

foreach(lib ${SsafDriverLib})
  add_ssaf_driver_file("${SsafDriverPackageDir}/${lib}"
                       "${SsafDriverPackageDir}" "${SsafDriverPackageDir}" FILES)
endforeach()

# Substituted at configure time into this directory's own build area, then
# copied like every other file, so that deleting the package from the build
# tree leaves a rule to put it back.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/${SsafDriverPackageDir}/version.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/version.py")
set(version_py "${LLVM_BINARY_DIR}/${SsafDriverPackageDir}/version.py")
add_custom_command(OUTPUT "${version_py}"
                   COMMAND ${CMAKE_COMMAND} -E make_directory
                     "${LLVM_BINARY_DIR}/${SsafDriverPackageDir}"
                   COMMAND ${CMAKE_COMMAND} -E copy
                     "${CMAKE_CURRENT_BINARY_DIR}/version.py" "${version_py}"
                   DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/version.py")
list(APPEND SSAF_DRIVER_DEPENDS "${version_py}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.py"
        DESTINATION "${SsafDriverPackageDir}"
        COMPONENT clang-ssaf-driver)

add_custom_target(clang-ssaf-driver ALL DEPENDS ${SSAF_DRIVER_DEPENDS})
set_target_properties(clang-ssaf-driver PROPERTIES FOLDER "Clang/Tools")
if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-clang-ssaf-driver"
                           DEPENDS clang-ssaf-driver
                           COMPONENT clang-ssaf-driver)
endif()

# The driver runs the orchestrator installed beside it.
if(CLANG_TOOL_CLANG_SSAF_ORCHESTRATOR_MAKE_BUILD)
  add_dependencies(clang-ssaf-driver clang-ssaf-orchestrator-make)
  if(NOT LLVM_ENABLE_IDE)
    add_dependencies(install-clang-ssaf-driver
                     install-clang-ssaf-orchestrator-make)
    add_dependencies(install-clang-ssaf-driver-stripped
                     install-clang-ssaf-orchestrator-make-stripped)
  endif()
endif()
