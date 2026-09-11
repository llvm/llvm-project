# get_subproject_title(titlevar)
#   Set ${outvar} to the title of the current LLVM subproject (Clang, MLIR ...)
#
# The title is set in the subproject's top-level using the variable
# LLVM_SUBPROJECT_TITLE. If it does not exist, it is assumed it is LLVM itself.
# The title is not semantically significant, but use to create folders in
# CMake-generated IDE projects (Visual Studio/XCode).
function(get_subproject_title outvar)
  if (LLVM_SUBPROJECT_TITLE)
    set(${outvar} "${LLVM_SUBPROJECT_TITLE}" PARENT_SCOPE)
  else ()
    set(${outvar} "LLVM" PARENT_SCOPE)
  endif ()
endfunction(get_subproject_title)
