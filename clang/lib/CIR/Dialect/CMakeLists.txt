# IR must be processed first: it defines MLIRCIR, whose PCH other CIR
# dialect libraries below (Analysis, OpenACC, OpenMP, Transforms) can
# automatically reuse via llvm_update_pch's LLVM_PCH_PRIORITY dependency
# scan -- that scan only sees targets that already exist, so MLIRCIR must
# be defined before anything that wants to reuse its PCH.
add_subdirectory(IR)
add_subdirectory(Analysis)
add_subdirectory(OpenACC)
add_subdirectory(OpenMP)
add_subdirectory(Transforms)
