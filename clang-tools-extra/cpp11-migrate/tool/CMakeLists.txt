set(LLVM_LINK_COMPONENTS support)

set (Cpp11MigrateSources
  Cpp11Migrate.cpp
  )

# FIXME: Lib-ify the transforms to simplify the build rules.

# For each transform subdirectory.
file(GLOB_RECURSE LoopConvertSources "../LoopConvert/*.cpp")
list(APPEND Cpp11MigrateSources ${LoopConvertSources})

file(GLOB_RECURSE UseNullptrSources "../UseNullptr/*.cpp")
list(APPEND Cpp11MigrateSources ${UseNullptrSources})

file(GLOB_RECURSE UseAutoSources "../UseAuto/*.cpp")
list(APPEND Cpp11MigrateSources ${UseAutoSources})

file(GLOB_RECURSE AddOverrideSources "../AddOverride/*.cpp")
list(APPEND Cpp11MigrateSources ${AddOverrideSources})

add_clang_executable(cpp11-migrate
  ${Cpp11MigrateSources}
  )

add_dependencies(cpp11-migrate
  clang-headers
  )

target_link_libraries(cpp11-migrate
  migrateCore
  )
