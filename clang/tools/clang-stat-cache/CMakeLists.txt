set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_tool(clang-stat-cache
  clang-stat-cache.cpp
  )

if(APPLE)
set(CLANG_STAT_CACHE_LIB_DEPS
  "-framework CoreServices"
  )
endif()

clang_target_link_libraries(clang-stat-cache
  PRIVATE
  ${CLANG_STAT_CACHE_LIB_DEPS}
  )
