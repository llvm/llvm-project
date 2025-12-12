include_directories(..)

# Runtime library sources and build flags.
set(SIGSAN_SOURCES
  sigsan.cpp
)

add_compiler_rt_component(sigsan)
add_compiler_rt_runtime(clang_rt.sigsan
    STATIC
    ARCHS x86_64
    SOURCES ${SIGSAN_SOURCES}
    OBJECT_LIBS RTInterception
                RTSanitizerCommon
                RTSanitizerCommonLibc
                RTSanitizerCommonSymbolizer
    PARENT_TARGET sigsan
)
