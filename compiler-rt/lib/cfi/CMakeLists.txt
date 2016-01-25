add_custom_target(cfi)

set(CFI_SOURCES cfi.cc)

include_directories(..)

set(CFI_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
)

set(CFI_DIAG_CFLAGS
  -DCFI_ENABLE_DIAG=1
)

foreach(arch ${CFI_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.cfi
    STATIC
    ARCHS ${arch}
    SOURCES ${CFI_SOURCES}
    OBJECT_LIBS RTInterception
                RTSanitizerCommon
                RTSanitizerCommonLibc
    CFLAGS ${CFI_CFLAGS}
    PARENT_TARGET cfi)
  add_compiler_rt_runtime(clang_rt.cfi_diag
    STATIC
    ARCHS ${arch}
    SOURCES ${CFI_SOURCES}
    OBJECT_LIBS RTInterception
                RTSanitizerCommon
                RTSanitizerCommonLibc
		RTUbsan
    CFLAGS ${CFI_CFLAGS} ${CFI_DIAG_CFLAGS}
    PARENT_TARGET cfi)
endforeach()

add_compiler_rt_resource_file(cfi_blacklist cfi_blacklist.txt)
add_dependencies(cfi cfi_blacklist)
add_dependencies(compiler-rt cfi)
