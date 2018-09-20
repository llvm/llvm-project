##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../irif/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../oclc/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

opencl_bc_lib(NAME ockl SOURCES ${sources})

install(FILES
        inc/amd_hsa_common.h
        inc/amd_hsa_elf.h
        inc/amd_hsa_kernel_code.h
        inc/amd_hsa_queue.h
        inc/amd_hsa_signal.h
        inc/as.h
        inc/device_amd_hsa.h
        inc/hsa.h
        inc/ockl_hsa.h
        inc/ockl.h
        DESTINATION include COMPONENT OpenCL)
