#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(ARCH_DEP_ILI_DIR  ${UTILS_ILI_DIR}/${TARGET_ARCHITECTURE})
set(ARCH_DEP_ILM_DIR  ${UTILS_ILM_DIR}/${TARGET_ARCHITECTURE})

add_subdirectory(ilitp)
add_subdirectory(ilmtp)
add_subdirectory(symtab)
add_subdirectory(machar)
add_subdirectory(upper)
add_subdirectory(n2rst)

# Local Variables:
# mode: cmake
# End:
