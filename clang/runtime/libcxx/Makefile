##===- clang/runtime/libcxx/Makefile -----------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
#
# This file defines support for installing a copy of the libcxx headers where
# the driver expects them.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..
include $(CLANG_LEVEL)/Makefile

PROJ_libcxx_hdrs := $(DESTDIR)$(PROJ_prefix)/lib

# Expect libcxx to be in llvm/projects/libcxx
LIBCXX_SRC_ROOT := $(LLVM_SRC_ROOT)/projects/libcxx

ifneq ($(CLANG_NO_RUNTIME),1)
ifeq ($(shell test -d $(LIBCXX_SRC_ROOT) && echo OK),OK)

install-local::
	$(MAKE) -C $(LIBCXX_SRC_ROOT) \
	  HEADER_DIR=$(PROJ_libcxx_hdrs) installheaders

endif
endif
