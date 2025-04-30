# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

SRC2 = $(SRC)/src

build: nv2044769.$(OBJX)

run:
	@echo ------------ executing test $@
	-$(RUN2) ./nv2044769.$(EXESUFFIX)

nv2044769.$(OBJX): $(SRC2)/nv2044769.f90
	@echo ------------ building test $@
	-$(FC) -cpp $(CFLAGS) $(SRC2)/nv2044769.f90 -c
	-$(FC) $(LDFLAGS) nv2044769.$(OBJX) -o nv2044769.$(EXESUFFIX)

