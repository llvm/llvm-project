# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

SRC2 = $(SRC)/src

build: fs25053.$(OBJX)

run:
	@echo ------------ executing test $@
	-$(RUN2) ./fs25053.$(EXESUFFIX)

fs25053.$(OBJX): $(SRC2)/fs25053.f90
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC2)/fs25053.f90 -c
	-$(FC) $(LDFLAGS) fs25053.$(OBJX) -o fs25053.$(EXESUFFIX)

