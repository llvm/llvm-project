#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
SRC2 = $(SRC)/src

build: tp08.$(OBJX)

run:
	@echo ------------ executing test $@
	-$(RUN2) ./tp08.$(EXESUFFIX) $(LOG)

tp08.$(OBJX): $(SRC2)/tp08.f90 check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC2)/tp08.f90
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) tp08.$(OBJX) check.$(OBJX) $(LIBS) -o tp08.$(EXESUFFIX)

