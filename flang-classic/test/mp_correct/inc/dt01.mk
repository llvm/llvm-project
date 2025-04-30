#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
SRC2 = $(SRC)/src

build: dt01.$(OBJX)

run:
	@echo ------------ executing test $@
	-$(RUN2) ./dt01.$(EXESUFFIX) $(LOG)

verify: ;

dt01.$(OBJX): $(SRC2)/dt01.f90 check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC2)/dt01.f90
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) dt01.$(OBJX) check.$(OBJX) $(LIBS) -o dt01.$(EXESUFFIX)

