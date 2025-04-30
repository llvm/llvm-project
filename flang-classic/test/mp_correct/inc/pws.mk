#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
SRC2 = $(SRC)/src

build: pws.$(OBJX)

run:
	@echo ------------ executing test $@
	-$(RUN2) ./pws.$(EXESUFFIX) $(LOG)

pws.$(OBJX): $(SRC2)/pws.f90 check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC2)/pws.f90
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) pws.$(OBJX) check.$(OBJX) $(LIBS) -o pws.$(EXESUFFIX)

