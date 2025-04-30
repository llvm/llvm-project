#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test chartoq  ########


qp00: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp00.f08 fcheck.$(OBJX)
	-$(RM) qp00.$(EXESUFFIX) qp00.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp00.f08 -o qp00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp00.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp00
	qp00.$(EXESUFFIX)

verify: ;

qp00.run: run

