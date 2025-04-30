#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test dtoq  ########


qp75: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp75.f08 fcheck.$(OBJX)
	-$(RM) qp75.$(EXESUFFIX) qp75.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp75.f08 -o qp75.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp75.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp75.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp75
	qp75.$(EXESUFFIX)

verify: ;

qp75.run: run

