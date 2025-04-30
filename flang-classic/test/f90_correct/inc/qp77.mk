#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test inttoq########


qp77: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp77.f08 fcheck.$(OBJX)
	-$(RM) qp77.$(EXESUFFIX) qp77.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp77.f08 -o qp77.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp77.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp77.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp77
	qp77.$(EXESUFFIX)

verify: ;

qp77.run: run

