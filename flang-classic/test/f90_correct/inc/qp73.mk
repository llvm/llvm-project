#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test cmplxtoq  ########


qp73: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp73.f08 fcheck.$(OBJX)
	-$(RM) qp73.$(EXESUFFIX) qp73.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp73.f08 -o qp73.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp73.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp73.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp73
	qp73.$(EXESUFFIX)

verify: ;

qp73.run: run

