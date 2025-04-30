#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test cmplxtoq  ########


qp51: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp51.f08 fcheck.$(OBJX)
	-$(RM) qp51.$(EXESUFFIX) qp51.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp51.f08 -o qp51.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp51.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp51.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp51
	qp51.$(EXESUFFIX)

verify: ;

qp51.run: run

