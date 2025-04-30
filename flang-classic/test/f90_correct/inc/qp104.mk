#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test cmplxtoq  ########


qp104: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp104.f08 fcheck.$(OBJX)
	-$(RM) qp104.$(EXESUFFIX) qp104.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp104.f08 -o qp104.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp104.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp104.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp104
	qp104.$(EXESUFFIX)

verify: ;

qp104.run: run

