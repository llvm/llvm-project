#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test dcmplxtoq ########


qp74: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp74.f08 fcheck.$(OBJX)
	-$(RM) qp74.$(EXESUFFIX) qp74.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp74.f08 -o qp74.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp74.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp74.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp74
	qp74.$(EXESUFFIX)

verify: ;

qp74.run: run

