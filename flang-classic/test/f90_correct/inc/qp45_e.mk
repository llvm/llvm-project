#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test PARAMETER  ########


qp45_e: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp45_e.f08 fcheck.$(OBJX)
	-$(RM) qp45_e.$(EXESUFFIX) qp45_e.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp45_e.f08 -o qp45_e.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp45_e.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp45_e.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp45_e
	qp45_e.$(EXESUFFIX)

verify: ;

qp45_e.run: run

