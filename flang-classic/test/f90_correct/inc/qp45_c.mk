#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test PARAMETER  ########


qp45_c: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp45_c.f08 fcheck.$(OBJX)
	-$(RM) qp45_c.$(EXESUFFIX) qp45_c.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp45_c.f08 -o qp45_c.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp45_c.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp45_c.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp45_c
	qp45_c.$(EXESUFFIX)

verify: ;

qp45_c.run: run

