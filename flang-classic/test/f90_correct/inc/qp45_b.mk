#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test PARAMETER  ########


qp45_b: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp45_b.f08 fcheck.$(OBJX)
	-$(RM) qp45_b.$(EXESUFFIX) qp45_b.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp45_b.f08 -o qp45_b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp45_b.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp45_b.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp45_b
	qp45_b.$(EXESUFFIX)

verify: ;

qp45_b.run: run

