#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test PARAMETER  ########


qp45_d: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp45_d.f08 fcheck.$(OBJX)
	-$(RM) qp45_d.$(EXESUFFIX) qp45_d.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp45_d.f08 -o qp45_d.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp45_d.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp45_d.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp45_d
	qp45_d.$(EXESUFFIX)

verify: ;

qp45_d.run: run

