#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtolog  ########


qp65: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp65.f08 fcheck.$(OBJX)
	-$(RM) qp65.$(EXESUFFIX) qp65.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp65.f08 -o qp65.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp65.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp65.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp65
	qp65.$(EXESUFFIX)

verify: ;

qp65.run: run

