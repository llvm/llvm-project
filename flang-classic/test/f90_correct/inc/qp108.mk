#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test int8toq  ########


qp108: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp108.f08 fcheck.$(OBJX)
	-$(RM) qp108.$(EXESUFFIX) qp108.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp108.f08 -o qp108.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp108.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp108.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp108
	qp108.$(EXESUFFIX)

verify: ;

qp108.run: run

