#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test rrspacing function take quadruple precision  ########


qrrspacing: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qrrspacing.f08 fcheck.$(OBJX)
	-$(RM) qrrspacing.$(EXESUFFIX) qrrspacing.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qrrspacing.f08 -o qrrspacing.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qrrspacing.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qrrspacing.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qrrspacing
	qrrspacing.$(EXESUFFIX)

verify: ;

qrrspacing.run: run

