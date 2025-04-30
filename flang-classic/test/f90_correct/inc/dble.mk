#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test dble function take quadruple precision  ########


dble: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/dble.f08 fcheck.$(OBJX)
	-$(RM) dble.$(EXESUFFIX) dble.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dble.f08 -o dble.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dble.$(OBJX) fcheck.$(OBJX) $(LIBS) -o dble.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dble
	dble.$(EXESUFFIX)

verify: ;

dble.run: run

