#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test set_exponent function take quadruple precision  ########


set_exponent: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/set_exponent.f08 fcheck.$(OBJX)
	-$(RM) set_exponent.$(EXESUFFIX) set_exponent.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/set_exponent.f08 -o set_exponent.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) set_exponent.$(OBJX) fcheck.$(OBJX) $(LIBS) -o set_exponent.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test set_exponent
	set_exponent.$(EXESUFFIX)

verify: ;

set_exponent.run: run

