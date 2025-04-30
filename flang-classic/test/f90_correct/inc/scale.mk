#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test scale function take quadruple precision  ########


scale: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/scale.f08 fcheck.$(OBJX)
	-$(RM) scale.$(EXESUFFIX) scale.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/scale.f08 -o scale.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) scale.$(OBJX) fcheck.$(OBJX) $(LIBS) -o scale.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test scale
	scale.$(EXESUFFIX)

verify: ;

scale.run: run

