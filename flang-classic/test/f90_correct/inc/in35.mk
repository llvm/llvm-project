#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in35  ########


in35: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in35.f90 fcheck.$(OBJX)
	-$(RM) in35.$(EXESUFFIX) in35.$(OBJX) 
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in35.f90 -o in35.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in35.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in35.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in35
	in35.$(EXESUFFIX)

verify: ;

in35.run: run

