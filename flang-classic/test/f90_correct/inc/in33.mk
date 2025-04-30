#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in33  ########


in33: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in33.f90 fcheck.$(OBJX)
	-$(RM) in33.$(EXESUFFIX) in33.$(OBJX) 
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in33.f90 -o in33.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in33.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in33.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in33
	in33.$(EXESUFFIX)

verify: ;

in33.run: run

