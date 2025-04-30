#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in18  ########


in18: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in18.f90 $(SRC)/in18_expct.c fcheck.$(OBJX)
	-$(RM) in18.$(EXESUFFIX) in18.$(OBJX) in18_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in18_expct.c -o in18_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in18.f90 -o in18.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in18.$(OBJX) in18_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in18.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in18
	in18.$(EXESUFFIX)

verify: ;

in18.run: run

