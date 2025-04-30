#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in30  ########


in30: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in30.f90 $(SRC)/in30_expct.c fcheck.$(OBJX)
	-$(RM) in30.$(EXESUFFIX) in30.$(OBJX) in30_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in30_expct.c -o in30_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in30.f90 -o in30.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in30.$(OBJX) in30_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in30.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test in30
	in30.$(EXESUFFIX)

verify: ;

in30.run: run

