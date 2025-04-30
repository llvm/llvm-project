#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in21  ########


in21: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in21.f90 $(SRC)/in21_expct.c fcheck.$(OBJX)
	-$(RM) in21.$(EXESUFFIX) in21.$(OBJX) in21_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in21_expct.c -o in21_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in21.f90 -o in21.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in21.$(OBJX) in21_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in21.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in21
	in21.$(EXESUFFIX)

verify: ;

in21.run: run

