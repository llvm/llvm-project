#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in12  ########


in12: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in12.f90 $(SRC)/in12_expct.c fcheck.$(OBJX)
	-$(RM) in12.$(EXESUFFIX) in12.$(OBJX) in12_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in12_expct.c -o in12_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in12.f90 -o in12.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in12.$(OBJX) in12_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in12.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in12
	in12.$(EXESUFFIX)

verify: ;

in12.run: run

