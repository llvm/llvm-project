#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in15  ########


in15: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in15.f90 $(SRC)/in15_expct.c fcheck.$(OBJX)
	-$(RM) in15.$(EXESUFFIX) in15.$(OBJX) in15_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in15_expct.c -o in15_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in15.f90 -o in15.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in15.$(OBJX) in15_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in15.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in15
	in15.$(EXESUFFIX)

verify: ;

in15.run: run

