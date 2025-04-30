#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in10  ########


in10: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in10.f90 $(SRC)/in10_expct.c fcheck.$(OBJX)
	-$(RM) in10.$(EXESUFFIX) in10.$(OBJX) in10_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in10_expct.c -o in10_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in10.f90 -o in10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in10.$(OBJX) in10_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in10
	in10.$(EXESUFFIX)

verify: ;

in10.run: run

