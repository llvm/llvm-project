#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in19  ########


in19: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in19.f90 $(SRC)/in19_expct.c fcheck.$(OBJX)
	-$(RM) in19.$(EXESUFFIX) in19.$(OBJX) in19_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in19_expct.c -o in19_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in19.f90 -o in19.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in19.$(OBJX) in19_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in19.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in19
	in19.$(EXESUFFIX)

verify: ;

in19.run: run

