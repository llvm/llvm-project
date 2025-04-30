#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in25  ########


in25: run
	

fcheck.$(OBJX) check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in25.f90 $(SRC)/in25_expct.c fcheck.$(OBJX) check_mod.mod
	-$(RM) in25.$(EXESUFFIX) in25.$(OBJX) in25_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in25_expct.c -o in25_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in25.f90 -o in25.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in25.$(OBJX) in25_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in25.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in25
	in25.$(EXESUFFIX)

verify: ;

in25.run: run

