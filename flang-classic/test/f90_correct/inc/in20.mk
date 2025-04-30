#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in20  ########


in20: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in20.f90 $(SRC)/in20_expct.c fcheck.$(OBJX)
	-$(RM) in20.$(EXESUFFIX) in20.$(OBJX) in20_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in20_expct.c -o in20_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in20.f90 -o in20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in20.$(OBJX) in20_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in20
	in20.$(EXESUFFIX)

verify: ;

in20.run: run

