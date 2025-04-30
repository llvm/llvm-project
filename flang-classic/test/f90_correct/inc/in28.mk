#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in28  ########


in28: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in28.f90 $(SRC)/in28_expct.c fcheck.$(OBJX)
	-$(RM) in28.$(EXESUFFIX) in28.$(OBJX) in28_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in28_expct.c -o in28_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in28.f90 -o in28.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in28.$(OBJX) in28_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in28.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in28
	in28.$(EXESUFFIX)

verify: ;

in28.run: run

