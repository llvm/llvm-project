#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in17  ########


in17: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in17.f90 $(SRC)/in17_expct.c fcheck.$(OBJX)
	-$(RM) in17.$(EXESUFFIX) in17.$(OBJX) in17_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in17_expct.c -o in17_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in17.f90 -o in17.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in17.$(OBJX) in17_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in17.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in17
	in17.$(EXESUFFIX)

verify: ;

in17.run: run

