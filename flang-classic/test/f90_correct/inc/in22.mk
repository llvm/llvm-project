#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in22  ########


in22: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in22.f90 $(SRC)/in22_expct.c fcheck.$(OBJX)
	-$(RM) in22.$(EXESUFFIX) in22.$(OBJX) in22_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in22_expct.c -o in22_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in22.f90 -o in22.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in22.$(OBJX) in22_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in22.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in22
	in22.$(EXESUFFIX)

verify: ;

in22.run: run

