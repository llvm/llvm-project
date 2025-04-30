#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in24  ########


in24: run
	

fcheck.$(OBJX) check_mod.mod: $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in24.f90 $(SRC)/in24_expct.c fcheck.$(OBJX) check_mod.mod
	-$(RM) in24.$(EXESUFFIX) in24.$(OBJX) in24_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in24_expct.c -o in24_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in24.f90 -o in24.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in24.$(OBJX) in24_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in24.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in24
	in24.$(EXESUFFIX)

verify: ;

in24.run: run

