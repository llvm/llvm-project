#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in32  ########


in32: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in32.f90 $(SRC)/in32_expct.c fcheck.$(OBJX)
	-$(RM) in32.$(EXESUFFIX) in32.$(OBJX) in32_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in32_expct.c -o in32_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in32.f90 -o in32.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in32.$(OBJX) in32_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in32.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in32
	in32.$(EXESUFFIX)

verify: ;

in32.run: run

