#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test in27  ########


in27: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in27.f90 $(SRC)/in27_expct.c fcheck.$(OBJX)
	-$(RM) in27.$(EXESUFFIX) in27.$(OBJX) in27_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in27_expct.c -o in27_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in27.f90 -o in27.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in27.$(OBJX) in27_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in27.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in27
	in27.$(EXESUFFIX)

verify: ;

in27.run: run

