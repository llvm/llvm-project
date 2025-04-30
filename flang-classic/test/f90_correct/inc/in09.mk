#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in09  ########


in09: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in09.f90 $(SRC)/in09_expct.c fcheck.$(OBJX)
	-$(RM) in09.$(EXESUFFIX) in09.$(OBJX) in09_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in09_expct.c -o in09_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in09.f90 -o in09.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in09.$(OBJX) in09_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in09.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in09
	in09.$(EXESUFFIX)

verify: ;

in09.run: run

