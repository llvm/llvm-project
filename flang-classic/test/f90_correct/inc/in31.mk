#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in31  ########


in31: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in31.f90 $(SRC)/in31_expct.c fcheck.$(OBJX)
	-$(RM) in31.$(EXESUFFIX) in31.$(OBJX) in31_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in31_expct.c -o in31_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in31.f90 -o in31.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in31.$(OBJX) in31_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in31.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in31
	in31.$(EXESUFFIX)

verify: ;

in31.run: run

