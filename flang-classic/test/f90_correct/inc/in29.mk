#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in29  ########


in29: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in29.f90 $(SRC)/in29_expct.c fcheck.$(OBJX)
	-$(RM) in29.$(EXESUFFIX) in29.$(OBJX) in29_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in29_expct.c -o in29_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in29.f90 -o in29.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in29.$(OBJX) in29_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in29.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in29
	in29.$(EXESUFFIX)

verify: ;

in29.run: run

