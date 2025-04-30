#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmul_misc  ########


mmul_misc: run
	

build:  $(SRC)/mmul_misc.f90
	-$(RM) mmul_misc.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmul_misc.f90 -o mmul_misc.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmul_misc.$(OBJX) check.$(OBJX) $(LIBS) -o mmul_misc.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmul_misc
	mmul_misc.$(EXESUFFIX)

verify: ;

mmul_misc.run: run

