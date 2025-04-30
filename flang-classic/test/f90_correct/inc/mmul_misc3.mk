#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmul_misc3  ########


mmul_misc3: run


build:  $(SRC)/mmul_misc3.f90
	-$(RM) mmul_misc3.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmul_misc3.f90 -o mmul_misc3.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmul_misc3.$(OBJX) check.$(OBJX) $(LIBS) -o mmul_misc3.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmul_misc3
	mmul_misc3.$(EXESUFFIX)

verify: ;

mmul_misc3.run: run
