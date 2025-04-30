#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test be11  ########


be11: run
	

build:  $(SRC)/be11.f90
	-$(RM) be11.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/be11.f90 -o be11.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) be11.$(OBJX) check.$(OBJX) $(LIBS) -o be11.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test be11
	be11.$(EXESUFFIX)

verify: ;

be11.run: run

