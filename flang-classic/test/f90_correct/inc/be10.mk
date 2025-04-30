#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test be10  ########


be10: run
	

build:  $(SRC)/be10.f
	-$(RM) be10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/be10.f -o be10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) be10.$(OBJX) check.$(OBJX) $(LIBS) -o be10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test be10
	be10.$(EXESUFFIX)

verify: ;

be10.run: run

