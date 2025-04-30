#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qnint  ########


qnint: run
	

build:  $(SRC)/qnint.f08
	-$(RM) qnint.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qnint.f08 -o qnint.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qnint.$(OBJX) check.$(OBJX) $(LIBS) -o qnint.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qnint 
	qnint.$(EXESUFFIX)

verify: ;


