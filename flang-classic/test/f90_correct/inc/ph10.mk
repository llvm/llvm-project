#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ph10  ########


ph10: run
	

build:  $(SRC)/ph10.f
	-$(RM) ph10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ph10.f -o ph10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ph10.$(OBJX) check.$(OBJX) $(LIBS) -o ph10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ph10
	ph10.$(EXESUFFIX)

verify: ;

ph10.run: run

