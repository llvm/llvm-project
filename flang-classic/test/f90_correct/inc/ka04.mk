#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka04  ########


ka04: run
	

build:  $(SRC)/ka04.f
	-$(RM) ka04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka04.f -o ka04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka04.$(OBJX) check.$(OBJX) $(LIBS) -o ka04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka04
	ka04.$(EXESUFFIX)

verify: ;

ka04.run: run

