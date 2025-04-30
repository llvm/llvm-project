#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka02  ########


ka02: run
	

build:  $(SRC)/ka02.f
	-$(RM) ka02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka02.f -o ka02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka02.$(OBJX) check.$(OBJX) $(LIBS) -o ka02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka02
	ka02.$(EXESUFFIX)

verify: ;

ka02.run: run

