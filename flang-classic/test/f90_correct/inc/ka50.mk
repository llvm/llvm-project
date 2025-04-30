#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka50  ########


ka50: run
	

build:  $(SRC)/ka50.f
	-$(RM) ka50.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka50.f -o ka50.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka50.$(OBJX) check.$(OBJX) $(LIBS) -o ka50.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka50
	ka50.$(EXESUFFIX)

verify: ;

ka50.run: run

