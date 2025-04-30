#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bf00  ########


bf00: run
	

build:  $(SRC)/bf00.f
	-$(RM) bf00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bf00.f -o bf00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bf00.$(OBJX) check.$(OBJX) $(LIBS) -o bf00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bf00
	bf00.$(EXESUFFIX)

verify: ;

bf00.run: run

