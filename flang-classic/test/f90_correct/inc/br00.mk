#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test br00  ########


br00: run
	

build:  $(SRC)/br00.f
	-$(RM) br00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/br00.f -o br00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) br00.$(OBJX) check.$(OBJX) $(LIBS) -o br00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test br00
	br00.$(EXESUFFIX)

verify: ;

br00.run: run

